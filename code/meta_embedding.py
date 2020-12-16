import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os, pickle

# specify the GPU device
# os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Dense
import keras
from keras.preprocessing.sequence import pad_sequences

'''
Config
'''
# batch size per iteration
BATCHSIZE = 200
# mini-batch size for few-shot learning
MINIBATCHSIZE = 20 
# learning rate
LR = 1e-3 
# coefficient to balance `cold-start' and `warm-up'
ALPHA = 0.1
# length of embedding vectors
EMB_SIZE = 128
# model
MODEL = 'deepFM'
# log file
LOG = "logs/{}.csv".format(MODEL)
# path to save the model
saver_path ="saver/model-"+LOG.split("/")[-1][:-4]

def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

# training data of big ads
train = read_pkl("../data/big_train_main.pkl")
# some pre-processing
num_words_dict = {
    'MovieID': 4000,
    'UserID': 6050,
    'Age': 7,
    'Gender': 2,
    'Occupation': 21,
    'Year': 83,
}
ID_col = 'MovieID'
item_col = ['Year']
context_col = ['Age', 'Gender', 'Occupation', 'UserID']
train_y = train['y']
train_x = train[[ID_col]+item_col+context_col]
train_t = pad_sequences(train.Title, maxlen=8)
train_g = pad_sequences(train.Genres, maxlen=4)

# few-shot data for the small ads
test_a = read_pkl("../data/test_oneshot_a.pkl")
test_b = read_pkl("../data/test_oneshot_b.pkl")
test_c = read_pkl("../data/test_oneshot_c.pkl")
test_test = read_pkl("../data/test_test.pkl")

test_x_a = test_a[[ID_col]+item_col+context_col]
test_y_a = test_a['y'].values
test_t_a = pad_sequences(test_a.Title, maxlen=8)
test_g_a = pad_sequences(test_a.Genres, maxlen=4)

test_x_b = test_b[[ID_col]+item_col+context_col]
test_y_b = test_b['y'].values
test_t_b = pad_sequences(test_b.Title, maxlen=8)
test_g_b = pad_sequences(test_b.Genres, maxlen=4)

test_x_c = test_c[[ID_col]+item_col+context_col]
test_y_c = test_c['y'].values
test_t_c = pad_sequences(test_c.Title, maxlen=8)
test_g_c = pad_sequences(test_c.Genres, maxlen=4)

test_x_test = test_test[[ID_col]+item_col+context_col]
test_y_test = test_test['y'].values
test_t_test = pad_sequences(test_test.Title, maxlen=8)
test_g_test = pad_sequences(test_test.Genres, maxlen=4)

class Meta_Model(object):
    def __init__(self, ID_col, item_col, context_col, nb_words, model='FM',
                 emb_size=128, alpha=0.1,
                 warm_lr=1e-3, cold_lr=1e-4, ME_lr=1e-3):
        """
        ID_col: string, the column name of the item ID
        item_col: list, the columns of item features
        context_col: list, the columns of other features
        nb_words: dict, nb of words in each of these columns
        """
        columns = [ID_col] + item_col + context_col
        def get_embeddings():
            inputs, tables = {}, []
            item_embs, other_embs = [], []
            for col in columns:
                inputs[col] = tf.compat.v1.placeholder(tf.int32, [None])
                table = tf.get_variable(
                    "table_{}".format(col), [nb_words[col], emb_size],
                    initializer=tf.random_normal_initializer(stddev=0.01))
                emb = tf.nn.embedding_lookup(table, inputs[col])
                if col==ID_col:
                    ID_emb = emb
                    ID_table = table
                elif col in item_col:
                    item_embs.append(emb)
                else:
                    other_embs.append(emb)

            inputs["title"] = tf.compat.v1.placeholder(tf.int32, [None, 8])
            inputs["genres"] = tf.compat.v1.placeholder(tf.int32, [None, 4])

            title_emb = tf.contrib.layers.embed_sequence(
                inputs["title"], 20001, emb_size, scope="word_emb")
            genre_emb = tf.contrib.layers.embed_sequence(
                inputs["genres"], 21, emb_size, scope="genre_table")
            item_embs.append(tf.reduce_mean(title_emb, axis=1))
            item_embs.append(tf.reduce_mean(genre_emb, axis=1))
            
            return inputs, ID_emb, item_embs, other_embs, ID_table
        
        def generate_meta_emb(item_embs):
            """
            This is the simplest architecture of the embedding generator,
            with only a dense layer.
            You can customize it if you want have a stronger performance, 
            for example, you can add an l2 regularization term or alter 
            the pooling layer. 
            """
            embs = tf.stop_gradient(tf.stack(item_embs, 1))
            item_h = tf.layers.flatten(embs)
            emb_pred_Dense = tf.layers.Dense(
                emb_size, activation=tf.nn.tanh, use_bias=False,
                name='emb_predictor') 
            emb_pred = emb_pred_Dense(item_h) / 5.
            ME_vars = emb_pred_Dense.trainable_variables
            return emb_pred, ME_vars

        def get_yhat_deepFM(ID_emb, item_embs, other_embs, **kwargs):
            embeddings = [ID_emb] + item_embs + other_embs
            sum_of_emb = tf.add_n(embeddings)
            diff_of_emb = [sum_of_emb - x for x in embeddings]
            dot_of_emb = [tf.reduce_sum(embeddings[i]*diff_of_emb[i], 
                                        axis=1, keepdims=True) 
                          for i in range(len(columns))]
            h = tf.concat(dot_of_emb, 1)
            h2 = tf.concat(embeddings, 1)
            for i in range(2):
                h2 = tf.nn.relu(tf.layers.dense(h2, emb_size, name='deep-{}'.format(i)))
            h = tf.concat([h,h2], 1)
            y = tf.nn.sigmoid(tf.layers.dense(h, 1, name='out'))
            return y
        def get_yhat_PNN(ID_emb, item_embs, other_embs, **kwargs):
            embeddings = [ID_emb] + item_embs + other_embs
            sum_of_emb = tf.add_n(embeddings)
            diff_of_emb = [sum_of_emb - x for x in embeddings]
            dot_of_emb = [tf.reduce_sum(embeddings[i]*diff_of_emb[i], 
                                        axis=1, keepdims=True)
                          for i in range(len(columns))]
            dots = tf.concat(dot_of_emb, 1)
            h2 = tf.concat(embeddings, 1)
            h = tf.concat([dots,h2], 1)
            w = tf.get_variable('MLP_1/kernel', shape=(h.shape[1],emb_size))
            b = tf.get_variable('MLP_1/bias', shape=(emb_size,), 
                                initializer=tf.initializers.zeros)
            h = tf.nn.relu(tf.matmul(h,w)+b)
            w = tf.get_variable('MLP_2/kernel', shape=(h.shape[1],1))
            b = tf.get_variable('MLP_2/bias', shape=(1,), 
                                initializer=tf.initializers.constant(0.))
            y = tf.nn.sigmoid(tf.matmul(h,w)+b)
            return y
        '''
        *CHOOSE THE BASE MODEL HERE*
        '''
        get_yhat = {
            "PNN": get_yhat_PNN, 
            "deepFM": get_yhat_deepFM
        }[model]
        
        with tf.compat.v1.variable_scope("model"):
            # build the base model
            inputs, ID_emb, item_embs, other_embs, ID_table = get_embeddings()
            label = tf.compat.v1.placeholder(tf.float32, [None, 1])
            # outputs and losses of the base model
            yhat = get_yhat(ID_emb, item_embs, other_embs)
            warm_loss = tf.losses.log_loss(label, yhat)
            # Meta-Embedding: build the embedding generator
            meta_ID_emb, ME_vars = generate_meta_emb(item_embs)

        with tf.compat.v1.variable_scope("model", reuse=True):
            # Meta-Embedding: step 1, cold-start, 
            #     use the generated meta-embedding to make predictions
            #     and calculate the cold-start loss_a
            cold_yhat_a = get_yhat(meta_ID_emb, item_embs, other_embs)
            cold_loss_a = tf.losses.log_loss(label, cold_yhat_a)
            # Meta-Embedding: step 2, apply gradient descent once
            #     get the adapted embedding
            cold_emb_grads = tf.gradients(cold_loss_a, meta_ID_emb)[0]
            meta_ID_emb_new = meta_ID_emb - cold_lr * cold_emb_grads
            # Meta-Embedding: step 3, 
            #     use the adapted embedding to make prediction on another mini-batch 
            #     and calculate the warm-up loss_b
            inputs_b, _, item_embs_b, other_embs_b, _ = get_embeddings()
            label_b = tf.compat.v1.placeholder(tf.float32, [None, 1])
            cold_yhat_b = get_yhat(meta_ID_emb_new, item_embs_b, other_embs_b)
            cold_loss_b = tf.losses.log_loss(label_b, cold_yhat_b)            
        
        # build the optimizer and update op for the original model
        warm_optimizer = tf.train.AdamOptimizer(warm_lr)
        warm_update_op = warm_optimizer.minimize(warm_loss)
        warm_update_emb_op = warm_optimizer.minimize(warm_loss, var_list=[ID_table])
        # build the optimizer and update op for meta-embedding
        # Meta-Embedding: step 4, calculate the final meta-loss
        ME_loss = cold_loss_a * alpha + cold_loss_b * (1-alpha)
        ME_optimizer = tf.train.AdamOptimizer(ME_lr)
        ME_update_op = ME_optimizer.minimize(ME_loss, var_list=ME_vars)
        
        ID_table_new = tf.compat.v1.placeholder(tf.float32, ID_table.shape)
        ME_assign_op = tf.assign(ID_table, ID_table_new)
        
        def predict_warm(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(yhat, feed_dict)
        def predict_ME(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(cold_yhat_a, feed_dict)
        def get_meta_embedding(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(meta_ID_emb, feed_dict)
        def assign_meta_embedding(sess, ID, emb):
            # take the embedding matrix
            table = sess.run(ID_table)
            # replace the ID^th row by the new embedding
            table[ID, :] = emb
            return sess.run(ME_assign_op, feed_dict={ID_table_new: table})
        def train_warm(sess, X, Title, Genres, y, embedding_only=False):
            # original training on batch
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            feed_dict[label] = y.reshape((-1,1))
            return sess.run([
                warm_loss, warm_update_emb_op if embedding_only else warm_update_op 
            ], feed_dict=feed_dict)
        def train_ME(sess, X, Title, Genres, y, 
                     X_b, Title_b, Genres_b, y_b):
            # train the embedding generator
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            feed_dict[label] = y.reshape((-1,1))
            feed_dict_b = {inputs_b[col]: X_b[col] for col in columns}
            feed_dict_b = {inputs_b["title"]: Title_b,
                           inputs_b["genres"]: Genres_b,
                           **feed_dict_b}
            feed_dict_b[label_b] = y_b.reshape((-1,1))
            return sess.run([
                cold_loss_a, cold_loss_b, ME_update_op
            ], feed_dict={**feed_dict, **feed_dict_b})
        self.predict_warm = predict_warm
        self.predict_ME = predict_ME
        self.train_warm = train_warm
        self.train_ME = train_ME
        self.get_meta_embedding = get_meta_embedding
        self.assign_meta_embedding = assign_meta_embedding

model = Meta_Model(ID_col, item_col, context_col, num_words_dict, model=MODEL,
                   emb_size=EMB_SIZE, alpha=ALPHA,
                   warm_lr=LR, cold_lr=LR/10., ME_lr=LR)
sys.exit(0)

def predict_on_batch(sess, predict_func, test_x, test_t, test_g, batchsize=800):
    n_samples_test = test_x.shape[0]
    n_batch_test = n_samples_test//batchsize
    test_pred = np.zeros(n_samples_test)
    for i_batch in range(n_batch_test):
        batch_x = test_x.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_t = test_t[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_g = test_g[i_batch*batchsize:(i_batch+1)*batchsize]
        _pred = predict_func(sess, batch_x, batch_t, batch_g)
        test_pred[i_batch*batchsize:(i_batch+1)*batchsize] = _pred.reshape(-1)
    if n_batch_test*batchsize<n_samples_test:
        batch_x = test_x.iloc[n_batch_test*batchsize:]
        batch_t = test_t[n_batch_test*batchsize:]
        batch_g = test_g[n_batch_test*batchsize:]
        _pred = predict_func(sess, batch_x, batch_t, batch_g)
        test_pred[n_batch_test*batchsize:] = _pred.reshape(-1)
    return test_pred

"""
Pre-train the base model
"""
batchsize = BATCHSIZE

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

n_samples = train_x.shape[0]
n_batch = n_samples//batchsize

for i_batch in tqdm(range(n_batch)):
    batch_x = train_x.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
    batch_t = train_t[i_batch*batchsize:(i_batch+1)*batchsize]
    batch_g = train_g[i_batch*batchsize:(i_batch+1)*batchsize]
    batch_y = train_y.iloc[i_batch*batchsize:(i_batch+1)*batchsize].values
    loss, _ = model.train_warm(sess, batch_x, batch_t, batch_g, batch_y)

test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_base_cold = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[pre-train]\n\ttest-test loss: {:.6f}".format(test_loss_test))
auc_base_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[pre-train]\n\ttest-test auc: {:.6f}".format(test_auc_test))
save_path = saver.save(sess, saver_path)
print("Model saved in path: %s" % save_path)

minibatchsize = MINIBATCHSIZE
batch_n_ID = 25
batchsize = minibatchsize*batch_n_ID
n_epoch = 3

'''
Train the Meta-Embedding generator
'''
best_auc = 0
best_loss = 10
for i_epoch in range(n_epoch):
    # Read the few-shot training data of big ads
    if i_epoch==0:
        _train_a = read_pkl("../data/train_oneshot_a.pkl")
        _train_b = read_pkl("../data/train_oneshot_b.pkl")
    elif i_epoch==1:
        _train_a = read_pkl("../data/train_oneshot_c.pkl")
        _train_b = read_pkl("../data/train_oneshot_d.pkl")
    elif i_epoch==2:
        _train_a = read_pkl("../data/train_oneshot_b.pkl")
        _train_b = read_pkl("../data/train_oneshot_c.pkl")
    elif i_epoch==3:
        _train_a = read_pkl("../data/train_oneshot_d.pkl")
        _train_b = read_pkl("../data/train_oneshot_a.pkl")
    train_x_a = _train_a[[ID_col]+item_col+context_col]
    train_y_a = _train_a['y'].values
    train_t_a = pad_sequences(_train_a.Title, maxlen=8)
    train_g_a = pad_sequences(_train_a.Genres, maxlen=4)

    train_x_b = _train_b[[ID_col]+item_col+context_col]
    train_y_b = _train_b['y'].values
    train_t_b = pad_sequences(_train_b.Title, maxlen=8)
    train_g_b = pad_sequences(_train_b.Genres, maxlen=4)
    
    n_samples = train_x_a.shape[0]
    n_batch = n_samples//batchsize
    # Start training
    for i_batch in tqdm(range(n_batch)):
        batch_x_a = train_x_a.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_t_a = train_t_a[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_g_a = train_g_a[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_y_a = train_y_a[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_x_b = train_x_b.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_t_b = train_t_b[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_g_b = train_g_b[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_y_b = train_y_b[i_batch*batchsize:(i_batch+1)*batchsize]
        loss_a, loss_b, _ = model.train_ME(sess, 
                                           batch_x_a, batch_t_a, batch_g_a, batch_y_a, 
                                           batch_x_b, batch_t_b, batch_g_b, batch_y_b, )
    # on epoch end
    test_pred_test = predict_on_batch(sess, model.predict_ME, 
                                      test_x_test, test_t_test, test_g_test)
    logloss_ME_cold = test_loss_test = log_loss(test_y_test, test_pred_test)
    print("[Meta-Embedding]\n\ttest-test loss: {:.6f}".format(test_loss_test))
    auc_ME_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
    print("[Meta-Embedding]\n\ttest-test auc: {:.6f}".format(test_auc_test))

save_path = saver.save(sess, saver_path)
print("Model saved in path: %s" % save_path)

print("COLD-START BASELINE:")
print("\t Loss: {:.4f}".format(logloss_base_cold))
print("\t AUC: {:.4f}".format(auc_base_cold))
'''
Testing
'''
minibatchsize = MINIBATCHSIZE
batch_n_ID = 25
batchsize = minibatchsize * batch_n_ID
i = 1
test_n_ID = len(test_x_c[ID_col].drop_duplicates())
saver.restore(sess, save_path)
for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_a[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_a[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_a[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_a[i*batchsize:(i+1)*batchsize]
    model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_base_batcha = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_base_batcha = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))

for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_b[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_b[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_b[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_b[i*batchsize:(i+1)*batchsize]
    model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_base_batchb = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_base_batchb = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))
for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_c[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_c[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_c[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_c[i*batchsize:(i+1)*batchsize]
    model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_base_batchc = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_base_batchc = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[baseline]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))
print("="*60)

saver.restore(sess, save_path)

for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_a[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_a[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_a[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_a[i*batchsize:(i+1)*batchsize]
    aid = np.unique(batch_x[ID_col].values)
    for k in range(batch_n_ID):
        if k*minibatchsize>=len(batch_x):
            break
        ID = batch_x[ID_col].values[k*minibatchsize]
        embeddings = model.get_meta_embedding(
            sess, batch_x[k*minibatchsize:(k+1)*minibatchsize],
            batch_t[k*minibatchsize:(k+1)*minibatchsize],
            batch_g[k*minibatchsize:(k+1)*minibatchsize],
        )
        emb = embeddings.mean(0)
        model.assign_meta_embedding(sess, ID, emb)
    model.train_warm(sess, 
                     batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_ME_batcha = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_ME_batcha = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))

for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_b[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_b[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_b[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_b[i*batchsize:(i+1)*batchsize]
    model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_ME_batchb = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_ME_batchb = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))
for i in tqdm(range(int(np.ceil(test_n_ID/batch_n_ID)))):
    batch_x = test_x_c[i*batchsize:(i+1)*batchsize]
    batch_t = test_t_c[i*batchsize:(i+1)*batchsize]
    batch_g = test_g_c[i*batchsize:(i+1)*batchsize]
    batch_y = test_y_c[i*batchsize:(i+1)*batchsize]
    model.train_warm(sess, batch_x, batch_t, batch_g, batch_y, 
                     embedding_only=True)
test_pred_test = predict_on_batch(sess, model.predict_warm, 
                                  test_x_test, test_t_test, test_g_test)
logloss_ME_batchc = test_loss_test = log_loss(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
    test_loss_test, 1-test_loss_test/logloss_base_cold))
auc_ME_batchc = test_auc_test = roc_auc_score(test_y_test, test_pred_test)
print("[Meta-Embedding]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
    test_auc_test, test_auc_test/auc_base_cold-1))

# write the scores into file.
res = [logloss_base_cold, logloss_ME_cold, 
       logloss_base_batcha, logloss_ME_batcha, 
       logloss_base_batchb, logloss_ME_batchb, 
       logloss_base_batchc, logloss_ME_batchc, 
       auc_base_cold, auc_ME_cold, 
       auc_base_batcha, auc_ME_batcha, 
       auc_base_batchb, auc_ME_batchb, 
       auc_base_batchc, auc_ME_batchc]
with open(LOG, "a") as logfile:
    logfile.writelines(",".join([str(x) for x in res])+"\n")
