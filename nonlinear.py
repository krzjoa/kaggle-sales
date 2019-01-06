import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

SALES_TRAIN_V2 = "data/sales_train_v2_fv2.csv"
SAMPLE_SUBMISSION = "data/sample_submission.csv"
TEST = "data/test_fv2.csv"
OUTPUT = "output"
FEATURES = [
    'shop_id',
    'item_id',
    'total_cat_cnt',
    'min_cat_cnt',
    'max_cat_cnt',
    'mean_cat_cnt',
    'std_cat_cnt',
    'min_cat_price',
    'max_cat_price',
    'mean_cat_price',
    'std_cat_price',
    'total_shop_cnt',
    'min_shop_cnt',
    'max_shop_cnt',
    'mean_shop_cnt',
    'std_shop_cnt',
    'min_shop_price',
    'max_shop_price',
    'mean_shop_price',
    'std_shop_price'
]

# Dev dataset
sales_train = pd.read_csv(SALES_TRAIN_V2)

# Test & sample
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
test = pd.read_csv(TEST)

def rmse(y_true, y_pred):
    '''
    Root mean squared error.
    Parameters
    ----------
    y_true: ndarray
        Ground truth
    y_pred: ndarray
        Array of predictions
    Returns
    -------
    rmsle: float
        Root mean squared error
    References
    ----------
    .. [1] https://www.kaggle.com/wiki/RootMeanSquaredError
    '''

    # Check shapes
    #y_true, y_pred = align_shape(y_true, y_pred)
    return np.sqrt(((y_true - y_pred)**2).mean())

# Generate cross validation split
# Here we generate random split and we don't care about temporal order
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

folds = {}
fold_num = 1

for train_ids, test_ids in kfold.split(range(len(sales_train))):
    train_ids, val_ids = train_test_split(train_ids, test_size=.2)
    folds[fold_num] = {"train": list(train_ids),
                      "test": list(test_ids),
                      "val": len(val_ids)}
    fold_num += 1


import json

with open("splits/unordered_folds.json", "w") as f:
    json.dump(folds, f)


def get_training_dataset(idx, sales_train):
    X = sales_train[FEATURES].loc[idx].values
    y = sales_train['item_cnt_day'].loc[idx].values
    return X, y

### TF

print "Starting TF model..."

X  = tf.placeholder(tf.float32, [None, 20])
y = tf.placeholder(tf.float32, [None, ])

with tf.device("/gpu:0"):
    net = tf.layers.dense(X, 30, activation=tf.nn.elu)
    net = tf.layers.dense(net, 5, activation=tf.nn.elu)
    net = tf.layers.dense(net, 1, activation=tf.identity)
    net = tf.clip_by_value(net, -20, 100)

# Loss function
loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((y - net)**2)))

opt = tf.train.AdamOptimizer()
opt_op = opt.minimize(loss)

init = tf.global_variables_initializer()

import tqdm
def batch_iterator(idx, batch_size):
    ln = len(idx)
    for i in range(0, ln, batch_size):
        yield idx[i:i + batch_size]

import time

N_EPOCHS = 25
BATCH_SIZE = 5000

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)

    fold_results = []

    for fold in folds:
        print "Fold: ", fold
        train_idx = folds[fold]['train']
        val_idx = folds[fold]['val']
        test_idx = folds[fold]['test']

        train_loss_history = []
        val_loss_history = []

        for n in range(N_EPOCHS):
            start = time.time()
            train_batch_losses = []
            # val_batch_losses = []

            # TRAINING
            for train_batch_idx in batch_iterator(train_idx, BATCH_SIZE):
                X_train, y_train = get_training_dataset(train_batch_idx, sales_train)
                # print X_train.shape, y_train.shape
                y_pred, train_loss_value, _ = sess.run([net, loss, opt_op], feed_dict={X: X_train, y: y_train})
                train_batch_losses.append(train_loss_value)

            # VALIDATION
            X_test, y_test = get_training_dataset(train_batch_idx, sales_train)
            y_pred, val_epoch_loss, = sess.run([net, loss], feed_dict={X: X_test, y: y_test})
            # val_batch_losses.append(test_loss_value)

            end = time.time()

            train_epoch_loss = np.mean(train_batch_losses)
            # val_epoch_loss = np.mean(val_batch_losses)

            train_loss_history.append(train_epoch_loss)
            val_loss_history.append(val_epoch_loss)

            print "Epoch: {} | {} s| train. loss: {} | val. loss: {}".format(n + 1, end - start, train_epoch_loss,
                                                                             val_epoch_loss)
        # TESTING
        X_test, y_test = get_training_dataset(train_batch_idx, sales_train)
        y_pred, test_loss_value, = sess.run([net, loss], feed_dict={X: X_test, y: y_test})


import pdb
pdb.set_trace()