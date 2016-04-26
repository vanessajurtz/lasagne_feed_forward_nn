#!/usr/bin/env python

from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

import data_io_func

theano.config.floatX='float32'
# set a random numberinitialization seed to make code reproducible:
lasagne.random.set_rng(np.random.RandomState(seed=1)) # for lasagne
np.random.seed(seed=1) # for shuffling training examples

MAX_PEP_SEQ_LEN=9

# read in peptide sequences and targets:
X_train,y_train = data_io_func.read_pep("data/f000",MAX_PEP_SEQ_LEN)
X_val,y_val= data_io_func.read_pep("data/c000",MAX_PEP_SEQ_LEN)

# encode data using BLOSUM50:
X_train= data_io_func.encode_pep(X_train,MAX_PEP_SEQ_LEN)
y_train=np.array(y_train,dtype=theano.config.floatX)
X_val= data_io_func.encode_pep(X_val,MAX_PEP_SEQ_LEN)
y_val=np.array(y_val,dtype=theano.config.floatX)

# data dimensions now:
# (N_SEQS, SEQ_LENGTH, N_FEATURES)
print(X_train.shape)
print(X_val.shape)


def build_NN(max_pep_seq_len, n_features, n_hid):

    # input layer:
    l_in= lasagne.layers.InputLayer((None,max_pep_seq_len,n_features))

    # add hidden layer:
    l_hid = lasagne.layers.DenseLayer(
            l_in,
            num_units=n_hid,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.Normal())

    # output layer:
    l_out = lasagne.layers.DenseLayer(
            l_hid,
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.Normal())
    return l_out,l_in

# build a network with 10 hidden neurons:
N_FEATURES=21
N_HID=10
network,inp = build_NN(max_pep_seq_len=MAX_PEP_SEQ_LEN, n_features=N_FEATURES, n_hid=N_HID)

sym_target = T.vector('targets',dtype='float32')
sym_l_rate=T.scalar()

# TRAINING FUNCTION -----------------------------------------------------------

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction.flatten(), sym_target)
loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=sym_l_rate)

# training function:
train_fn = theano.function([inp.input_var, sym_target, sym_l_rate], loss, updates=updates)

# VALIDATION FUNCTION ----------------------------------------------------------

val_prediction = lasagne.layers.get_output(network, deterministic=True)
val_loss = lasagne.objectives.squared_error(val_prediction.flatten(),sym_target)
val_loss = val_loss.mean()

# validation function:
val_fn = theano.function([inp.input_var, sym_target], val_loss)


def iterate_minibatches(pep, targets, batchsize):
    assert pep.shape[0] == targets.shape[0]
    # shuffle:
    indices = np.arange(len(pep))
    np.random.shuffle(indices)
    for start_idx in range(0, len(pep) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield pep[excerpt],targets[excerpt]



# TRAINING LOOP ----------------------------------------------------------------

EPOCHS=range(1,200)
LEARNING_RATE=0.01
BATCH_SIZE=200

print("# Start training loop...")

start_time = time.time()

b_epoch=0
b_train_err=99
b_val_err=99

for e in EPOCHS:

    train_err = 0
    train_batches = 0
    val_err = 0
    val_batches = 0
    e_start_time = time.time()

    # shuffle training examples and iterate through minbatches:
    for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE):
        Xinp, target = batch
        train_err += train_fn(Xinp, target, LEARNING_RATE)
        train_batches += 1

    if e%10 == 0:
        # predict validation set:
        for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE):
            Xinp, target = batch
            val_err += val_fn(Xinp, target)
            val_batches += 1


        # save only best model:
        if (val_err/val_batches) < b_val_err:
            np.savez('params.npz', lasagne.layers.get_all_param_values(network))
            b_val_err = val_err/val_batches
            b_train_err = train_err/train_batches
            b_epoch = e
        # print performance:
        print("Epoch " + str(e) +
        "\ttraining error: " + str(round(train_err/train_batches, 4)) +
        "\tvalidation error: " + str(round(val_err/val_batches, 4)) +
        "\ttime: " + str(round(time.time()-e_start_time, 3)) + " s")

    else:
        # print performance:
        print("Epoch " + str(e) +
        "\ttraining error: " + str(round(train_err/train_batches, 4)) +
        "\ttime: " + str(round(time.time()-e_start_time, 3)) + " s")

# print best performance:
print("# Best epoch: " + str(b_epoch) +
        "\ttrain error: " + str(round(b_train_err, 4)) +
        "\tvalidation error: " + str(round(b_val_err, 4) ))
# report total time used for training:
print("# Time for training: " + str(round((time.time()-start_time)/60, 3)) + " min" )
print("# Done!")
