#!/usr/bin/env python

"""
Functions for data IO for neural network training.
"""

from __future__ import print_function
import argparse
import sys
import os
import time

from operator import add
import math
import numpy as np
from scipy.io import netcdf
import theano
import theano.tensor as T

import lasagne

theano.config.floatX='float32'


def read_pep(filename, MAX_PEP_SEQ_LEN):
    '''
    read AA seq of peptides and MHC molecule from text file

    parameters:
        - filename : file in which data is stored
    returns:
        - pep_aa : list of amino acid sequences of peptides (as string)
        - target : list of log transformed IC50 binding values
    '''
    pep_aa=[]
    target=[]
    infile = open(filename, "r")

    for l in infile:
        l=filter(None, l.strip().split())
        if len(l[0]) <= MAX_PEP_SEQ_LEN:
            pep_aa.append(l[0])
            target.append(l[1])
    infile.close()

    return pep_aa,target

def read_blosum_MN(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = {}
    B_idx = 99
    Z_idx = 99
    star_idx = 99

    for l in blosumfile:
        l = l.strip()

        if l[0] != '#':
            l= filter(None,l.strip().split(" "))

            if (l[0] == 'A') and (B_idx==99):
                B_idx = l.index('B')
                Z_idx = l.index('Z')
                star_idx = l.index('*')
            else:
                aa = str(l[0])
                if (aa != 'B') &  (aa != 'Z') & (aa != '*'):
                    tmp = l[1:len(l)]
                    # tmp = [float(i) for i in tmp]
                    # get rid of BJZ*:
                    tmp2 = []
                    for i in range(0, len(tmp)):
                        if (i != B_idx) &  (i != Z_idx) & (i != star_idx):
                            tmp2.append(float(tmp[i]))

                    #save in BLOSUM matrix
                    blosum[aa]=tmp2
    blosumfile.close()
    return(blosum)


def encode_pep(Xin, max_pep_seq_len):
    '''
    encode AA seq of peptides using BLOSUM50

    parameters:
        - Xin : list of peptide sequences in AA
    returns:
        - Xout : encoded peptide seuqneces (batch_size, max_pep_seq_len, n_features)
    '''
    # read encoding matrix:
    blosum = read_blosum_MN('data/BLOSUM50')
    n_features=len(blosum['A'])
    n_seqs=len(Xin)

    # make variable to store output:
    Xout = np.zeros((n_seqs, max_pep_seq_len, n_features),
                       dtype=theano.config.floatX)

    for i in range(0,len(Xin)):
        for j in range(0,len(Xin[i])):
            Xout[i, j, :n_features] = blosum[ Xin[i][j] ]
    return Xout

# modified from nntools:--------------------------------------------------------
def conv_seqs(X, length):
    '''
    Convert a list of matrices into np.ndarray

    parameters:
        - X : list of np.ndarray
            List of matrices
        - length : int
            Desired sequence length.  Smaller sequences will be padded with 0s,
            longer will be truncated.
        - batch_size : int
            Mini-batch size

    returns:
        - X_batch : np.ndarray
            Tensor of time series matrix batches,
            shape=(n_batches, batch_size, length, n_features)
    '''

    n_seqs = len(X)
    n_features = X[0].shape[1]

    X_pad = np.zeros((n_seqs, length, n_features),
                       dtype=theano.config.floatX)
    for i in range(0,len(X)):
        X_pad[i, :X[i].shape[0], :n_features] = X[i]
    return X_pad
