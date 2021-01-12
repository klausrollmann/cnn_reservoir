import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg") # No display
import matplotlib.pyplot as plt

# import utils
from data_utils import *

# import the necessary packages from lib
#from utils.funcoes_imagem import show_images_side_by_side
from simplecnn2 import SimpleCNN

def train(path_df_dip_dsw, manual_train_path, manual_train_opts, opt_cfg):
    # Read model DIP and DSw
    df = pd.read_pickle(path_df_dip_dsw)
    df = df.set_index(['reference_time', 'region_index', 'simulation_index'])
    if manual_train_path != None:
        trainX, trainY = get_train_data(df, manual_train_path, manual_train_opts)

    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=1)
    print('Train data', trainX.shape, trainY.shape)
    print('Val data', valX.shape, valY.shape)

    ## Duplicate examples by adding y=[0,1] pairs
    trainX, trainY = add_reversed_pairs(trainX, trainY)
    valX, valY = add_reversed_pairs(valX, valY)

    print('Train data', trainX.shape, trainY.shape)
    print('Val data', valX.shape, valY.shape)

    train_size = trainY.shape[0]
    print('Training size', train_size)

    suffix = '_train_sz_{}_weights'.format(train_size)
    # checkpoint to save epoch with best validation accuracy
    checkpoint = ModelCheckpoint('runs/' + 'simplecnn2' + suffix + '.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    opt = SGD(lr=opt_cfg['INIT_LR'], decay=opt_cfg['INIT_LR']/opt_cfg['EPOCHS'])

    model = SimpleCNN.build(width=32, height=32, depth=3, classes=2)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    X, Y = trainX[:train_size], trainY[:train_size]
    H = model.fit(X, Y, validation_data=(valX, valY), batch_size=opt_cfg['BS'], callbacks=[checkpoint], epochs=opt_cfg['EPOCHS'])
    # Plot training & validation accuracy values
    with open('runs/history' + suffix + '.pickle', 'wb') as fp:
        pickle.dump(H.history, fp)

    return H


path_df_dip_dsw = 'data/dataset_single_region_dip_dsw.pkl'
manual_train_path = 'data/labeled_dataset.csv'
manual_train_opts = {'header':None, 'names':['time', 'region_index', 'good_model', 'bad_model']}
opt_cfg = {'BS':32, 'EPOCHS':60, 'INIT_LR':0.01}

train(path_df_dip_dsw, manual_train_path, manual_train_opts, opt_cfg)

