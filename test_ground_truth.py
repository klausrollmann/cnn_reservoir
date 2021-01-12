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

from sacred import Experiment
from sacred.observers import FileStorageObserver
ex = Experiment('evaluate_ground_truth_specialists')
ex.observers.append(FileStorageObserver.create('runs'))

# import utils
from data_utils import *

# import the necessary packages from lib
import sys
sys.path.append('../../')
#from utils.funcoes_imagem import show_images_side_by_side
from networks.simplecnn2 import SimpleCNN

@ex.config
def cfg():
    path_df_dip_dsw = '../../dataset/dataset_single_region_dip_dsw.pkl'
    answers_path = '../../dataset/answers_agree2_specialists.csv'
    #answers_path = '../../dataset/answers_agree5_all.csv'
    answers_opts = {'header':0, 'names':['time', 'region_index', 'good_model_index', 'bad_model_index']}
    opt_cfg = {'BS':32, 'EPOCHS':60, 'INIT_LR':0.01}

@ex.automain
def test(path_df_dip_dsw, answers_path, answers_opts, opt_cfg):
    # Read model DIP and DSw
    df = pd.read_pickle(path_df_dip_dsw)
    df = df.set_index(['reference_time', 'region_index', 'simulation_index'])
    testX, testY = get_ground_truth_specialists(df, answers_path, answers_opts)
    testX, testY = add_reversed_pairs(testX, testY)

    opt = SGD(lr=opt_cfg['INIT_LR'], decay=opt_cfg['INIT_LR']/opt_cfg['EPOCHS'])
    model = SimpleCNN.build(width=32, height=32, depth=3, classes=2)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    for trained_samples in list(range(1000, 2501, 500)) + [2896]:
        model.load_weights('runs/simplecnn2_train_sz_{}_weights.h5'.format(trained_samples))
        print('Samples:', trained_samples)
        print('Evaluate:', dict(zip(model.metrics_names, model.evaluate(testX, testY))))
