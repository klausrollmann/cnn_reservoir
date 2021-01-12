import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model, load_model
import keras.layers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import json
import pandas as pd

import matplotlib
matplotlib.use("Agg") # No display
import matplotlib.pyplot as plt

# import utils
from data_utils import *

# import the necessary packages from lib
import sys
#sys.path.append('../../')
#from utils.funcoes_imagem import show_images_side_by_side
from networks.simplecnn2_siamese import SimpleCNN, SimpleCNNTriplet
import glob

path_df_dip_dsw = 'data/dataset_single_region_dip_dsw.pkl'
answers_path = 'data/test_data.csv'
answers_opts = {'header':0, 'names':['time', 'region_index', 'good_model_index', 'bad_model_index']}
opt_cfg = {'BS':32, 'EPOCHS':60, 'INIT_LR':0.01}
#models_path_glob = 'models/models_sizes/simplecnn2_train_sz_*_run_*_weights.h5'
models_path_glob = 'models/simplecnn2_train_all_weights.h5'

def test(path_df_dip_dsw, answers_path, answers_opts, models_path_glob, opt_cfg):
    # Read model DIP and DSw
    df = pd.read_pickle(path_df_dip_dsw)
    df = df.set_index(['reference_time', 'region_index', 'simulation_index'])
    testX, testY = get_ground_truth_specialists(df, answers_path, answers_opts)

    model = SimpleCNNTriplet.build(32,32,1)
    print(model.summary())

    results = {}
    for weights_path in glob.glob(models_path_glob):
        model_name = weights_path.split('_weights')[0]
        print('Testing:', model_name)
        # Load weights
        model.load_weights(weights_path)

        # Get single network (embedding network to generate representation given image as input)
        embedding = Model(inputs=model.get_layer('embedding').get_input_at(0), outputs=model.get_layer('embedding').get_output_at(0))
        #print(embedding.summary())

        correct = 0
        for i in range(testY.size):
            r = embedding.predict(testX['ref'][i].reshape(1,32,32,1))
            a = embedding.predict(testX['a'][i].reshape(1,32,32,1))
            b = embedding.predict(testX['b'][i].reshape(1,32,32,1))

            if np.sum((r-a)**2) < np.sum((r-b)**2) :
                correct += 1

        results[model_name] = correct/testY.shape[0]
        print('Acc {} : {}'.format(model_name.split('/')[1], correct/testY.shape[0]))
        exit()

    # Write JSON file
    with open('accuracy_train_sizes.json', 'w') as f:
        json.dump(results, f)

## Test
test(path_df_dip_dsw, answers_path, answers_opts, models_path_glob, opt_cfg)
