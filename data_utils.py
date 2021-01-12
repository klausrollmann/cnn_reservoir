import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from cnn_utils import get_hybrid_image
#from funcoes_imagem import show_images_side_by_side

def add_reversed_pairs(X, Y):
    allX = []
    allY = []
    for x, y in zip(X, Y):
        rev_x = np.zeros_like(x)
        rev_x[:,:,0] = x[:,:,0]
        rev_x[:,:,1] = x[:,:,2]
        rev_x[:,:,2] = x[:,:,1]
        rev_y = y[::-1] 
        allX.append(x)
        allX.append(rev_x)
        allY.append(y)
        allY.append(rev_y)

    return np.array(allX), np.array(allY)

def get_ground_truth_specialists(df, answers_path, answers_info, size=(32,32)):
    test = pd.read_csv(answers_path, **answers_info)

    features=[]
    labels = []
    ## Train data from manual labeling
    for i, row in test.iterrows():
        time = row['time']
        region_index = row['region_index']
        good_model = row['good_model_index']
        bad_model = row['bad_model_index']

        reference = df.at[(time, region_index, good_model), 'reference_image']

        domain = 'dsw'
        good_model_image = df.at[(time, region_index, good_model), 'simulation_image_' + domain]
        bad_model_image = df.at[(time, region_index, bad_model), 'simulation_image_' + domain]

        # Label 1, 0 (first is the good model selected)
        hybrid_image = get_hybrid_image(reference, good_model_image, bad_model_image, size=size)
        features.append(hybrid_image)
        labels.append([1,0])

    testX = np.array(features)
    testY = np.array(labels)
    return testX, testY

def get_train_data_agreement(df, manual_train_path, manual_train_info, size=(32,32), vote_diff=2):
    train = pd.read_csv(manual_train_path, **manual_train_info)

    features=[]
    labels = []

    ## Train data from manual labeling
    for i, row in train.iterrows():
        time = row['time']
        region_index = int(row['region_index'])
        votes = int(row['votes_a']), int(row['votes_b'])
        #if votes == (1,0) or votes == (2,0):
        if votes[0] - votes[1] >= vote_diff:
            good_model = int(row['model_a'])
            bad_model = int(row['model_b'])
        #elif votes == (0,1) or votes == (0,2):
        elif votes[1] - votes[0] >= vote_diff:
            good_model = int(row['model_b'])
            bad_model = int(row['model_a'])
        else:
            continue
        #else:
        #    good_model = int(row['model_a'])
        #    bad_model = int(row['model_b'])
        #    labels.append(votes)

        reference = df.at[(time, region_index, good_model), 'reference_image']

        domain = 'dsw'
        good_model_image = df.at[(time, region_index, good_model), 'simulation_image_' + domain]
        bad_model_image = df.at[(time, region_index, bad_model), 'simulation_image_' + domain]

        # Label 1, 0
        hybrid_image = get_hybrid_image(reference, good_model_image, bad_model_image, size=size)
        features.append(hybrid_image)
        labels.append([1,0])

    trainX = np.array(features)
    trainY = np.array(labels)
    return trainX, trainY

def get_train_data(df, manual_train_path, manual_train_info, size=(32,32), agreement=False):
    train = pd.read_csv(manual_train_path, **manual_train_info)

    features=[]
    labels = []
    ## Train data from manual labeling
    for i, row in train.iterrows():
        time = row['time']
        region_index = row['region_index']
        good_model = row['good_model']
        bad_model = row['bad_model']

        reference = df.at[(time, region_index, good_model), 'reference_image']

        domain = 'dsw'
        good_model_image = df.at[(time, region_index, good_model), 'simulation_image_' + domain]
        bad_model_image = df.at[(time, region_index, bad_model), 'simulation_image_' + domain]

        # Label 1, 0
        hybrid_image = get_hybrid_image(reference, good_model_image, bad_model_image, size=size)
        features.append(hybrid_image)
        labels.append([1,0])

    trainX = np.array(features)
    trainY = np.array(labels)
    return trainX, trainY

# Augment data and generate a new batch with variations in the image
def get_batch_transformations(X, Y, batch_size):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.0,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    return datagen.flow(X, Y, batch_size)
