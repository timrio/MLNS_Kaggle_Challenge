import numpy as np
import os
import pandas as pd


def create_and_save_folds(initial_train_set, number_of_folds = 5, validation_size = 0.05): 

    # split between positive and negative
    positive_edges_df = initial_train_set.query('label==1')
    negative_edges_df = initial_train_set.query('label==0')

    positive_samples_for_validation = int(validation_size*positive_edges_df.shape[0])
    negative_samples_for_validation = int(validation_size*negative_edges_df.shape[0])

    # retrieve the the lines that will be used to create all the validation sets
    total_positive_validation_indexes = np.random.choice(list(positive_edges_df.index), positive_samples_for_validation*number_of_folds)
    total_negative_validation_indexes = np.random.choice(list(negative_edges_df.index), negative_samples_for_validation*number_of_folds)

    # create folds iteratively
    for i in range(number_of_folds):
        if os.path.isfile(f"Data/folds/train_set_{i+1}"):
            print(f"fold {i+1} already exists !")
            continue
        else:
            positive_validation_indexes = total_positive_validation_indexes[positive_samples_for_validation*i:positive_samples_for_validation*(i+1)]
            negative_validation_indexes = total_negative_validation_indexes[negative_samples_for_validation *i:negative_samples_for_validation *(i+1)]
            validation_set = pd.concat([positive_edges_df.loc[positive_validation_indexes], negative_edges_df.loc[negative_validation_indexes]], axis = 0).sample(frac=1)
            train_set = pd.concat([positive_edges_df.loc[set(positive_edges_df.index)-set(positive_validation_indexes)], negative_edges_df.loc[set(negative_edges_df.index)-set(negative_validation_indexes)]], axis = 0).sample(frac=1)

            validation_set.to_csv(f"Data/folds/validation_set_{i+1}", index = False)
            train_set.to_csv(f"Data/folds/train_set_{i+1}", index = False)
            print(f"fold_{i+1} created and saved !")

    return