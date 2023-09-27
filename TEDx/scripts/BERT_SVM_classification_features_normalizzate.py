#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:54:26 2022

@author: eleonoracocciu
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

import spacy.cli
from spacy.language import Language
import math

from transformers import BertTokenizer, BertModel
import torch

from IPython.display import display

from transformers import logging
import torch.nn

import random

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV

from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pickle

def plot_decision_boundary(X_train, y_train, model, df_name, layer):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_train)
    y = y_train.values

    model.fit(X, y)
    plt.figure(figsize=(8, 5))
    fig = plot_decision_regions(X=X, y=y, clf=model, legend=2)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("{} test's - {}th layer decision boundary".format(df_name, layer))
    plt.legend(loc='best')
    plt.grid(False)
    #plt.show()
    plt.savefig(path_grafici + "{} test's - {}th layer decision boundary.png".format(df_name, layer))


def train_and_predict(layer, X_train, X_test, y_train, y_test, ids_train, ids_test):
    
    #https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities
    #https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
    
    model = SVC(kernel='rbf', gamma='auto', C=1.0)

    tic = time.time()
    clf = CalibratedClassifierCV(model, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    tac = time.time()
    print("clf calibrato", clf)
    print("Durata fit: {} secondi".format(tac-tic))
    print()
    
    tic = time.time()
    y_pred = clf.predict(X_test)
    tac = time.time()
    print("Durata predict: {} secondi".format(tac-tic))
    print()

    tic = time.time()
    y_score = clf.predict_proba(X_test)
    tac = time.time()
    print("Durata predict proba: {} secondi".format(tac-tic))
    
    chiave = 'clf_{}'.format(layer)
    diz_classifiers[chiave] = {}
    diz_classifiers[chiave]['accuracy'] = accuracy_score(y_test, y_pred)
    diz_classifiers[chiave]['f1_score'] = f1_score(y_test, y_pred, average=None)
    diz_classifiers[chiave]['y_test'] = y_test
    diz_classifiers[chiave]['y_pred'] = y_pred
    diz_classifiers[chiave]['y_score'] = y_score
    diz_classifiers[chiave]['classes'] = clf.classes_
    diz_classifiers[chiave]['ids_proba'] = {id_event:proba for id_event, proba in zip(ids_test, y_score)}
    diz_classifiers[chiave]['ids_preds'] = {id_event:pred for id_event, pred in zip(ids_test, y_pred)}
    diz_classifiers[chiave]['proba_of_positive_class'] = y_score[:, 1]

    #plot_decision_boundary(X_train, y_train, clf, df_name, layer)


def parse_arg():
    parser = argparse.ArgumentParser(description='SVM classification')
    
    parser.add_argument('-tr', '--training_set', type=str,
                        help='Training set')
    parser.add_argument('-ts', '--test_set', type=str,
                        help='Test set')
    parser.add_argument('-n', '--name', type=str,
                        help='Name of dataset')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arg()

    path_risorse = "../bert_features"
    path_salvataggio = "../pickle"
    path_grafici = "../grafici/"

    #leggo il file di features di BERT
    tr_set = args.training_set
    tst_set = args.test_set
    df_name = args.name

    training_to_read = open(path_risorse + "/" + tr_set, "rb")
    test_to_read = open(path_risorse + "/" + tst_set, "rb")

    df_training_bert_features_for_layers = pickle.load(training_to_read)
    print("Ho caricato il file di training {}".format(tr_set))

    df_test_bert_features_for_layers = pickle.load(test_to_read)
    print("Ho caricato il file di test {}".format(tst_set))

    #classificazione con SVM
    diz_classifiers = {}

    #for layer, i in zip(df_bert_features_for_layers['{}'.format(df_name)], enumerate(tqdm(list(range(len(df_bert_features_for_layers['{}'.format(df_name)])))))):
    for layer in tqdm(list(range(1,13))):

        #in ogni chiave (livello) ho id, features e classe di training e test

        #creo training e test con le features (vettori di BERT) per ogni specifico livello
        ids_train = pd.DataFrame([i[0] for i in df_training_bert_features_for_layers["{}_training".format(df_name)]["Layer_{}".format(layer)]], columns = ["Id"])["Id"]
        X_train = pd.DataFrame([i[1] for i in df_training_bert_features_for_layers["{}_training".format(df_name)]["Layer_{}".format(layer)]])
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train))
        y_train = pd.DataFrame([i[2] for i in df_training_bert_features_for_layers["{}_training".format(df_name)]["Layer_{}".format(layer)]], columns = ["Class"])['Class']

        ids_test = pd.DataFrame([i[0] for i in df_test_bert_features_for_layers["{}_test".format(df_name)]["Layer_{}".format(layer)]], columns = ["Id"])["Id"]
        X_test = pd.DataFrame([i[1] for i in df_test_bert_features_for_layers["{}_test".format(df_name)]["Layer_{}".format(layer)]])
        X_test_scaled = pd.DataFrame(scaler.transform(X_test))
        y_test = pd.DataFrame([i[2] for i in df_test_bert_features_for_layers["{}_test".format(df_name)]["Layer_{}".format(layer)]], columns = ["Class"])['Class']

        print("Train shape: {}".format(X_train.shape))
        print("Test shape: {}".format(X_test.shape))
        print()
        print("Train e predict per il livello {}".format(layer))
        train_and_predict(layer, X_train_scaled, X_test_scaled, y_train, y_test, ids_train, ids_test)

    diz_classifiers_df = {}
    diz_classifiers_df[df_name] = diz_classifiers
    
    with open(path_salvataggio+'/'+'diz_classifiers_bert_svm_scaled_{}.pickle'.format(df_name), 'wb') as handle:
	    pickle.dump(diz_classifiers_df, handle)


