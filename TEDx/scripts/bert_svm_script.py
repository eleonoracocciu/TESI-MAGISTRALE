import json
import os
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from docx import Document
from docx.shared import Pt
import pickle
import os
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.common.exceptions import StaleElementReferenceException
from tqdm import tqdm
import time
from langdetect import detect
from selenium.common.exceptions import NoSuchElementException
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry_convert as pc
import pycountry
import collections
from IPython.display import display

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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import pickle


diz_classifiers = {}

def train_and_predict(original_features, df, layer):
        
    labels = df['Class']
    
    #facciamo lo split di train e test stratificando la classe, in modo da avere un numero equo di 0 e 1 anche nel test
    X_train, X_test, y_train, y_test = train_test_split(original_features, labels, test_size=0.2, stratify=labels)
    
    #GridSearchCV per il fine tuning dei parametri della SVM
    model = SVC()
    
    kernel = ['poly', 'rbf', 'sigmoid']
    C = [0.001, 0.01, 1.0, 10.0, 50.0, 100.0]
    gamma = ['scale']
    tol = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    class_weight = [None, 'balanced']
    
    param_grid = dict(kernel=kernel, 
                C=C, 
                gamma=gamma,
                tol=tol,
                class_weight=class_weight)
    
    #refit default=True: Refit an estimator using the best found parameters on the whole dataset.
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    clf = grid_search.best_estimator_
        
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    
    diz_classifiers[layer] = {}
    chiave = 'clf_{}'.format(layer)
    diz_classifiers[chiave]['accuracy'] = accuracy_score(y_test, y_pred)
    diz_classifiers[chiave]['f1_score'] = f1_score(y_test, y_pred, average=None)
    diz_classifiers[chiave]['y_test'] = y_test
    diz_classifiers[chiave]['y_pred'] = y_pred
    diz_classifiers[chiave]['y_score'] = y_score
    diz_classifiers[chiave]['ids_proba'] = {Id:proba for Id, proba in zip(df['Id'], y_score)}


if __name__ == '__main__':
    
	for layer, i in zip(df_dist_5_bert_features_for_layers['df_dist_5'], enumerate(tqdm(list(range(len(df_dist_5_bert_features_for_layers['df_dist_5'])))))):
	    original_features = df_dist_5_bert_features_for_layers['df_dist_5'][layer]
	    train_and_predict(original_features, df_3_train, layer)
	    del original_features


	with open('diz_classifiers.pickle', 'wb') as handle:
	    pickle.dump(diz_classifiers, handle)    





