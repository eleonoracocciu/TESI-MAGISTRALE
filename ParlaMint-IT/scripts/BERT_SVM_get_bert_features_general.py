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
import seaborn as sns
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


def get_bert_vectors(dataset, dataset_name, tokenizer, model):
    
    #dizionario che conterra' i dati di tutti i livelli
    diz_all_layers = {}

    #creo un dizionario che abbia al suo interno tutti i livelli (12) --> avro' 12 dataset alla fine
    dict_features_layers = {}
    for layer_num in list(range(1,13)):
        dict_features_layers['Layer_{}'.format(layer_num)] = []
    
    for row in tqdm(dataset.itertuples(), total = dataset.shape[0]):
         
        logging.set_verbosity_error()
        
        inputs_sent_1 = tokenizer(row.Sentence_1, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True)
        inputs_sent_2 = tokenizer(row.Sentence_2, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True)

        inputs_sent_1.to(device)
        inputs_sent_2.to(device)

        outputs_sent_1 = model(**inputs_sent_1)
        outputs_sent_2 = model(**inputs_sent_2)
 
        #(len(outputs_1['hidden_states']) == 13 because the first element is the input embeddings, 
        #the rest is the outputs of each of BERT’s 12 layers.)
        all_hidden_states_sent_1 = outputs_sent_1.hidden_states[1:]
        all_hidden_states_sent_2 = outputs_sent_2.hidden_states[1:]
        
        #ogni frase la rappresentiamo con la rappresentazione a ognuno degli strati con il suo cls
        
        # just select the vector associated to the [CLS] symbol used as
        # first token for ALL sentences
        #encoded_cls = model_outputs.last_hidden_state[:,0]
        all_layers_sent_1_cls = [layer[:,0] for layer in all_hidden_states_sent_1]
        all_layers_sent_2_cls = [layer[:,0] for layer in all_hidden_states_sent_2]

        #per ogni evento (coppia di frasi), salvo nella rispettiva chiave del dizionario (numero del livello) 
        #l'id dell'evento, i vettori restituiti da bert (concatenati) e la classe dell'evento (1 o 0)
        for cls_sent_1, cls_sent_2, layer_num in zip(all_layers_sent_1_cls, all_layers_sent_2_cls, list(range(1,13))):
                        
            #concateno il vettore cls della frase 1 e il vettore cls della frase 2 e prendo anche \
            #l'id della riga (evento) e la classe dell'evento
            dict_features_layers['Layer_{}'.format(layer_num)].append(
                (   
                    row.Id, \
                    torch.squeeze(torch.cat((cls_sent_1.detach(), cls_sent_2.detach()), 1)).tolist(), \
                    row.Class
                )
            )
            
    diz_all_layers[dataset_name] = dict_features_layers
    
    return diz_all_layers


def parse_arg():
    parser = argparse.ArgumentParser(description='Script for BERT unsupervised + SVM')
    parser.add_argument('-tr', '--training_set', type=str,
                        help='Training set')
    parser.add_argument('-ts', '--test_set', type=str,
                        help='Test set')
    parser.add_argument('-n', '--name', type=str,
                        help='Name of dataset')
    parser.add_argument('-ptrts', '--path_train_test', type=str,
                        help='Path of train and test')
    parser.add_argument('-p_res', '--path_results', type=str,
                        help='Path of pickle results')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_arg()
    
    #estraggo training e test
    tr_set = args.training_set
    tst_set = args.test_set
    training_set = ''
    test_set = ''

    if args.path_train_test:
        training_set = pd.read_csv(args.path_train_test+tr_set, sep="\t")
        test_set = pd.read_csv(args.path_train_test+tst_set, sep="\t")

    else:
        training_set = pd.read_csv(tr_set, sep="\t")
        test_set = pd.read_csv(tst_set, sep="\t")

    #definisco tokenizer e modello
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-cased", verbose=False)
    model = BertModel.from_pretrained("dbmdz/bert-base-italian-cased", output_hidden_states=True)

    print("Tokenizer do_lower_case: ", tokenizer.do_lower_case)

    #uso la gpu se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    df_name_training = "{}_training".format(args.name)
    df_name_test = "{}_test".format(args.name)

    #estraggo le features (vettori di BERT) separatamente per training e test
    training_features_for_layers = get_bert_vectors(training_set, df_name_training, tokenizer, model)
    test_features_for_layers = get_bert_vectors(test_set, df_name_test, tokenizer, model)

    if args.path_results:
        with open('{}{}_bert_features_for_layers.pickle'.format(args.path_results, df_name_training), 'wb') as handle:
            pickle.dump(training_features_for_layers, handle)

        with open('{}{}_bert_features_for_layers.pickle'.format(args.path_results, df_name_test), 'wb') as handle:
            pickle.dump(test_features_for_layers, handle)

    else:
        with open('{}_bert_features_for_layers.pickle'.format(df_name_training), 'wb') as handle:
            pickle.dump(training_features_for_layers, handle)

        with open('{}_bert_features_for_layers.pickle'.format(df_name_test), 'wb') as handle:
            pickle.dump(test_features_for_layers, handle)

