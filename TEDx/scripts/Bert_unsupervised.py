import json
import os
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
from transformers import logging
import argparse


def save_dataframe(file_name, dataframe_to_save, path):
	
	try:
		dataframe_to_save.to_csv(path+file_name+".csv", index = False, sep = "\t")
		print("File salvato correttamente")
		
	except:
		print("Errore nel salvataggio")
	
	return 


def get_cosine_distance(dataset, tokenizer, model):
	
	diz_ids_cosine_distances = {}

	df = dataset.copy()

	#calcolo la lista degli id degli eventi del dataset di test per aggiungere al nuovo dataset le distanze
	#coseniche calcolate per ogni livello di BERT
	list_of_test_ids = list(dataset['Id'])
	
	for id_event, row in zip(list_of_test_ids, tqdm(dataset.itertuples(), total = dataset.shape[0])):
		 
		logging.set_verbosity_error()

		#creo nel dizionario la chiave per quest'evento
		diz_ids_cosine_distances[id_event] = {}
		
		inputs_sent_1 = tokenizer(row.Sentence_1, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True)
		inputs_sent_2 = tokenizer(row.Sentence_2, add_special_tokens=True, return_tensors="pt", padding='max_length', truncation=True)

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
		
		for cls_sent_1, cls_sent_2, layer_num in zip(all_layers_sent_1_cls, all_layers_sent_2_cls, list(range(1,13))):
			cosine_similarity = torch.nn.CosineSimilarity()
			cosine_distance = 1 - cosine_similarity(cls_sent_1, cls_sent_2).item()
													
			cosine_distance = float("%.4f"%(cosine_distance))
			#df.loc[row.Index, "Cosine_distance_layer_" + str(layer_num)] = cosine_distance
			
			#localizzo nel dataset la riga dove l'id dell'evento e' quello dell'iterazione corrente 
			#e agggiungo e distanze coseniche
			df.loc[df['Id'] == id_event, "Cosine_distance_layer_" + str(layer_num)] = cosine_distance

			#aggiungo nel dizionario le diverse distanze coseniche
			diz_ids_cosine_distances[id_event]["Cosine_distance_layer_" + str(layer_num)] = cosine_distance


	return df, diz_ids_cosine_distances


def parse_arg():
	parser = argparse.ArgumentParser(description='Script for BERT unsupervised')
	parser.add_argument('-d', '--test_set', type=str,
						help='Dataset')
	parser.add_argument('-n', '--name', type=str,
						help='Name of dataset')

	return parser.parse_args()


if __name__ == '__main__':

	path_train_test = "../dataset/train_test/"
	path_cosine_distance = "../dataset/cosine_distances/"
	path_pickle = "../pickle/"

	#leggo il dataset di test su cui eseguire il calcolo delle distanze coseniche
	args = parse_arg()
	test_set = args.test_set
	df_with_ids = pd.read_csv(path_train_test+test_set, sep="\t")

	#carico tokenizer e modello
	tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-cased", verbose=False)
	model = BertModel.from_pretrained("dbmdz/bert-base-italian-cased", output_hidden_states=True)

	#calcolo le distanze coseniche
	df_cosine_distance, diz_ids_cosine_distances = get_cosine_distance(df_with_ids, tokenizer, model)

	#salvo il dataset con le distanze coseniche
	save_dataframe("{}_cosine_distance".format(args.name), df_cosine_distance, path_cosine_distance)

	#salvo il dizionario con id_evento - distanza cosenica per ogni livello
	with open('{}{}_diz_ids_cosine_distances.pickle'.format(path_pickle, args.name), 'wb') as handle:
		pickle.dump(diz_ids_cosine_distances, handle)
