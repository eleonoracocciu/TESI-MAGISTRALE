import json
import os
import numpy as np
import pandas as pd
import re
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import collections
from IPython.display import display
import time
import spacy.cli
from spacy.language import Language
import math
import argparse

import torch
import torch.nn as nn
import random

from IPython.display import display

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score

import sys
import datetime

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertModel, BertConfig, BertTokenizer, AutoModel


class Classifier(nn.Module):
	
	# def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
	# 	super(Classifier, self).__init__()
	# 	# Load the BERT-based encoder
	# 	self.encoder = BertModel.from_pretrained(model_name)
	# 	# The configuration is needed to derive the size of the embedding, which 
	# 	# is produced by BERT (and similar models) to encode the input elements. 
	# 	# config = BertConfig.from_pretrained(model_name)
	# 	self.cls_size = int(config.hidden_size)
	# 	# Dropout is applied before the final classifier
	# 	self.input_dropout = nn.Dropout(p=dropout_rate)
	# 	# Final linear classifier
	# 	self.fully_connected_layer = nn.Linear(self.cls_size, num_labels)

	def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
	  super(Classifier, self).__init__()
	  #Procedimento: https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodel
	  #configurazione modello con pesi del pre-train
	  config_pretrain = BertConfig.from_pretrained(model_name)
	  # Load the BERT-based encoder --> modello con configurazione ottenuta, ma senza pesi del pre-train
	  # self.encoder = BertModel(config)
	  self.encoder = AutoModel.from_config(config_pretrain)
	  config = self.encoder.config
	  # The configuration is needed to derive the size of the embedding, which 
	  # is produced by BERT (and similar models) to encode the input elements. 
	  self.cls_size = int(config.hidden_size)
	  # Dropout is applied before the final classifier
	  self.input_dropout = nn.Dropout(p=dropout_rate)
	  # Final linear classifier
	  self.fully_connected_layer = nn.Linear(self.cls_size, num_labels)

	def forward(self, input_ids, attention_mask):
		# encode all outputs
		model_outputs = self.encoder(input_ids, attention_mask)
		# just select the vector associated to the [CLS] symbol used as
		# first token for ALL sentences
		encoded_cls = model_outputs.last_hidden_state[:,0]
		# apply dropout
		encoded_cls_dp = self.input_dropout(encoded_cls)
		# apply the linear classifier
		logits = self.fully_connected_layer(encoded_cls_dp)
		# return the logits
	
		return logits, encoded_cls


"""
The following method is used to convert input material into DataLoader that will be used to handle 
examples (during the training and the evaluation phase). Dataloaders are used in PyTorch to 
handle data, to split it into batches and to shuffle data.
"""
def generate_data_loader(examples, label_map, tokenizer, do_shuffle = False):
	
	'''
	Generate a Dataloader given the input examples

	examples: dataset columns = [Id, Sentence_1, Sentence_2, Class]
	label_map: a dictionary used to assign an ID to each label
	tokenize: the tokenizer used to convert input sentences into word pieces
	do_shuffle: a boolean parameter to shuffle input examples (usefull in training) 
	''' 
	#-----------------------------------------------
	# Generate input examples to the Transformer
	#-----------------------------------------------
	input_ids = []
	input_mask_array = []
	label_id_array = []
	sentences_id = []

	# Tokenization 
	for row in examples.itertuples():
		# tokenizer.encode_plus is a crucial method which:
		# 1. tokenizes examples
		# 2. trims sequences to a max_seq_length
		# 3. applies a pad to shorter sequences
		# 4. assigns the [CLS] special wor-piece such as the other ones (e.g., [SEP])
		encoded_sent = tokenizer.encode_plus(row.Sentence_1, row.Sentence_2, add_special_tokens=True, padding='max_length', truncation=True)
		# convert input word pieces to IDs of the corresponding input embeddings
		input_ids.append(encoded_sent['input_ids'])
		# store the attention mask to avoid computations over "padded" elements
		input_mask_array.append(encoded_sent['attention_mask'])
		
		# get id of event
		sentences_id.append(row.Id)

		# converts labels to IDs
		id = -1
		if row.Class in label_map:
			id = label_map[row.Class]
		label_id_array.append(id)

	# Convert to Tensor which are used in PyTorch
	input_ids = torch.tensor(input_ids) 
	input_mask_array = torch.tensor(input_mask_array)
	label_id_array = torch.tensor(label_id_array, dtype=torch.long)
	sentences_id = torch.tensor(sentences_id)

	# Building the TensorDataset
	dataset = TensorDataset(input_ids, input_mask_array, label_id_array, sentences_id)
	
	if do_shuffle:
		# this will shuffle examples each time a new batch is required
		sampler = RandomSampler
	else:
		sampler = SequentialSampler
	
	# Building the DataLoader
	return DataLoader(
			  dataset,  # The training samples.
			  sampler = sampler(dataset), # the adopted sampler
			  batch_size = batch_size) # Trains with this batch size.


def evaluate(dataloader, classifier, print_classification_output=False, print_result_summary=True):

	'''
	Evaluation method which will be applied to test datasets.
	It returns the pair (average loss, accuracy)

	dataloader: a dataloader containing examples to be classified
	classifier: the BERT-based classifier
	print_classification_output: to log the classification outcomes 
	'''

	total_loss = 0

	#original classes
	gold_classes = [] 

	#predicted classes
	system_classes = []
	
	#event ids
	total_sentences_ids = []
	#probabilities
	all_logits = []

	if print_classification_output:
		print("\n------------------------")
		print("  Classification outcomes")
		print("is_correct\tgold_label\tsystem_label\ttext")
		print("------------------------")
  
	# For each batch of examples from the input dataloader
	for batch in dataloader:   
		
		# Unpack this training batch from our dataloader. Notice this is populated 
		# in the method `generate_data_loader`
		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_labels = batch[2].to(device)
		b_sentence_ids = batch[3].to(device)
		
		#se sono in fase di test prendo, per ogni batch, gli id degli eventi che classifico
		total_sentences_ids.extend(b_sentence_ids.detach().cpu().numpy())


		# Tell pytorch not to bother with constructing the compute graph during
		# the forward pass, since this is only needed for backprop (training).
		with torch.no_grad():
			# Each batch is classifed        
			logits, _ = classifier(b_input_ids, b_input_mask)

			# Evaluate the loss. 
			total_loss += nll_loss(logits, b_labels)
		
		#prendo per ogni batch i logits della classificazione
		all_logits.append(logits)

		# Accumulate the predictions and the input labels
		_, preds = torch.max(logits, 1)
		system_classes += preds.detach().cpu()
		gold_classes += b_labels.detach().cpu()

		# Print the output of the classification for each input element
		if print_classification_output:
			for ex_id in range(len(b_input_mask)):
				input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
				# convert class id to the real label
				predicted_label = id_to_label_map[preds[ex_id].item()]
				gold_standard_label = "UNKNOWN"
				# convert the gold standard class ID into a real label
				if b_labels[ex_id].item() in id_to_label_map:
					gold_standard_label = id_to_label_map[b_labels[ex_id].item()]
				# put the prefix "[OK]" if the classification is correct
				print(predicted_label)
				output = '[OK]' if predicted_label == gold_standard_label else '[NO]'
				# print the output
				print(output+"\t", gold_standard_label, "\t", predicted_label, "\t"+input_strings)

	# Calculate the average loss over all of the batches.
	avg_loss = total_loss / len(dataloader)
	avg_loss = avg_loss.item()

	# Report the final accuracy for this test run.
	system_classes = torch.stack(system_classes).numpy()
	gold_classes = torch.stack(gold_classes).numpy()
	accuracy = np.sum(system_classes == gold_classes) / len(system_classes)
	print('Accuracy Sklearn %s' % accuracy_score(gold_classes, system_classes))
	#(TP + TN / TP + TN + FP + FN)

	#concateno i logits da ogni batch
	all_logits = torch.cat(all_logits, dim = 0)
	
	#applichiamo la softmax per ottenere le probabilita
	probabilities = F.softmax(all_logits, dim=1).cpu().numpy()
					
	diz_classification = {}			   
	diz_classification['accuracy'] = accuracy
	diz_classification['avg_loss'] = avg_loss
	diz_classification['ids_proba'] = {id_sents:proba for id_sents, proba in zip(total_sentences_ids, \
																				 probabilities)}

	diz_classification['ids_preds'] = {id_sents:pred for id_sents, pred in zip(total_sentences_ids, \
																				 system_classes)}
	diz_classification['gold_classes'] = gold_classes
	diz_classification['system_classes'] = system_classes
	diz_classification['label_list'] = label_list
	diz_classification['probabilities'] = probabilities
	diz_classification['total_sentences_ids'] = total_sentences_ids
									   
	
	if print_result_summary:
		print("\n------------------------")
		print("  Summary")
		print("------------------------")
		print(classification_report(gold_classes, system_classes))

		print("\n------------------------")
		print("  Confusion Matrix")
		print("------------------------")
		conf_mat = confusion_matrix(gold_classes, system_classes)
		
		print(conf_mat)
		
		# cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
		
		# cm.plot()
		
		# plt.show()

	return avg_loss, accuracy, diz_classification



def train(train_dataloader, test_dataloader):

	# Define the LOSS function. A BinaryCrossEntropyLoss is used for binary classification tasks.
	#nll_loss = torch.nn.BCEWithLogitsLoss()
	# All loss functions are available at:
	# - https://pytorch.org/docs/stable/nn.html#loss-functions

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	# NOTICE: the measure to be maximized should depends on the task. 
	# Here accuracy is used.
	best_test_accuracy = -1

	#for each epoch...
	for epoch_i in range(0, num_train_epochs):

		diz_classification_total["Epoch"+str(epoch_i)] = {}
		# ========================================
		#               Training
		# ========================================
		# Perform one full pass over the training set.
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_train_epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		train_loss = 0

		# Put the model into training mode.
		classifier.train() 

		#for each batch of training data...
		for step, batch in enumerate(tqdm(train_dataloader)):

			# Progress update every print_each_n_step batches.
			if step % print_each_n_step == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)

				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			# Unpack this training batch from our dataloader. 
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			# forward pass: compute predicted outputs by passing inputs to the model
			train_logits, _ = classifier(b_input_ids, b_input_mask)
			# calculate the loss        
			loss = nll_loss(train_logits, b_labels)
			# backward pass: compute gradient of the loss with respect to model parameters
			loss.backward() 
			# perform a single optimization step (parameter update)
			optimizer.step()
			# update running training loss
			train_loss += loss.item()

			# Update the learning rate with the scheduler, if specified
			if apply_scheduler:
				scheduler.step()

		# Calculate the average loss over all of the batches.
		avg_train_loss = train_loss / len(train_dataloader)

		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.3f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(training_time))


		# ========================================
		#     Evaluate on the Test set
		# ========================================
		# After the completion of each training epoch, measure our performance on our test set.
		print("")
		print("Running Test on epoch {}...".format(epoch_i))

		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently during evaluation.
		classifier.eval()

		# Apply the evaluate_method defined above to estimate 
		avg_test_loss, test_accuracy, diz_classification = evaluate(test_dataloader, classifier, False)

		# Measure how long the validation run took.
		test_time = format_time(time.time() - t0)

		print("  Accuracy: {0:.3f}".format(test_accuracy))
		print("  Test Loss: {0:.3f}".format(avg_test_loss))
		print("  Test took: {:}".format(test_time))

		# Record all statistics from this epoch.
		diz_classification_total["Epoch"+str(epoch_i)] = diz_classification
		diz_classification_total["Epoch"+str(epoch_i)]['training_loss'] = avg_train_loss
		diz_classification_total["Epoch"+str(epoch_i)]['training_time'] = training_time
		diz_classification_total["Epoch"+str(epoch_i)]['test_time'] = test_time

		# Save the model if the performance on the test set increases
		if test_accuracy > best_test_accuracy:
			best_test_accuracy = test_accuracy
			torch.save(classifier, output_model_name)
			print("\n  Saving the model during epoch " + str(epoch_i))
			print("  Actual Best Test Accuracy: {0:.3f}".format(best_test_accuracy))

		print()
		print("---------------------------------------------------------------------------------------------")
		print()


def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


def randomize_model(model):
	for module_ in model.named_modules():
		if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
			print("Precedente: {}".format(module_[1].weight))
			module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
			print("Successivo: {}".format(module_[1].weight))
		elif isinstance(module_[1], torch.nn.LayerNorm):
			print("Precedente: ({}, {})".format(module_[1].bias.data.zero_(), module_[1].weight.data.fill_(1.0)))
			module_[1].bias.data.zero_()
			module_[1].weight.data.fill_(1.0)
			print("Successivo: ({}, {})".format(module_[1].bias.data.zero_(), module_[1].weight.data.fill_(1.0)))
		if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
			print("Precedente: {}".format(module_[1].bias.data.zero_()))
			module_[1].bias.data.zero_()
			print("Successivo: {}".format(module_[1].bias.data.zero_()))
	return model


def parse_arg():
	parser = argparse.ArgumentParser(description='Script for BERT unsupervised + SVM')
	parser.add_argument('-tr', '--training_set', type=str,
						help='Training set')
	parser.add_argument('-ts', '--test_set', type=str,
						help='Test set')
	parser.add_argument('-bs', '--batch_size', type=str,
						help='Size of batch')
	parser.add_argument('-n', '--name', type=str,
						help='Name of dataset')
	parser.add_argument('-p_res', '--path_results', type=str,
						help='Path of pickle results')

	return parser.parse_args()


if __name__ == '__main__':

	args = parse_arg()

	# If there's a GPU available...
	if torch.cuda.is_available():    
		# Tell PyTorch to use the GPU.    
		device = torch.device("cuda")
		print('There are %d GPU(s) available.' % torch.cuda.device_count())
		print('We will use the GPU:', torch.cuda.get_device_name(0))
	# If not...
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")

	#estraggo training e test
	tr_set = args.training_set
	tst_set = args.test_set
	training_set = pd.read_csv(tr_set, sep="\t")
	test_set = pd.read_csv(tst_set, sep="\t")

	##Set random values
	seed_val = 213
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_val)

	#definisco il modello
	model_name = "dbmdz/bert-base-italian-cased"
	tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-cased", verbose=False)

	# --------------------------------
	# Parametri dell'encoder (BERT)
	# --------------------------------

	# dropout applied to the embedding produced by BERT before the classifiation
	out_dropout_rate = 0.1

	# --------------------------------
	# Training parameters
	# --------------------------------

	# the batch size = il batch è il numero di campioni che saranno passati alla rete in una volta
	batch_size = int(args.batch_size)

	# the learning rate used during the training process
	# learning_rate = 2e-5
	# learning_rate = 2e-6
	learning_rate = 2e-8
 
	# if you use large models (such as Bert-large) it is a good idea to use 
	# smaller values, such as 5e-6

	# name of the fine_tuned_model
	output_model_name = "{}_best_model_bert_finetuning_no_pretrain.pickle".format(args.name)

	# number of training epochs
	num_train_epochs = 5

	# ADVANCED: Schedulers allow to define dynamic learning rates.
	# You can find all available schedulers here
	# https://huggingface.co/transformers/main_classes/optimizer_schedules.html
	apply_scheduler = False
	# Here a `Constant schedule with warmup`can be activated. More details here
	# https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_constant_schedule_with_warmup
	warmup_proportion = 0.1

	# --------------------------------
	# Log parameters
	# --------------------------------

	# Print a log each n steps
	print_each_n_step = 500


	label_list = sorted(list(training_set['Class'].unique()))
	print("Label list", label_list)

	# Initialize a map to associate labels to the dimension of the embedding 
	# produced by the classifier
	label_to_id_map = {}
	id_to_label_map = {}
	for (i, label) in enumerate(label_list):
		label_to_id_map[label] = i
		id_to_label_map[i] = label


	print("Label to id map")
	print(label_to_id_map)

	# Build Train Dataloader
	train_dataloader = generate_data_loader(training_set, label_to_id_map, tokenizer, do_shuffle = True)
		
	# Build Test Dataloader
	test_dataloader = generate_data_loader(test_set, label_to_id_map, tokenizer, do_shuffle = False)

	classifier = Classifier(model_name, dropout_rate=out_dropout_rate)

	#classifier = randomize_model(classifier)
		
	# Put everything in the GPU if available
	if torch.cuda.is_available():    
		classifier.cuda()

	# Define the Optimizer. Here the ADAM optimizer (a sort of standard de-facto) is
	# used. AdamW is a variant which also adopts Weigth Decay.
	optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
	# More details about the Optimizers can be found here:
	# https://huggingface.co/transformers/main_classes/optimizer_schedules.html

	# Define the LOSS function. A CrossEntropyLoss is used for multi-class classification tasks. 
	nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

	diz_classification_total = {}

	train(train_dataloader, test_dataloader)

	diz_final = {}
	diz_final[args.name] = diz_classification_total

	if args.path_results:
		with open('{}{}_diz_bert_finetuning_no_pretrain.pickle'.format(args.path_results, args.name), 'wb') as handle:
			pickle.dump(diz_final, handle)

	else:
		with open('{}_diz_bert_finetuning_no_pretrain.pickle'.format(args.name), 'wb') as handle:
			pickle.dump(diz_final, handle)





