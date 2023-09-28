import subprocess

if __name__ == '__main__':

	subprocess.check_call(['python3', './BERT_SVM_classification_no_pretrain.py', '--training_set', \
						   'df_dist_5_training_bert_features_for_layers_no_pretrain.pickle', \
						   '--test_set', 'df_dist_5_test_bert_features_for_layers_no_pretrain.pickle', \
						   '--name', 'df_dist_5'])


	subprocess.check_call(['python3', './BERT_SVM_classification_no_pretrain.py', '--training_set', \
						   'df_dist_10_training_bert_features_for_layers_no_pretrain.pickle', \
						   '--test_set', 'df_dist_10_test_bert_features_for_layers_no_pretrain.pickle', \
						   '--name', 'df_dist_10'])


	subprocess.check_call(['python3', './BERT_SVM_classification_no_pretrain.py', '--training_set', \
						   'df_dist_20_training_bert_features_for_layers_no_pretrain.pickle', \
						   '--test_set', 'df_dist_20_test_bert_features_for_layers_no_pretrain.pickle', \
						   '--name', 'df_dist_20'])


	subprocess.check_call(['python3', './BERT_SVM_classification_no_pretrain.py', '--training_set', \
						   'df_dist_30_training_bert_features_for_layers_no_pretrain.pickle', \
						   '--test_set', 'df_dist_30_test_bert_features_for_layers_no_pretrain.pickle', 	
						   '--name', 'df_dist_30'])

	subprocess.check_call(['python3', './BERT_SVM_classification_no_pretrain.py', '--training_set', \
						   'df_random_speech_training_bert_features_for_layers_no_pretrain.pickle', \
						   '--test_set', 'df_random_speech_test_bert_features_for_layers_no_pretrain.pickle', 	
						   '--name', 'df_random_speech'])