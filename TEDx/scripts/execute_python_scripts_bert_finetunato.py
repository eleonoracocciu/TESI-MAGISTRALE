import subprocess

if __name__ == '__main__':

	subprocess.check_call(['python3', './Bert_finetunato.py', \
							'--training_set', 'df_dist_5_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_5_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_5'])


	subprocess.check_call(['python3', './Bert_finetunato.py', \
							'--training_set', 'df_dist_10_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_10_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_10'])


	subprocess.check_call(['python3', './Bert_finetunato.py', \
							'--training_set', 'df_dist_20_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_20_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_20'])


	subprocess.check_call(['python3', './Bert_finetunato.py', \
							'--training_set', 'df_dist_30_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_30_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_30'])
