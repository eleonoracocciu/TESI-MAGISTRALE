import subprocess

if __name__ == '__main__':

	subprocess.check_call(['python3', './Bert_finetunato_no_pretrain.py', \
							'--training_set', 'df_dist_5_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_5_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_5', \
							'--path_results', './results/'])


	subprocess.check_call(['python3', './Bert_finetunato_no_pretrain.py', \
							'--training_set', 'df_dist_10_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_10_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_10', \
							'--path_results', './results/'])


	subprocess.check_call(['python3', './Bert_finetunato_no_pretrain.py', \
							'--training_set', 'df_dist_20_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_20_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_20', \
							'--path_results', './results/'])


	subprocess.check_call(['python3', './Bert_finetunato_no_pretrain.py', \
							'--training_set', 'df_dist_30_with_ids_ridotto_train.csv', \
							'--test_set', 'df_dist_30_with_ids_ridotto_test.csv', \
							'--batch_size', '8', \
							'--name', 'df_dist_30', \
							'--path_results', './results/'])
