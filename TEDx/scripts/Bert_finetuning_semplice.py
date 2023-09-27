import argparse

import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")

def tokenize_function(examples):
    return tokenizer(examples["Sentence_1"], examples["Sentence_2"], add_special_tokens=True, padding="max_length", truncation=True)


def fine_tuning(data_dir, training, test, checkpoint, task_type, epochs):

    # Load dataset
    data_files = {"train": "{}".format(training), "test": "{}".format(test)}
    dataset = load_dataset("../dataset/train_test", data_files=data_files)

    model_name = "dbmdz/bert-base-italian-cased"

    # Tokenize data
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load model
    all_labels = []
    for sample in dataset["train"]:
        all_labels.append(sample["label"])
    all_labels = list(set(all_labels))
    print("all_labels: {}".format(all_labels))
    num_labels = len(all_labels)
    model = AutoModel.from_pretrained.from_pretrained(model_name, num_labels=num_labels)
    

    # Load and modify training arguments
    training_args = TrainingArguments(
                        output_dir=output_dir
                    )
    training_args.num_train_epochs = float(epochs)
    training_args.save_strategy = 'no'

    training_args.learning_rate = 0.00002

    # Load trainer, train and save the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
    )
    trainer.train()
    trainer.save_model()
    
    # Predict and save predictions
    predictions = trainer.predict(tokenized_datasets["test"])
    num_checkpoint = checkpoint.split('/')[-2].split('-')[1]
    output = open(output_dir + '/predictions_' + str(epochs) + '_' + str(num_checkpoint) + '.tsv', 'w')
    output.write("y_true\ty_preds\n")
    
    preds = predictions.predictions
    preds = np.argmax(preds, axis=-1)
    for y_true, pred in zip(predictions.label_ids, preds):
        output.write(str(y_true) + '\t' + str(pred) + '\n')

    output.close()


def parse_arg():
    parser = argparse.ArgumentParser(description='Script for fine-tuning BERT')
    parser.add_argument('-tr', '--training set', type=str,
                        help='Dataset for training')
    parser.add_argument('-ts', '--test set', type=str,
                        help='Dataset for training')
    parser.add_argument('-c', '--checkpoint', type=str,
                        help='Checkpoint of the model')
    parser.add_argument('-e', '--epochs', type=str,
                        help='Number of epochs')

    return parser.parse_args()

        
if __name__ == '__main__':
    args = parse_arg()
    data_dir = args.data_dir
    checkpoint = args.checkpoint
    task_type = args.task_type
    epochs = args.epochs
    fine_tuning(data_dir, training, test, checkpoint, task_type, epochs)
