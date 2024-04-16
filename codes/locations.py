import os

data_folder = '../data/'
treebank_training_filename = 'treebank_training.csv'
treebank_testing_filename = 'treebank_testing.csv'
ptb_training_filename = 'ptb.train.txt'
ptb_testing_filename = 'ptb.test.txt'

treebank_training_path = os.path.join(data_folder,treebank_training_filename)
treebank_testing_path = os.path.join(data_folder,treebank_testing_filename)
ptb_training_path = os.path.join(data_folder,ptb_training_filename)
ptb_testing_path = os.path.join(data_folder,ptb_testing_filename)