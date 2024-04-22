import os

data_folder = '../data/'
model_folder = '../models/'

treebank_training_filename = 'treebank_training.csv'
treebank_testing_filename = 'treebank_testing.csv'
ptb_training_filename = 'ptb.train.txt'
ptb_testing_filename = 'ptb.test.txt'
conll_training_filename = 'conll_training.txt'

conll_8k_modelname_v1 = 'conll_8k_model_v1.keras'
conll_8k_modelname_v2 = 'conll_8k_model_v2.keras'
treebank_modelname = 'new_trained_model_treebank.keras'
penn_treebank_modelname = 'trained_model_10k_v1.keras'

treebank_training_path = os.path.join(data_folder,treebank_training_filename)
treebank_testing_path = os.path.join(data_folder,treebank_testing_filename)
ptb_training_path = os.path.join(data_folder,ptb_training_filename)
ptb_testing_path = os.path.join(data_folder,ptb_testing_filename)
conll_8k_training_path = os.path.join(data_folder,conll_training_filename)


conll_8k_model_v1_path = os.path.join(model_folder,conll_8k_modelname_v1)
conll_8k_model_v2_path = os.path.join(model_folder,conll_8k_modelname_v2)
treebank_model_path = os.path.join(model_folder,treebank_modelname)
penn_treebank_model_path = os.path.join(model_folder,treebank_modelname)
