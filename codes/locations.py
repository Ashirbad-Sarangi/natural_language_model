import os

input_path = '../data/'
filename_train = 'ptb.train.txt'
filename_test = 'ptb.test.txt'
filename_validation = 'ptb.validate.txt'
filename_vocabulary = 'vocabulary.csv'

train_location = os.path.join(input_path, filename_train)
test_location = os.path.join(input_path, filename_test)
validation_location = os.path.join(input_path, filename_validation)
vocabulary_location = os.path.join(input_path, filename_vocabulary)

