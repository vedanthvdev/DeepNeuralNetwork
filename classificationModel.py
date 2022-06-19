from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

#Defining some constants to help us later on
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_data = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_data = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_data, names = CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_data, names = CSV_COLUMN_NAMES, header = 0)

training_ans = train.pop("Species")
testing_ans = test.pop("Species")
lala = train.pop('SepalLength')

print(lala)

def input_function(featues, label, training: bool = True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(featues),label))
    if training:
        dataset = dataset.shuffle(2000).repeat()
        
    return dataset.batch(batch_size)


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns, hidden_units = [30, 10], n_classes = 3)

classifier.train(input_fn= lambda: input_function(train, training_ans, training = True), steps = 3000)

eval = classifier.evaluate(input_fn= lambda: input_function(test, testing_ans, training = False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval))

