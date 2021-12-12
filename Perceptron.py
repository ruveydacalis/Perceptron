import numpy as np
import pandas as pd
class Perceptron():

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):   # Perceptron learning rate is between 0 and 1
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
    # np.zeros(n), will create a vector with an n-number of 0’s

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if sum > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[0] += self.learning_rate * (label - prediction)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs

# Read and  modify data
data = pd.read_csv('perceptron.csv')
data = data[:]
data[4] = np.where(data.iloc[:, -1] == 'Iris-setosa', 0, 1)
training_inputs = data.iloc[:, 0:4].values
labels = data.iloc[:, -1].values

# Activate the perceptron
perceptron = Perceptron(4)

# Train the perceptron
perceptron.train(training_inputs, labels)

# Test the perceptron
test_data = pd.read_csv('perceptron.test.csv')
test_data = test_data[:]
test_data[4] = np.where(test_data.iloc[:, -1] == 'Iris-setosa', 0, 1)
test_data_input = test_data.iloc[:, 0:4].values

# Count number of total miss classified
miss_classified = 0
miss_classified_list = []
test_data.iat[0, 5]
for i in range(0, test_data_input.shape[0]):
    test_prediction1 = perceptron.predict(test_data_input[i])
    miss_classified_list.append(test_prediction1)
    if test_prediction1 != test_data.iat[i, 5]:
        miss_classified = miss_classified + 1

# Accuracy is here giving 100%
accuracy = (1 - (miss_classified / test_data.shape[0])) * 100
print("ACCURACY: ", accuracy)

# Take a input from user
list_input = [0, 0, 0, 0]
for i in range(0, 4):
    input_user = float(input('GIVE A INPUT OF THE VALUE :   '))
    list_input[i] = input_user
print('THE DATA YOU HAVE PROVIDED IS : ', list_input)
output_prediction = perceptron.predict(list_input)
if output_prediction == 0:
    print('YOU HAVE GIVEN, : Iris-setosa')
elif output_prediction == 1:
    print('YOU HAVE GIVEN : Iris-versicolor')