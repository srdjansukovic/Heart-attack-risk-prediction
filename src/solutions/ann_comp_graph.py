from __future__ import print_function

from abc import abstractmethod
import math
import random
import copy
import pandas as pd
import numpy as np
import pickle

from matplotlib import pyplot
from sklearn.model_selection import train_test_split


class ComputationalNode(object):

    @abstractmethod
    def forward(self, x):  # x is an array of scalars
        pass

    @abstractmethod
    def backward(self, dz):  # dz is a scalar
        pass


class MultiplyNode(ComputationalNode):

    def __init__(self):
        self.x = [0., 0.]  # x[0] is input, x[1] is weight

    def forward(self, x):
        self.x = x
        return self.x[0] * self.x[1]

    def backward(self, dz):
        return [dz * self.x[1], dz * self.x[0]]


class SumNode(ComputationalNode):

    def __init__(self):
        self.x = []  # x is in an array of inputs

    def forward(self, x):
        self.x = x
        return sum(self.x)

    def backward(self, dz):
        return [dz for xx in self.x]


class SigmoidNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._sigmoid(self.x)

    def backward(self, dz):
        return dz * self._sigmoid(self.x) * (1. - self._sigmoid(self.x))

    def _sigmoid(self, x):
        return 1. / (1. + math.exp(-x))


class TanHNode(ComputationalNode):
    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._tanh(self.x)

    def backward(self, dz):
        return dz * (1. - self._tanh(self.x)**2)

    def _tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


class ReluNode(ComputationalNode):

    def __init__(self):
        self.x = 0.  # x is an input

    def forward(self, x):
        self.x = x
        return self._relu(self.x)

    def backward(self, dz):
        return dz * (1. if self.x > 0. else 0.)

    def _relu(self, x):
        return max(0., x)


class NeuronNode(ComputationalNode):

    def __init__(self, n_inputs, activation):
        self.n_inputs = n_inputs
        self.multiply_nodes = []  # for inputs and weights
        self.sum_node = SumNode()  # for sum of inputs*weights
        self.batch_size_cnt = 0
        self.batch_gradients = []

        for n in range(n_inputs):  # collect inputs and corresponding weights
            mn = MultiplyNode()
            mn.x = [1., random.gauss(0., 0.1)]  # init input weights
            self.multiply_nodes.append(mn)

        mn = MultiplyNode()  # init bias node
        mn.x = [1., random.gauss(0., 0.01)]  # init bias weight
        self.multiply_nodes.append(mn)

        if activation == 'sigmoid':
            self.activation_node = SigmoidNode()
        elif activation == 'relu':
            self.activation_node = ReluNode()
        elif activation == 'tanh':
            self.activation_node = TanHNode()
        else:
            raise RuntimeError('Unknown activation function "{0}".'.format(activation))

        self.previous_deltas = [0.] * (self.n_inputs + 1)
        self.gradients = []

    def forward(self, x):  # x is a vector of inputs
        x = copy.copy(x)
        x.append(1.)  # for bias

        for_sum = []
        for i, xx in enumerate(x):
            inp = [x[i], self.multiply_nodes[i].x[1]]
            for_sum.append(self.multiply_nodes[i].forward(inp))

        summed = self.sum_node.forward(for_sum)
        summed_act = self.activation_node.forward(summed)
        return summed_act

    def backward(self, dz):
        dw = []
        dx = []
        b = dz[0] if type(dz[0]) == float else sum(dz)
        
        b = self.activation_node.backward(b)
        b = self.sum_node.backward(b)
        for i, bb in enumerate(b):
            dw.append(self.multiply_nodes[i].backward(bb)[1])
            dx.append(self.multiply_nodes[i].backward(bb)[0])

        self.gradients = dw
        return dx

    def update_weights(self, learning_rate, momentum):
        self.batch_gradients.append(self.gradients)

        if self.batch_size_cnt % 32 == 0:
            #print('batch grad len: ', len(self.batch_gradients))
            #print('batch grad elem len: ', len(self.batch_gradients[0]))

            #self.gradients = np.array(np.mean(np.array(self.batch_gradients), axis=0)).tolist()
            self.gradients = [float(sum(col))/len(col) for col in zip(*self.batch_gradients)]
            #print('mean grad: ', self.gradients)

            for i, multiply_node in enumerate(self.multiply_nodes):
                mean_gradient = self.gradients[i]
                delta = learning_rate*mean_gradient + momentum*self.previous_deltas[i]
                self.previous_deltas[i] = delta
                self.multiply_nodes[i].x[1] -= delta

            self.batch_gradients = []

        self.gradients = []
        self.batch_size_cnt += 1

    '''def update_weights(self, learning_rate, momentum):
        for i, multiply_node in enumerate(self.multiply_nodes):
            mean_gradient = self.gradients[i]
            #mean_gradient = sum([grad[i] for grad in self.gradients]) / len(self.gradients)
            delta = learning_rate*mean_gradient + momentum*self.previous_deltas[i]
            self.previous_deltas[i] = delta
            self.multiply_nodes[i].x[1] -= delta

        self.gradients = []'''



class NeuralLayer(ComputationalNode):

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        self.neurons = []
        # construct layer
        for _ in range(n_neurons):
            neuron = NeuronNode(n_inputs, activation)
            self.neurons.append(neuron)

    def forward(self, x):  # x is a vector of "n_inputs" elements
        layer_output = []
        for neuron in self.neurons:
            neuron_output = neuron.forward(x)
            layer_output.append(neuron_output)

        return layer_output

    def backward(self, dz):  # dz is a vector of "n_neurons" elements
        b = []
        for idx, neuron in enumerate(self.neurons):
            neuron_dz = [d[idx] for d in dz]
            neuron_dz = neuron.backward(neuron_dz)
            b.append(neuron_dz[:-1])

        return b  # b is a vector of "n_neurons" elements

    def update_weights(self, learning_rate, momentum):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate, momentum)


class NeuralNetwork(ComputationalNode):

    def __init__(self):
        # construct neural network
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):  # x is a vector which is an input for neural net
        prev_layer_output = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:  # input layer
                prev_layer_output = layer.forward(x)
            else:
                prev_layer_output = layer.forward(prev_layer_output)

        return prev_layer_output  # actually an output from last layer

    def backward(self, dz):
        next_layer_dz = None
        for idx, layer in enumerate(self.layers[::-1]):
            if idx == 0:
                next_layer_dz = layer.backward(dz)
            else:
                next_layer_dz = layer.backward(next_layer_dz)

        return next_layer_dz

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            layer.update_weights(learning_rate, momentum)

    def fit(self, X, Y, learning_rate, momentum, nb_epochs, shuffle=False, verbose=0):
        assert len(X) == len(Y)

        hist = []
        for epoch in range(nb_epochs):
            if shuffle:
                random.seed(epoch)
                random.shuffle(X)
                random.seed(epoch)
                random.shuffle(Y)

            total_loss = 0.0
            #i = 0
            for x, y in zip(X, Y):
                #i += 1
                # forward pass to compute output
                pred = self.forward(x)
                # compute loss
                grad = 0.0
                for o, t in zip(pred, y):
                    total_loss += (t - o) ** 2.
                    grad += -(t - o)
                # backward pass to compute gradients
                self.backward([[grad]])
                # update weights with computed gradients
                #if i % 25 == 0:
                self.update_weights(learning_rate, momentum)
            #print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
            hist.append(total_loss)
            if verbose == 1:
                print('Epoch {0}: loss {1}'.format(epoch + 1, total_loss))
        print('Loss: {0}'.format(total_loss))
        return hist

    def predict(self, x):
        return self.forward(x)


def create_bar_graph(values, stroke, xlabel='', ylabel='', title=''):
    pyplot.style.use('ggplot')

    x = [] # vrednosti barova
    for item in values:
        if item not in x:
            x.append(item)

    x_dict = {x[i]: 0. for i in range(len(x))}  # vrednost za koju se crta grafik i broj pojavljivanja
    x_dict_all = {x[i]: 0. for i in range(len(x))}  # ukupan broj po kategoriji nezavisno od polja stroke

    for item, str in zip(values, stroke):  # povecavanje broja pojavljivanja za svaki item kad je stroke 1
        if str == 1:
            x_dict[item] += 1.

    for item in values:
        x_dict_all[item] += 1.

    #scale data
    for key in x_dict:
        x_dict[key] /= x_dict_all[key]

    y = [] # broj pojava
    for item in x:
        y.append(x_dict[item])

    x_pos = [i for i, _ in enumerate(x)]

    pyplot.bar(x_pos, y, color='green')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)

    x_ticks = []
    for idx, i in enumerate(x):
        if i == 1:
            x_ticks.append('Yes')
        elif i == 0:
            x_ticks.append('No')
        else:
            x_ticks.append(i)

    pyplot.xticks(x_pos, x_ticks)
    pyplot.show()


def normalize_data(data):
    min_data = []
    max_data = []

    for i in range(len(data[0])):
        temp_min = data[i][0]
        temp_max = data[i][0]
        for point in data:
            if point[i] < temp_min:
                temp_min = point[i]
            if point[i] > temp_max:
                temp_max = point[i]
        min_data.append(temp_min)
        max_data.append(temp_max)

    for point in data:
        for i in range(len(point)):
            new_value = (point[i] - min_data[i]) / (max_data[i] - min_data[i])
            point[i] = new_value


def normalize_column(values):
    min_val = min(values)
    max_val = max(values)
    for i in range(len(values)):
        new_value = (values[i] - min_val)/(max_val - min_val)
        values[i] = new_value


def find_average_bmi(values):
    sum = 0.
    nans = 0.
    for value in values:
        if math.isnan(value):
            nans += 1.
        else:
            sum += value
    return sum/(len(values) - nans)


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(NeuralLayer(16, 2, 'tanh'))
    nn.add(NeuralLayer(2, 1, 'sigmoid'))

    data = pd.read_csv('../../data/dataset.csv')

    # normalize data
    normalize_column(data.age.values)
    normalize_column(data.avg_glucose_level.values)
    data.bmi = data.bmi.fillna(find_average_bmi(data.bmi.values))
    normalize_column(data.bmi.values)

    create_bar_graph(data.gender.values, data.stroke.values, 'Gender', 'Number of strokes')
    create_bar_graph(data.hypertension.values, data.stroke.values, 'Hypertension', 'Number of strokes')
    create_bar_graph(data.heart_disease.values, data.stroke.values, 'Heart disease', 'Number of strokes')
    create_bar_graph(data.ever_married.values, data.stroke.values, 'Ever married', 'Number of strokes')
    create_bar_graph(data.work_type.values, data.stroke.values, 'Work type', 'Number of strokes')
    create_bar_graph(data.Residence_type.values, data.stroke.values, 'Residence type', 'Number of strokes')
    create_bar_graph(data.smoking_status.values, data.stroke.values, 'Smoking status', 'Number of strokes')

    # drop irrelevant data
    data.drop('gender', inplace=True, axis=1)
    data.drop('Residence_type', inplace=True, axis=1)
    data.drop('id', inplace=True, axis=1)

    data = pd.get_dummies(data)

    data.to_csv('../../data/parsed_data.csv', index=False)
    #data = pd.read_csv('../../data/parsed_data.csv')
    data_x = copy.deepcopy(data)
    data_x.drop('stroke', inplace=True, axis=1)
    data_x = data_x.values
    data_y = data.stroke.values

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    x_train = np.array(x_train).tolist()
    y_train = [[float(item)] for item in y_train]
    y_test = [[float(item)] for item in y_test]
    x_test = np.array(x_test).tolist()


    # replicate data for training (used for oversampling)
    x_train_replicate = []
    y_train_replicate = []
    for i in range(len(y_train)):
        if y_train[i][0] == 1.0:
            for _ in range(19):
                x_train_replicate.append(x_train[i])
                y_train_replicate.append(y_train[i])
        else:
            x_train_replicate.append(x_train[i])
            y_train_replicate.append(y_train[i])

    zip_for_shuffle = list(zip(x_train_replicate, y_train_replicate))
    random.shuffle(zip_for_shuffle)
    x_train_replicate, y_train_replicate = zip(*zip_for_shuffle)
    x_train_replicate = list(x_train_replicate)
    y_train_replicate = list(y_train_replicate)

    # cut out some data for equal number of samples (option for undersampling)
    x_train_cut = []
    y_train_cut = []
    for i in range(len(y_train)):
        if y_train[i][0] == 1.0:
            x_train_cut.append(x_train[i])
            y_train_cut.append(y_train[i])
    cut_len = len(x_train_cut)
    for i in range(len(y_train)):
        if y_train[i][0] == 0.0:
            if len(y_train_cut) <= cut_len * 2:
                x_train_cut.append(x_train[i])
                y_train_cut.append(y_train[i])


    history = nn.fit(x_train_replicate, y_train_replicate, learning_rate=0.01, momentum=0.9, nb_epochs=60, shuffle=True, verbose=1)
    file_nn = open('../../data/final.obj', 'wb')
    pickle.dump(nn, file_nn)
    file_nn.close()
    pyplot.plot(history)
    pyplot.show()


    '''file_nn = open('../../data/final.obj', 'rb')
    nn = pickle.load(file_nn)
    file_nn.close()'''

    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    predicts = []
    x_test_plot = []
    print(len(y_test))
    for i in range(len(y_test)):
        pred = nn.predict(x_test[i])[0]
        predicts.append(pred)
        x_test_plot.append(1)
        if pred <= 0.5:
            if y_test[i][0] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if y_test[i][0] == 1:
                TP += 1
            else:
                FP += 1
    pyplot.scatter(x_test_plot, predicts)
    pyplot.show()
    print('TP: ', TP)
    print('FP: ', FP)
    print('TN: ', TN)
    print('FN: ', FN)

    overall_correct_percentage = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = (precision * recall)/(precision + recall)

    print('overall correct score: ', overall_correct_percentage)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1 score: ', f1_score)

