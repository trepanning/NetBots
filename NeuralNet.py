'''
A basic fully-connected feedforward neural network
'''

import random
import math

# lower bound for random weights
NEURON_WEIGHT_INIT_LOW = -1
# upper bound for random weights
NEURON_WEIGHT_INIT_HIGH = 1
# default hidden layer activation function
HIDDEN_LAYER_FUNC = 'leakyrelu6'
# default output layer activation function
OUTPUT_LAYER_FUNC = 'tanh'

# these activation functions can be used in hidden and output layers
ACTIVATION_FUNCTIONS = { \
'logistic' : lambda x : 1 / (1 + math.exp(-1 * x)), \
'tanh' : lambda x : (2 / (1 + math.exp(-2 * x))) - 1, \
'relu' : lambda x : x if x > 0 else 0, \
'leakyrelu' : lambda x : x if x > 0 else 0.001 * x, \
'relu6' : lambda x : 6 if x >= 6 else x if x > 0 else 0, \
'leakyrelu6' : lambda x : 6 if x >= 6 else x if x > 0 else 0.001 * x \
}


class NeuralNetwork():
    def __init__(self, \
    n_inputs, \
    n_outputs, \
    n_layers, \
    h_af = HIDDEN_LAYER_FUNC, \
    o_af = OUTPUT_LAYER_FUNC):
        self.hidden_layers = []
        self.hidden_af = h_af
        self.output_layer = []
        self.output_af = o_af

        # keep track of required number of inputs at each layer
        inputs = n_inputs
        #build hidden layers
        for l in n_layers:
            layer = []
            for _ in range(l):
                # initialize neuron weights adding extra one for the bias
                layer.append( \
                    [random.uniform(-1, 1) for i in range(inputs + 1)])
            self.hidden_layers.append(layer)
            inputs = l
        #build output layer
        for _ in range(n_outputs):
            self.output_layer.append( \
                [random.uniform(-1, 1) for i in range(inputs + 1)])

    def calculate_neuron(self, neuron, data, af):
            # list of weights * data products
            products = []
            # index value to match each input to the corresponding weight
            i = 0
            while i < len(neuron) - 1:
                products.append(neuron[i] * data[i])
                i += 1
            # add bias
            products.append(neuron[i] * -1)
            return ACTIVATION_FUNCTIONS[af](sum(products))

    def feed_forward(self, data):
        # data being processed at the current layer
        layer_input = data
        # data to be used at the next layer
        next_layer_input = []

        # calculate hidden layer
        for layer in self.hidden_layers:
            for neuron in layer:
                next_layer_input.append( \
                    self.calculate_neuron( \
                        neuron, layer_input, self.hidden_af))
            # next layer processes this layer's output
            layer_input = next_layer_input
            # clear list to collect results from next layer
            next_layer_input = []

        # calculate output layer
        for neuron in self.output_layer:
            next_layer_input.append( \
                self.calculate_neuron( \
                    neuron, layer_input, self.output_af))

        # return results of output layer
        return next_layer_input

    # returns a serialized list [] of weights from all layers
    def encoded(self):
        e = [weight for neuron in self.hidden_layers for weight in neuron]
        e += [weight for neuron in self.output_layer for weight in neuron]
        return e

    # sets this network's weights with a serialized list
    def decode(self, l):
        weights = l
        # set hidden layers
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron = weights[:len(neuron)]
                weights = weights[len(neuron):]

        # set output layer
        for neuron in self.output_layer:
            neuron = weights[:len(neuron)]
            weights = weights[len(neuron):]

