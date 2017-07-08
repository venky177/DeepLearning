from numpy import exp, array, random, dot


class NeuronLayer():
    def __init__(self, number_of_inputs_per_neuron,number_of_neurons):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, n_inputNeuron, n_outputNeuron, n_hiddenLayers, n_hiddenNeuron):
        self.layers=[]
        self.n_inputNeuron=n_inputNeuron
        self.n_outputNeuron = n_outputNeuron
        self.n_hiddenLayers = n_hiddenLayers
        if n_hiddenLayers == 0:
            self.layers.append(NeuronLayer(n_inputNeuron,n_outputNeuron))
        else:
            self.layers.append(NeuronLayer(n_inputNeuron,n_hiddenNeuron[0]))

        for i in range(0,n_hiddenLayers-1):
            self.layers.append(NeuronLayer(n_hiddenNeuron[i],n_hiddenNeuron[i+1]))
        if n_hidden != 0:
            self.layers.append(NeuronLayer(n_hiddenNeuron[n_hidden-1],op))

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer = self.think(training_set_inputs)

            layer_delta=[]
            layer_error=[]

            for i in range(0,self.n_hiddenLayers+1):
                layer_delta.append(0)
                layer_error.append(0)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer_error[self.n_hiddenLayers] = training_set_outputs - output_from_layer[self.n_hiddenLayers]
            layer_delta[self.n_hiddenLayers] = layer_error[self.n_hiddenLayers] * self.__sigmoid_derivative(output_from_layer[self.n_hiddenLayers])

            for i in range(self.n_hiddenLayers,0,-1):
                layer_error[i-1] = layer_delta[i].dot(self.layers[i].synaptic_weights.T)
                layer_delta[i-1] = layer_error[i-1] * self.__sigmoid_derivative(output_from_layer[i-1])


            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            #layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            #layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            #layer_adjustment = training_set_inputs.T.dot(layer1_delta)
            #layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layers[0].synaptic_weights += training_set_inputs.T.dot(layer_delta[0])
            for i in range(1,self.n_hiddenLayers+1):
                self.layers[i].synaptic_weights += output_from_layer[i-1].T.dot(layer_delta[i])

    # The neural network thinks.
    def think(self, inputs):
        output_from_layers=[]
        output_from_layers.append(self.__sigmoid(dot(inputs, self.layers[0].synaptic_weights)))
        if self.n_hiddenLayers == 0:
            return output_from_layers

        for i in range(1,self.n_hiddenLayers+1):
            output_from_layers.append(self.__sigmoid(dot(output_from_layers[i-1], self.layers[i].synaptic_weights)))
        return output_from_layers

    # The neural network prints its weights
    def print_weights(self):
        print "    Layer 1 (4 neurons, each with 3 inputs): "
        for i in range(0,self.n_hiddenLayers+1):
            print self.layers[i].synaptic_weights


if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)
    ip=3
    n_hidden=1
    op=1
    print "Number of Input Neurons: ",ip
    print "Number of hidden Layers: ",n_hidden
    print "Number of Output Neurons: ",op

    hid=[4]
    #for i in range(0,n_hidden):
    #    hid.append(input("Enter hidden neuron size"))
    print hid
    # Create layer 1 (4 neurons, each with 3 inputs)



    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(ip,op,n_hidden,hid)

    print "Stage 1) Random starting synaptic weights: "
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print "Stage 2) New synaptic weights after training: "
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print output