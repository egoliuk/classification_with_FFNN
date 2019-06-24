import numpy
import scipy.special

# одношарова штучна нейронна мережа
class NeuralNetwork:

    # ініціалізація нейронної мережі
    def __init__(self,
                 input_nodes_number,
                 hidden_nodes_number,
                 output_nodes_number,
                 training_factor):

        self.input_neurons_number = input_nodes_number
        self.hidden_neurons_number = hidden_nodes_number
        self.output_neurons_number = output_nodes_number

        self.weights_matrix_IH = numpy.random.normal(
            0.0, pow(self.hidden_neurons_number, -0.5),
            (self.hidden_neurons_number, self.input_neurons_number))

        self.weights_matrix_HO = numpy.random.normal(
            0.0, pow(self.output_neurons_number, -0.5),
            (self.output_neurons_number, self.hidden_neurons_number))

        self.learning_factor = training_factor

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренування нейронної мережі
    def train(self, input_signals, correct_output_signals):
        input_values_matrix = numpy.array(input_signals, ndmin=2).T
        correct_values_matrix = numpy.array(correct_output_signals, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_matrix_IH, input_values_matrix)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights_matrix_HO, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = correct_values_matrix - final_outputs
        hidden_errors = numpy.dot(self.weights_matrix_HO.T, output_errors)

        self.weights_matrix_HO += self.learning_factor * \
                                  numpy.dot(
                                      (output_errors * final_outputs * (1.0 - final_outputs)),
                                      numpy.transpose(hidden_outputs)
                                  )

        self.weights_matrix_IH += self.learning_factor * \
                                  numpy.dot(
                                      (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                      numpy.transpose(input_values_matrix)
                                  )
        pass

    # запит нейронної мережі
    def query(self, input_signals):
        input_values_matrix = numpy.array(input_signals, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_matrix_IH, input_values_matrix)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights_matrix_HO, hidden_outputs)
        final_output_signals = self.activation_function(final_inputs)

        return final_output_signals
