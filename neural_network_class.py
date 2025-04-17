import math, random, statistics, numpy, copy

class NeuralNetworkModel:
    def __init__(self, training_data_file_name, text_reference = None, model_size = 5, neuron_size_base = 2, training_epochs = 75, training_data_proportion = 0.75, delta = 1.0, learning_rate = 0.001, learning_rate_decay_rate = 10.0 ** -4, momentum_factor = 0.9, max_norm_benchmark = 90):
        self.training_data_file_name = training_data_file_name
        self.text_reference = text_reference
        self.delta = delta
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.max_norm_benchmark = max_norm_benchmark
        self.momentum_factor = momentum_factor
        self.model_size = model_size
        self.neuron_size_base = neuron_size_base
        self.training_epochs = training_epochs
        self.training_data_proportion = training_data_proportion
        self.pre_scaled_data = self.assemble_pre_scaled_data(self.training_data_file_name, self.text_reference)
        self.pre_scaled_training_data, self.pre_scaled_validation_data = self.split_pre_scaled_data(self.pre_scaled_data, self.training_data_proportion)
        self.z_score_scales = self.compute_z_score_scales(self.pre_scaled_training_data)
        self.training_data = self.z_score_scale_data(self.pre_scaled_training_data, self.z_score_scales)
        self.validation_data = self.z_score_scale_data(self.pre_scaled_validation_data, self.z_score_scales)
        self.parameters = self.initialize_parameters(self.training_data, self.model_size, self.neuron_size_base)
        self.parameter_velocities = self.initialize_parameter_velocities(self.parameters)
        self.average_validation_cost_value_over_epochs, self.minimum_cost_index = self.train_model(self.training_epochs, self.training_data, self.parameters, self.parameter_velocities, self.validation_data, self.model_size, self.neuron_size_base, self.delta, self.learning_rate, self.learning_rate_decay_rate, self.momentum_factor, self.max_norm_benchmark)

    """Private Methods - To Be Used Internally By the Model Only"""

    """This method returns the processed data line of an unprocessed data point from
    the training dataset or validation dataset during the Training Stage, or of a new data
    point passed into the model during the Inference Stage, with any text values of each data point
    converted to a numerical value using the model's text_reference and all values of each data
    point in float form."""
    def process_data_line(self, pre_processed_data_line, text_reference):
        processed_data_line = []
        for i in range(len(pre_processed_data_line)):
            if str(pre_processed_data_line[i]).isalpha():
                for key in text_reference:
                    if str(pre_processed_data_line[i]) == key:
                        processed_data_line.append(float(text_reference[key]))
            else:
                processed_data_line.append(float(pre_processed_data_line[i]))
        return processed_data_line

    """This method assembles and returns the pre-scaled data used by the model
    for training."""
    def assemble_pre_scaled_data(self, training_data_file_name, text_reference):
        pre_scaled_data = []
        file = open(training_data_file_name)
        for line in file:
            pre_processed_data_line = line.strip().split(",")
            processed_data_line = self.process_data_line(pre_processed_data_line, text_reference)
            pre_scaled_data.append(processed_data_line)
        return pre_scaled_data

    """This method returns the split training dataset and validation dataset for the model
    from the data passed into the model for the Training Stage."""
    def split_pre_scaled_data(self, pre_scaled_data, training_data_proportion):
        pre_scaled_training_data = []
        pre_scaled_validation_data = []
        for i in range(len(pre_scaled_data)):
            if i < (len(pre_scaled_data) * training_data_proportion) - 1:
                pre_scaled_training_data.append(pre_scaled_data[i])
            else:
                pre_scaled_validation_data.append(pre_scaled_data[i])
        return pre_scaled_training_data, pre_scaled_validation_data

    """This method computes and returns the mean and standard deviation for each input feature
    data and the target value data in the pre-scaled training data in order to
    scale the data processed by the model using z-score normalization."""
    def compute_z_score_scales(self, pre_scaled_training_data):
        z_score_scales = []
        for i in range(len(pre_scaled_training_data[0])):
            data_list = []
            for j in range(len(pre_scaled_training_data)):
                data_list.append(pre_scaled_training_data[j][i])
            mean = statistics.mean(data_list)
            standard_deviation = statistics.stdev(data_list)
            z_score_scales.append([mean, standard_deviation])
        return z_score_scales

    """This values returns the z-score scaled data point of an unscaled data point from the training dataset
    or validation dataset during the Training Stage, or of a new data point passed into the model during the
    Inference Stage, with each input feature or (if applicable) target value z-score scaled with respect to the
    mean and standard deviation of the values of the respective input feature or target in the training
    dataset."""
    def z_score_scale_data_line(self, data_line, z_score_scales):
        scaled_data_line = []
        for i in range(len(data_line)):
            scaled_data_line.append((data_line[i] - z_score_scales[i][0]) / z_score_scales[i][1])
        return scaled_data_line

    """This method z-score scales the input feature and target values in each data point of the
    training dataset and validation dataset - with use of the z_score_scale_data_line() method -
    during the training stage."""
    def z_score_scale_data(self, pre_scaled_data, z_score_scales):
        scaled_data = []
        for i in range(len(pre_scaled_data)):
            scaled_data.append(self.z_score_scale_data_line(pre_scaled_data[i], z_score_scales))
        return scaled_data

    """This method returns the rescaled the final output of the model for a given data point passed into
    the model during the Inference Stage, converting the z-score scaled output value to the raw output
    value with respect to the mean and standard deviation of the target values in the training dataset."""
    def z_score_rescale_model_output(self, output, z_score_scales):
        return (output * z_score_scales[len(z_score_scales) - 1][1]) + z_score_scales[len(z_score_scales) - 1][0]

    """This method initializes and returns the parameters of the model initializes the parameters of the model
    to uniform random numbers using the He/Kaiming Uniform weight initialization method."""
    def initialize_parameters(self, training_data, model_size, neuron_size_base):
        parameters = []
        for i in range(model_size):
            parameters.append([])
            for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                parameters[i].append([])
                parameters[i][j].append([])
                if i == 0:
                    """Kaiming He Weight Initialization Formula"""
                    limit = math.sqrt(6.0/float(len(training_data[0]) - 1))
                    for k in range(len(training_data[0]) - 1):
                        parameters[i][j][0].append(random.uniform(-1.0 * limit, 1.0 * limit))
                else:
                    """Kaiming He Weight Initialization Formula"""
                    limit = math.sqrt(6.0/float(math.pow(neuron_size_base, model_size - i)))
                    for k in range(int(math.pow(neuron_size_base, model_size - i))):
                        parameters[i][j][0].append(random.uniform(-1.0 * limit, 1.0 * limit))
                parameters[i][j].append(0.0)
                parameters[i][j].append(random.uniform(-1.0 * limit, 1.0 * limit))
        return parameters

    """This method initializes and returns the parameter velocities (update values) of each parameters
    which will be used to keep track of updates to parameters in order to implement Momentum-based
    parameter update regulation (values initialized to 0)."""
    def initialize_parameter_velocities(self, parameters):
        parameter_velocities = []
        for i in range(len(parameters)):
            parameter_velocities.append([])
            for j in range(len(parameters[i])):
                parameter_velocities[i].append([])
                for k in range(len(parameters[i][j][0]) + 2):
                    parameter_velocities[i][j].append(0.0)
        return parameter_velocities

    """This method returns the linear transformation weighted sum of a given neuron"""
    def run_pre_activation_neuron(self, inputs, layer, neuron, parameters):
        sum = 0.0;
        for i in range(len(parameters[layer][neuron][0])):
            sum += inputs[i] * parameters[layer][neuron][0][i]
        sum += parameters[layer][neuron][1] * parameters[layer][neuron][2]
        return sum

    """This method returns the normalized linear transformation weighted sums of all the neurons
    in a given layer (for hidden layers only)."""
    def layer_normalize(self, layer_weighted_sums):
        layer_normalized_sums = []
        mean = statistics.mean(layer_weighted_sums)
        standard_deviation = statistics.stdev(layer_weighted_sums)
        for i in range(len(layer_weighted_sums)):
            layer_normalized_sums.append((layer_weighted_sums[i] - mean) / standard_deviation)
        return layer_normalized_sums

    """This method returns the ReLU activation output of a given neuron's linear transformed
    weighted sum (for hidden layers only)"""
    def run_neuron_ReLU_activation(self, normalized_sum):
            if normalized_sum > 0:
                return normalized_sum
            else:
                return 0.0

    """This method computes and returns the loss value, cost value, and cost derivative value
    with respect to loss for the model for a given data point processed by the model during the
    training stage. L2 Regularization is added to the loss value and the Huber Loss (Cost) Function
    is used to determine the cost value and cost derivative value with respect to loss"""
    def compute_loss_and_cost_values_and_derivative(self, target, output, parameters, delta):
        loss = target - output
        for i in range(len(parameters)):
            for j in range(len(parameters[i])):
                for k in range(len(parameters[i][j][0])):
                    """Addition of L2 Regularization values to the loss value"""
                    loss += math.pow(parameters[i][j][0][k], 2)
        """Use of Huber Loss (Cost) Function"""
        if loss <= delta:
            cost = (0.5) * math.pow(loss, 2)
            cost_derivative_with_respect_to_loss = loss
        else:
            cost = delta * (abs(loss) - (0.5 * delta))
            cost_derivative_with_respect_to_loss = delta * numpy.sign(loss)
        return loss, cost, cost_derivative_with_respect_to_loss

    """This method runs the Forward Pass mechanism of the model. Input feature values from the data point
    passed into the model are processed by the input layer, and the activation values of each neurons
    in each layer - calculated through the run_pre_activation_neuron(), layer_normalize(),
    run_neuron_ReLU_activation() methods, and  move on as inputs for each of the neurons in the last layer
    until the final layer, where the activation output of the final neuron is outputed as the model's
    predicted target value for the data point. Returns the linear transformed weighted sums of each neuron,
    activation values of each neuron, and the model's cost value - through the
    compute_loss_and_cost_values_and_derivative() method."""
    def run_layers(self, initial_inputs, target, model_size, neuron_size_base, parameters, delta):
        weighted_sums = []
        outputs = []
        for i in range(model_size):
            outputs.append([])
            weighted_sums.append([])
            if i == 0 or i == model_size - 1:
                for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                    weighted_sums[i].append(self.run_pre_activation_neuron(initial_inputs, i, j, parameters))
                    outputs[i].append(self.run_pre_activation_neuron(initial_inputs, i, j, parameters))
            elif i != model_size - 1:
                layer_weighted_sums = []
                for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                    weighted_sums[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j, parameters))
                    layer_weighted_sums.append(self.run_pre_activation_neuron(outputs[i - 1], i, j, parameters))
                layer_normalized_sums = self.layer_normalize(layer_weighted_sums)
                for j in range(len(layer_normalized_sums)):
                    outputs[i].append(self.run_neuron_ReLU_activation(layer_normalized_sums[j]))
            else:
                for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                    weighted_sums[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j))
                    outputs[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j))
        _, cost, _ = self.compute_loss_and_cost_values_and_derivative(target, outputs[model_size - 1][0], parameters, delta)
        return weighted_sums, outputs, cost

    """This method runs the Backpropagation and Gradient Descent mechanism of the model after it
    processes a given data point from the training dataset. The gradient for each parameter in the model
    is computed based on the model's cost value and cost derivative value with respect to loss. The velocity
    (update shift) for each parameter is regulated through the model's Gradient Clipping and Momentum mechanisms.
    Learning rate is adjusted based on its decay value in order for the model to converge (towards minimum cost) faster.
    Each parameter is then updated with its respective final velocity. The parameter velocities of the model is
    return back."""
    def run_back_propagation_and_gradient_descent(self, initial_inputs, target, outputs, weighted_sums, parameters, parameter_velocities, model_size, neuron_size_base, delta, adjusted_learning_rate, momentum_factor, max_norm_benchmark):
        loss, cost, cost_derivative_with_respect_to_loss = self.compute_loss_and_cost_values_and_derivative(target, outputs[model_size - 1][0], parameters, delta)
        loss_derivative_with_respect_to_neuron_activation_outputs = []
        neuron_activation_output_derivative_with_respect_to_neuron_sums = []
        for i in range(model_size):
            loss_derivative_with_respect_to_neuron_activation_outputs.append([])
            neuron_activation_output_derivative_with_respect_to_neuron_sums.append([])
            for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                loss_derivative_with_respect_to_neuron_activation_outputs[i].append([])
                neuron_activation_output_derivative_with_respect_to_neuron_sums[i].append([])
        for i in reversed(range(model_size)):
            for j in range(int(math.pow(neuron_size_base, model_size - 1 - i))):
                neuron_gradients = []
                scaled_neuron_gradients = []
                if i == model_size - 1:
                    loss_derivative_with_respect_to_neuron_activation_output = -1.0
                else:
                    loss_derivative_with_respect_to_neuron_activation_output = 0
                    for k in range(len(parameters[i + 1])):
                        loss_derivative_with_respect_to_neuron_activation_output += (loss_derivative_with_respect_to_neuron_activation_outputs[i + 1][k] * neuron_activation_output_derivative_with_respect_to_neuron_sums[i + 1][k] * parameters[i + 1][k][0][j])
                loss_derivative_with_respect_to_neuron_activation_outputs[i][j] = loss_derivative_with_respect_to_neuron_activation_output
                if i == 0 or i == model_size - 1 or weighted_sums[i][j] > 0:
                    neuron_activation_output_derivative_with_respect_to_neuron_sum = 1.0
                else:
                    neuron_activation_output_derivative_with_respect_to_neuron_sum = 0.0
                neuron_activation_output_derivative_with_respect_to_neuron_sums[i][j] = neuron_activation_output_derivative_with_respect_to_neuron_sum
                for k in range(len(parameters[i][j][0])):
                    if i == 0:
                        neuron_sum_derivative_with_respect_to_input_weight = initial_inputs[k]
                    else:
                        neuron_sum_derivative_with_respect_to_input_weight = outputs[i - 1][k]
                    cost_derivative_with_respect_to_input_weight = cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_input_weight
                    neuron_gradients.append(cost_derivative_with_respect_to_input_weight)
                neuron_sum_derivative_with_respect_to_bias_weight = parameters[i][j][1]
                cost_derivative_with_respect_to_bias_weight = cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_bias_weight
                neuron_gradients.append(cost_derivative_with_respect_to_bias_weight)
                neuron_sum_derivative_with_respect_to_bias = 1.0
                cost_derivative_with_respect_to_bias = cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_bias
                neuron_gradients.append(cost_derivative_with_respect_to_bias)
                l2_norm = math.sqrt(sum(math.pow(x, 2) for x in neuron_gradients))
                max_norm = numpy.percentile(neuron_gradients, max_norm_benchmark)
                for k in range(len(neuron_gradients)):
                    if l2_norm > max_norm:
                        scaled_neuron_gradients.append(neuron_gradients[k] * (max_norm / l2_norm))
                    else:
                        scaled_neuron_gradients.append(neuron_gradients[k])
                for k in range(len(parameter_velocities[i][j])):
                    parameter_velocities[i][j][k] = (parameter_velocities[i][j][k] * momentum_factor) - (scaled_neuron_gradients[k] * adjusted_learning_rate)
                    if k < len(parameter_velocities[i][j]) - 2:
                        parameters[i][j][0][k] += parameter_velocities[i][j][k]
                    elif k == len(parameter_velocities[i][j]) - 2:
                        parameters[i][j][2] += parameter_velocities[i][j][k]
                    else:
                        parameters[i][j][1] += parameter_velocities[i][j][k]
        return parameter_velocities

    """This method trains the model based on the data it has received for the Training Stage.
    Forward Pass, Backpropagation, and Gradient Descent is executed for each data point processed
    by the model from the training dataset through the run_layers() and run_back_propagation_and_gradient_descent().
    The data points from the training data is processed by the model for a number of epochs specified by the model's
    training_epochs hyperparameter. After each epoch, the model's updated parameters are tested on the validation data
    and the average cost across the data points of the validation dataset is computed. The model's parameter are restored
    to the epoch version with the lowest average cost on the validation data."""
    def train_model(self, training_epochs, training_data, parameters, parameter_velocities, validation_data, model_size, neuron_size_base, delta, learning_rate, learning_rate_decay_rate, momentum_factor, max_norm_benchmark):
        parameter_epoch_versions = []
        average_validation_cost_value_over_epochs = []
        update_step = 0
        for i in range(training_epochs):
            random.shuffle(training_data)
            for j in range(len(training_data)):
                adjusted_learning_rate = learning_rate / (1.0 + (learning_rate_decay_rate * update_step))
                initial_inputs = training_data[i][0:len(training_data[i]) - 1]
                target = training_data[i][-1]
                weighted_sums, outputs, cost = self.run_layers(initial_inputs, target, model_size, neuron_size_base, parameters, delta)
                parameter_velocities = self.run_back_propagation_and_gradient_descent(initial_inputs, target, outputs, weighted_sums, parameters, parameter_velocities, model_size, neuron_size_base, delta, adjusted_learning_rate, momentum_factor, max_norm_benchmark)
                update_step += 1
            parameter_epoch_versions.append(copy.deepcopy(parameters))
            validation_cost_values_over_epoch = []
            for j in range(len(validation_data)):
                initial_inputs = validation_data[i][0:len(training_data[i]) - 1]
                target = validation_data[i][-1]
                _, _, cost = self.run_layers(initial_inputs, target, model_size, neuron_size_base, parameters, delta)
                validation_cost_values_over_epoch.append(cost)
            average_validation_cost_value_over_epoch = statistics.mean(validation_cost_values_over_epoch)
            average_validation_cost_value_over_epochs.append(average_validation_cost_value_over_epoch)
        minimum_cost_index = 0
        for i in range(len(average_validation_cost_value_over_epochs)):
            if average_validation_cost_value_over_epochs[i] < average_validation_cost_value_over_epochs[minimum_cost_index]:
                minimum_cost_index = i
        parameters = parameter_epoch_versions[minimum_cost_index]
        return average_validation_cost_value_over_epochs, minimum_cost_index

    """Private Methods - To Be Used Internally By the User"""

    """Runs the model on a new data point during the inference stage. The data point is processed and z-score
    scaled with respect to the mean and standard deviation of the training dataset through the process_data_line()
    and z_score_scale_data_line() methods respectively. A predicted target value for the data point is computed through
    the run_layers() function. The model's predicted value is rescaled through the z_score_rescale_model_output() method."""
    def run_model(self, inputs):
        processed_inputs = self.process_data_line(inputs, self.text_reference)
        scaled_processed_inputs = self.z_score_scale_data_line(processed_inputs, self.z_score_scales)
        _, outputs, _ = self.run_layers(scaled_processed_inputs, 1, self.model_size, self.neuron_size_base, self.parameters, self.delta)
        final_output = self.z_score_rescale_model_output(outputs[self.model_size - 1][0], self.z_score_scales)
        return final_output

    """This method returns the text_reference hyperparameter of the model."""
    def get_text_reference(self):
        return self.text_reference

    """This method returns the delta hyperparameter of the model."""
    def get_delta(self):
        return self.delta

    """This method returns the learning_rate hyperparameter of the model."""
    def get_learning_rate(self):
        return self.learning_rate

    """This method returns the learning_rate_decay_rate hyperparameter of the model."""
    def get_learning_rate_decay_rate(self):
        return self.learning_rate_decay_rate

    """This method returns the max_norm_benchmark hyperparameter of the model."""
    def get_max_norm_benchmark(self):
        return self.max_norm_benchmark

    """This method returns the momentum_factor hyperparameter of the model."""
    def get_momentum_factor(self):
        return self.momentum_factor

    """This method returns the model_size hyperparameter of the model."""
    def get_model_size(self):
        return self.model_size

    """This method returns the neuron_size_base hyperparameter of the model."""
    def get_neuron_size_base(self):
        return self.neuron_size_base

    """This method returns the training_epochs hyperparameter of the model."""
    def get_training_epochs(self):
        return self.training_epochs

    """This method returns the training_data_proportion hyperparameter of the model."""
    def get_training_data_proportions(self):
        return self.training_data_proportion

    """This method returns the average_validation_cost_value_over_epochs hyperparameter of the model."""
    def get_average_validation_cost_value_over_epochs(self):
        return self.average_validation_cost_value_over_epochs

    """This method returns the z_score_scales hyperparameter (the means and standard deviations of the values
    of each input feature and target in the training dataset)  of the model."""
    def get_z_score_scales(self):
        return self.z_score_scales

    """This method returns the minimum_cost_index hyperparameter (the epoch at which the model
    achieved the lowest cost value from the training dataset - indexed to 0)  of the model."""
    def get_minimum_cost_index(self):
        return self.minimum_cost_index

    """This method returns the training_data dataset of the model."""
    def get_training_data(self):
        return self.training_data

    """This method returns the validation_data dataset of the model."""
    def get_validation_data(self):
        return self.validation_data

