import math, random, statistics as stats, numpy as np, copy

class NeuralNetworkModel:
    def __init__(self, input_data, target_data, hidden_layers = 4, neuron_size_base = 2, training_epochs = 75, training_data_proportion = 0.75, delta = 1.0, learning_rate = 0.01, learning_rate_decay_rate = 0.001, momentum_factor = 0.9, max_norm_benchmark = 90, l2 = 0.01, leaky_relu_slope = 0.01, random_state = None):
        self.random_state = random_state
        if self.random_state != None:
            random.seed(self.random_state)
        self.input_data = input_data
        self.target_data = target_data
        self.delta = delta
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.max_norm_benchmark = max_norm_benchmark
        self.momentum_factor = momentum_factor
        self.l2 = l2
        self.hidden_layers = hidden_layers
        self.model_depth = self.hidden_layers + 1
        self.neuron_size_base = neuron_size_base
        self.training_epochs = training_epochs
        self.training_data_proportion = training_data_proportion
        self.leaky_relu_slope = leaky_relu_slope
        self.pre_scaled_training_data, self.validation_data = self.assemble_pre_scaled_data(self.input_data, self.target_data, self.training_data_proportion)
        self.z_score_scales = self.compute_z_score_scales(self.pre_scaled_training_data)
        self.training_data = self.z_score_scale_training_data(self.pre_scaled_training_data, self.z_score_scales)
        self.parameters = self.initialize_parameters(self.training_data, self.model_depth, self.neuron_size_base)
        self.parameter_velocities = self.initialize_parameter_velocities(self.parameters, self.model_depth, self.neuron_size_base)
        self.parameters, self.validation_mses_over_epochs, self.minimum_cost_index, self.training_cost_values_over_epochs = self.train_model(self.training_epochs, self.training_data, self.parameters, self.parameter_velocities, self.validation_data, self.model_depth, self.neuron_size_base, self.delta, self.learning_rate, self.learning_rate_decay_rate, self.momentum_factor, self.max_norm_benchmark, self.l2, self.leaky_relu_slope)

    """This method assembles and returns the pre-scaled training and validation data
    used by the model for training."""
    def assemble_pre_scaled_data(self, input_data, target_data, training_data_proportion):
        pre_scaled_data = []
        for i in range(len(input_data)):
            pre_scaled_data.append([input_data[i], target_data[i]])
        pre_scaled_training_data = pre_scaled_data[:int(len(pre_scaled_data) * training_data_proportion)]
        pre_scaled_validation_data = pre_scaled_data[int(len(pre_scaled_data) * training_data_proportion):]
        return pre_scaled_training_data, pre_scaled_validation_data

    """This method computes and returns the mean and standard deviation for each input feature
    data and the target value data in the pre-scaled training data in order to
    scale the data processed by the model using z-score normalization."""
    def compute_z_score_scales(self, pre_scaled_training_data):
        z_score_scales = [[]]
        for i in range(len(pre_scaled_training_data[0][0])):
            data_list = []
            for j in range(len(pre_scaled_training_data)):
                data_list.append(pre_scaled_training_data[j][0][i])
            z_score_scales[0].append([stats.mean(data_list), stats.stdev(data_list)])
        data_list = []
        for j in range(len(pre_scaled_training_data)):
            data_list.append(pre_scaled_training_data[j][1])
        z_score_scales.append([stats.mean(data_list), stats.stdev(data_list)])
        return z_score_scales

    """This values returns the z-score scaled data point of an unscaled data point from the training dataset
    or validation dataset during the Training Stage, or of a new data point passed into the model during the
    Inference Stage, with each input feature or (if applicable) target value z-score scaled with respect to the
    mean and standard deviation of the values of the respective input feature or target in the training
    dataset."""
    def z_score_scale_data_line(self, data_line, z_score_scales, target):
        if target:
            scaled_data_line = [[]]
            for i in range(len(data_line[0])):
                scaled_data_line[0].append((data_line[0][i] - z_score_scales[0][i][0]) / z_score_scales[0][i][1])
            scaled_data_line.append((data_line[1] - z_score_scales[1][0]) / z_score_scales[1][1])
        else:
            scaled_data_line = []
            for i in range(len(data_line)):
                scaled_data_line.append((data_line[i] - z_score_scales[0][i][0]) / z_score_scales[0][i][1])
        return scaled_data_line

    """This method z-score scales the input feature and target values in each data point of the
    training dataset dataset - with use of the z_score_scale_data_line() method -
    during the training stage."""
    def z_score_scale_training_data(self, pre_scaled_data, z_score_scales):
        scaled_data = []
        for i in range(len(pre_scaled_data)):
            scaled_data.append(self.z_score_scale_data_line(pre_scaled_data[i], z_score_scales, True))
        return scaled_data

    """This method returns the rescaled the final output of the model for a given data point passed into
    the model during the Inference Stage, converting the z-score scaled output value to the raw output
    value with respect to the mean and standard deviation of the target values in the training dataset."""
    def z_score_rescale_model_output(self, output, z_score_scales):
        return (output * z_score_scales[1][1]) + z_score_scales[1][0]

    """This method initializes and returns the parameters of the model initializes the parameters of the model
    to uniform random numbers using the He/Kaiming Uniform weight initialization method."""
    def initialize_parameters(self, training_data, model_depth, neuron_size_base):
        parameters = []
        for i in range(model_depth):
            parameters.append([])
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                parameters[i].append([])
                parameters[i][j].append([])
                if i == 0:
                    """Kaiming He Weight Initialization Formula"""
                    std_dev = math.sqrt(2.0 / (len(training_data[0]) - 1))
                    for k in range(len(training_data[0][0])):
                        parameters[i][j][0].append(np.random.normal(0, std_dev))
                else:
                    """Kaiming He Weight Initialization Formula"""
                    std_dev = math.sqrt(2.0/float(len(training_data[0]) - 1))
                    for k in range(int(math.pow(neuron_size_base, model_depth - i))):
                        parameters[i][j][0].append(np.random.normal(0, std_dev))
                parameters[i][j].append(0.0)
            if i != model_depth - 1:
                parameters[i].append(np.random.normal(0, std_dev))
                parameters[i].append(0.0)
        return parameters

    """This method initializes and returns the parameter velocities (update values) of each parameters
    which will be used to keep track of updates to parameters in order to implement Momentum-based
    parameter update regulation (values initialized to 0)."""
    def initialize_parameter_velocities(self, parameters, model_depth, neuron_size_base):
        parameter_velocities = []
        for i in range(model_depth):
            parameter_velocities.append([])
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                parameter_velocities[i].append([])
                parameter_velocities[i][j].append([])
                for k in range(len(parameters[i][j][0])):
                    parameter_velocities[i][j][0].append(0.0)
                parameter_velocities[i][j].append(0.0)
            for j in range(2):
                parameter_velocities[i].append(0.0)
        return parameter_velocities

    """This method returns the linear transformation weighted sum of a given neuron"""
    def run_pre_activation_neuron(self, inputs, layer_index, neuron_index, parameters):
        sum = 0.0;
        for i in range(len(inputs)):
            sum += inputs[i] * parameters[layer_index][neuron_index][0][i]
        sum += parameters[layer_index][neuron_index][1]
        return sum

    """This method returns the normalized linear transformation weighted sums of all the neurons
    in a given layer (for hidden layers only)."""
    def layer_normalize(self, layer_sums, parameters, layer_index):
        layer_normalized_sums = []
        mean = stats.mean(layer_sums)
        standard_deviation = stats.stdev(layer_sums)
        layer_data = [mean, standard_deviation]
        ld = parameters[layer_index][-2]
        b = parameters[layer_index][-1]
        for i in range(len(layer_sums)):
            layer_normalized_sums.append(ld * ((layer_sums[i] - mean) / standard_deviation) + b)
        return layer_normalized_sums, layer_data

    """This method returns the ReLU activation output of a given neuron's linear transformed
    weighted sum (for hidden layers only)"""
    def run_neuron_ReLU_activation(self, normalized_sum, leaky_relu_slope):
            if normalized_sum > 0:
                return normalized_sum
            else:
                return leaky_relu_slope * normalized_sum

    """This method computes and returns the loss value, cost value, and cost derivative value
    with respect to loss for the model for a given data point processed by the model during the
    training stage. L2 Regularization is added to the loss value and the Huber Loss (Cost) Function
    is used to determine the cost value and cost derivative value with respect to loss"""
    def compute_loss_and_cost_values_and_derivative(self, target, output, parameters, delta, l2, model_depth, neuron_size_base):
        loss = target - output
        """L2 Regularization"""
        input_weight_sums = 0
        for i in range(model_depth):
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                for k in range(len(parameters[i][j][0])):
                    input_weight_sums += math.pow(parameters[i][j][0][k], 2)
        l2Regularization = l2 * input_weight_sums
        """Use of Huber Loss (Cost) Function with L2 Regularization"""
        if loss <= delta:
            cost = ((0.5) * math.pow(loss, 2)) + l2Regularization
            cost_derivative_with_respect_to_loss = loss
        else:
            cost = (delta * (abs(loss) - (0.5 * delta))) + l2Regularization
            cost_derivative_with_respect_to_loss = delta * np.sign(loss)
        return loss, cost, cost_derivative_with_respect_to_loss

    """This method runs the Forward Pass mechanism of the model. Input feature values from the data point
    passed into the model are processed by the input layer, and the activation values of each neurons
    in each layer - calculated through the run_pre_activation_neuron(), layer_normalize(),
    run_neuron_ReLU_activation() methods, and  move on as inputs for each of the neurons in the last layer
    until the final layer, where the activation output of the final neuron is outputed as the model's
    predicted target value for the data point. Returns the linear transformed weighted sums of each neuron,
    activation values of each neuron, and the model's cost value - through the
    compute_loss_and_cost_values_and_derivative() method."""
    def run_layers(self, initial_inputs, model_depth, neuron_size_base, parameters, leaky_relu_slope):
        sums = []
        outputs = []
        layer_datas = []
        for i in range(model_depth):
            outputs.append([])
            sums.append([])
            if i == 0:
                for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                    sums[i].append(self.run_pre_activation_neuron(initial_inputs, i, j, parameters))
                sums[i], layer_data = self.layer_normalize(sums[i], parameters, i)
                layer_datas.append(layer_data)
                for j in range(len(sums[i])):
                    outputs[i].append(self.run_neuron_ReLU_activation(sums[i][j], leaky_relu_slope))
            elif i != model_depth - 1:
                for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                    sums[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j, parameters))
                sums[i], layer_data = self.layer_normalize(sums[i], parameters, i)
                layer_datas.append(layer_data)
                for j in range(len(sums[i])):
                    outputs[i].append(self.run_neuron_ReLU_activation(sums[i][j], leaky_relu_slope))
            else:
                for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                    sums[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j, parameters))
                    outputs[i].append(self.run_pre_activation_neuron(outputs[i - 1], i, j, parameters))
        return sums, outputs, layer_datas

    def clip_gradients(self, current_parameter_velocities, model_depth, neuron_size_base, max_norm_benchmark):
        gradients = []
        current_parameter_velocities_clipped = []
        for i in range(model_depth):
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                for k in range(len(current_parameter_velocities[i][j][0])):
                    gradients.append(current_parameter_velocities[i][j][0][k])
                gradients.append(current_parameter_velocities[i][j][1])
            if i != model_depth - 1:
                gradients.append(current_parameter_velocities[i][-2])
                gradients.append(current_parameter_velocities[i][-1])
        l2_norm = math.sqrt(sum(math.pow(x, 2) for x in gradients))
        max_norm = np.percentile(gradients, max_norm_benchmark)
        if l2_norm > max_norm:
            scaler = (max_norm / l2_norm)
        else:
            scaler = 1
        for i in range(model_depth):
            current_parameter_velocities_clipped.append([])
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                current_parameter_velocities_clipped[i].append([])
                current_parameter_velocities_clipped[i][j].append([])
                for k in range(len(current_parameter_velocities[i][j][0])):
                    current_parameter_velocities_clipped[i][j][0].append(current_parameter_velocities[i][j][0][k] * scaler)
                current_parameter_velocities_clipped[i][j].append(current_parameter_velocities[i][j][1] * scaler)
            if i != model_depth - 1:
                current_parameter_velocities_clipped[i].append(current_parameter_velocities[i][-2] * scaler)
                current_parameter_velocities_clipped[i].append(current_parameter_velocities[i][-1] * scaler)
        return current_parameter_velocities_clipped

    """This method runs the Backpropagation and Gradient Descent mechanism of the model after it
    processes a given data point from the training dataset. The gradient for each parameter in the model
    is computed based on the model's cost value and cost derivative value with respect to loss. The velocity
    (update shift) for each parameter is regulated through the model's Gradient Clipping and Momentum mechanisms.
    Learning rate is adjusted based on its decay value in order for the model to converge (towards minimum cost) faster.
    Each parameter is then updated with its respective final velocity. The parameter velocities of the model is
    return back."""
    def run_back_propagation_and_gradient_descent(self, loss, cost, cost_derivative_with_respect_to_loss, initial_inputs, outputs, sums, parameters, parameter_velocities, model_depth, neuron_size_base, delta, adjusted_learning_rate, momentum_factor, max_norm_benchmark, l2, layer_datas, leaky_relu_slope):
        """Back Propagation"""
        current_parameter_velocities = []
        loss_derivative_with_respect_to_neuron_activation_outputs = []
        neuron_activation_output_derivative_with_respect_to_neuron_sums = []
        for i in range(model_depth):
            current_parameter_velocities.append([])
            loss_derivative_with_respect_to_neuron_activation_outputs.append([])
            neuron_activation_output_derivative_with_respect_to_neuron_sums.append([])
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                current_parameter_velocities[i].append([])
                current_parameter_velocities[i][j].append([])
                loss_derivative_with_respect_to_neuron_activation_outputs[i].append([])
                neuron_activation_output_derivative_with_respect_to_neuron_sums[i].append([])
        for i in reversed(range(model_depth)):
            if i != model_depth - 1:
                cost_derivative_with_respect_to_ld = 0
                cost_derivative_with_respect_to_b = 0
                layer_data = layer_datas[i]
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                neuron_gradients = []
                if i == model_depth - 1:
                    loss_derivative_with_respect_to_neuron_activation_output = -1.0
                else:
                    loss_derivative_with_respect_to_neuron_activation_output = 0
                    for k in range(int(math.pow(neuron_size_base, model_depth - 2 - i))):
                        loss_derivative_with_respect_to_neuron_activation_output += (loss_derivative_with_respect_to_neuron_activation_outputs[i + 1][k] * neuron_activation_output_derivative_with_respect_to_neuron_sums[i + 1][k] * parameters[i + 1][k][0][j])
                loss_derivative_with_respect_to_neuron_activation_outputs[i][j] = loss_derivative_with_respect_to_neuron_activation_output
                if i == 0 or i == model_depth - 1 or sums[i][j] > 0:
                    neuron_activation_output_derivative_with_respect_to_neuron_sum = 1.0
                else:
                    neuron_activation_output_derivative_with_respect_to_neuron_sum = leaky_relu_slope
                neuron_activation_output_derivative_with_respect_to_neuron_sums[i][j] = neuron_activation_output_derivative_with_respect_to_neuron_sum
                for k in range(len(parameters[i][j][0])):
                    if i == 0:
                        neuron_sum_derivative_with_respect_to_input_weight =  initial_inputs[k]
                    else:
                        neuron_sum_derivative_with_respect_to_input_weight = outputs[i - 1][k]
                    if i != model_depth - 1:
                        neuron_sum_derivative_with_respect_to_input_weight *= (parameters[i][-2] / layer_data[1])
                    cost_derivative_with_respect_to_input_weight = (cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_input_weight) + (2 * l2 * parameters[i][j][0][k])
                    current_parameter_velocities[i][j][0].append(cost_derivative_with_respect_to_input_weight)
                if i != model_depth - 1:
                    neuron_sum_derivative_with_respect_to_bias = (parameters[i][-2] / layer_data[1])
                else:
                    neuron_sum_derivative_with_respect_to_bias = 1
                cost_derivative_with_respect_to_bias = cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_bias
                current_parameter_velocities[i][j].append(cost_derivative_with_respect_to_bias)
                if i != model_depth - 1:
                    neuron_sum_derivative_with_respect_to_ld = (sums[i][j] - parameters[i][-1]) / parameters[i][-2]
                    cost_derivative_with_respect_to_ld += cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum * neuron_sum_derivative_with_respect_to_ld
                    cost_derivative_with_respect_to_b += cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_neuron_activation_output * neuron_activation_output_derivative_with_respect_to_neuron_sum
            if i != model_depth - 1:
                current_parameter_velocities[i].append(cost_derivative_with_respect_to_ld)
                current_parameter_velocities[i].append(cost_derivative_with_respect_to_b)
        current_parameter_velocities_clipped = self.clip_gradients(current_parameter_velocities, model_depth, neuron_size_base, max_norm_benchmark)
        for i in reversed(range(model_depth)):
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                for k in range(len(parameter_velocities[i][j][0])):
                    parameter_velocities[i][j][0][k] = (parameter_velocities[i][j][0][k] * momentum_factor) - (current_parameter_velocities_clipped[i][j][0][k] * adjusted_learning_rate)
                parameter_velocities[i][j][1] = (parameter_velocities[i][j][1] * momentum_factor) - (current_parameter_velocities_clipped[i][j][1] * adjusted_learning_rate)
            if i != model_depth - 1:
                parameter_velocities[i][-2] = (parameter_velocities[i][-2] * momentum_factor) - (current_parameter_velocities_clipped[i][-2] * adjusted_learning_rate)
                parameter_velocities[i][-1] = (parameter_velocities[i][-1] * momentum_factor) - (current_parameter_velocities_clipped[i][-1] * adjusted_learning_rate)
        """Gradient Descent, using the calculated parameter velocity values for each parameter in each neuron in the model
        from Back Propagation"""
        for i in reversed(range(model_depth)):
            for j in range(int(math.pow(neuron_size_base, model_depth - 1 - i))):
                for k in range(len(parameter_velocities[i][j][0])):
                    parameters[i][j][0][k] += parameter_velocities[i][j][0][k]
                parameters[i][j][1] += parameter_velocities[i][j][1]
            if i != model_depth - 1:
                parameters[i][-2] += parameter_velocities[i][-2]
                parameters[i][-1] += parameter_velocities[i][-1]
        return parameters

    """This method trains the model based on the data it has received for the Training Stage.
    Forward Pass, Backpropagation, and Gradient Descent is executed for each data point processed
    by the model from the training dataset through the run_layers() and run_back_propagation_and_gradient_descent().
    The data points from the training data is processed by the model for a number of epochs specified by the model's
    training_epochs hyperparameter. After each epoch, the model's updated parameters are tested on the validation data
    and the average cost across the data points of the validation dataset is computed. The model's parameter are restored
    to the epoch version with the lowest average cost on the validation data."""
    def train_model(self, training_epochs, training_data, parameters, parameter_velocities, validation_data, model_depth, neuron_size_base, delta, learning_rate, learning_rate_decay_rate, momentum_factor, max_norm_benchmark, l2, leaky_relu_slope):
        parameter_epoch_versions = []
        """average_validation_cost_values_over_epochs = []"""
        training_cost_values_over_epochs = []
        validation_mses_over_epochs = []
        update_step = 0
        for i in range(training_epochs):
            r = int(random.random() * len(training_data))
            initial_inputs = training_data[r][0]
            target = training_data[r][1]
            sums, outputs, layer_datas = self.run_layers(initial_inputs, model_depth, neuron_size_base, parameters, leaky_relu_slope)
            loss, cost, cost_derivative_with_respect_to_loss = self.compute_loss_and_cost_values_and_derivative(target, outputs[model_depth - 1][0], parameters, delta, l2, model_depth, neuron_size_base)
            training_cost_values_over_epochs.append(cost)
            adjusted_learning_rate = learning_rate / (1.0 + (learning_rate_decay_rate * update_step))
            parameters = self.run_back_propagation_and_gradient_descent(loss, cost, cost_derivative_with_respect_to_loss, initial_inputs, outputs, sums, parameters, parameter_velocities, model_depth, neuron_size_base, delta, adjusted_learning_rate, momentum_factor, max_norm_benchmark, l2, layer_datas, leaky_relu_slope)
            update_step += 1
            parameter_epoch_versions.append(copy.deepcopy(parameters))
            validation_ses_over_epoch = []
            for j in range(len(validation_data)):
                initial_inputs = validation_data[j][0]
                target = validation_data[j][1]
                _, outputs, _ = self.run_layers(initial_inputs, model_depth, neuron_size_base, parameters, leaky_relu_slope)
                prediction = outputs[model_depth - 1][0]
                validation_ses_over_epoch.append(pow(target - prediction, 2))
            validation_mses_over_epochs.append(stats.mean(validation_ses_over_epoch))
        minimum_cost_index = 0
        for i in range(len(validation_mses_over_epochs)):
            if validation_mses_over_epochs[i] < validation_mses_over_epochs[minimum_cost_index]:
                minimum_cost_index = i
        parameters = parameter_epoch_versions[minimum_cost_index]
        return parameters, validation_mses_over_epochs, minimum_cost_index, training_cost_values_over_epochs

    """Runs the model on a list of input data during the inference stage. Each input data is processed and z-score
    scaled with respect to the mean and standard deviation of the training dataset through the process_data_line()
    and z_score_scale_data_line() methods respectively. Each predicted target value for a given input data is computed through
    the run_layers() function, and then rescaled through the z_score_rescale_model_output() method. A list of predicted targets
    corresponding to the list of input data is returned"""
    def run_model(self, input_data_list):
        final_outputs = []
        for i in range(len(input_data_list)):
            scaled_inputs = self.z_score_scale_data_line(input_data_list[i], self.z_score_scales, False)
            _, outputs, _ = self.run_layers(scaled_inputs, self.model_depth, self.neuron_size_base, self.parameters, self.leaky_relu_slope)
            final_outputs.append(self.z_score_rescale_model_output(outputs[self.model_depth - 1][0], self.z_score_scales))
        return final_outputs

    """This method returns the random_state hyperparameter of the model."""
    def get_random_state(self):
        return self.random_state

    """This method returns the input_data of the model."""
    def get_input_data(self):
        return self.input_data

    """This method returns the target_data of the model."""
    def get_target_data(self):
        return self.target_data

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

    """This method returns the l2 hyperparameter of the model."""
    def get_l2(self):
        return self.l2

    """This method returns the hidden_layers hyperparameter of the model."""
    def get_model_depth(self):
        return self.hidden_layers

    """This method returns the model_depth attribute of the model."""
    def get_model_depth(self):
        return self.model_depth

    """This method returns the neuron_size_base hyperparameter of the model."""
    def get_neuron_size_base(self):
        return self.neuron_size_base

    """This method returns the training_epochs hyperparameter of the model."""
    def get_training_epochs(self):
        return self.training_epochs

    """This method returns the training_data_proportion hyperparameter of the model."""
    def get_training_data_proportion(self):
        return self.training_data_proportion

    """This method returns the leaky_relu_slope hyperparameter of the model."""
    def get_leaky_relu_slope(self):
        return self.leaky_relu_slope

    """This method returns the pre_scaled_training_data of the model."""
    def get_pre_scaled_training_data(self):
        return self.pre_scaled_training_data

    """This method returns the pre_scaled_validation_data of the model."""
    def get_pre_scaled_validation_data(self):
        return self.pre_scaled_validation_data

    """This method returns the z_score_scales hyperparameter (the means and standard deviations of the values
    of each input feature and target in the training dataset)  of the model."""
    def get_z_score_scales(self):
        return self.z_score_scales

    """This method returns the training_data of the model."""
    def get_training_data(self):
        return self.training_data

    """This method returns the validation_data of the model."""
    def get_validation_data(self):
        return self.validation_data

    """This method returns the parameters of the model."""
    def get_parameters(self):
        return self.parameters

    """This method returns the parameter_velocities of the model."""
    def get_parameter_velocities(self):
        return self.parameter_velocities

    """This method returns the validation_mses_over_epochs (the average validation cost of the model
    across all epochs) of the model."""
    def get_validation_mses_over_epochs(self):
        return self.validation_mses_over_epochs

    """This method returns the minimum_cost_index (the epoch at which the model
    achieved the lowest validation cost from the training dataset - indexed to 0)  of the model."""
    def get_minimum_cost_index(self):
        return self.minimum_cost_index

    """This method returns the training_cost_values_over_epochs (the training cost of the model
    across all epochs) of the model."""
    def get_training_cost_values_over_epochs(self):
        return self.training_cost_values_over_epochs
