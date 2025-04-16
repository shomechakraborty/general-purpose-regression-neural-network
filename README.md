# general-purpose-regression-neural-network

Programmed By: Shome Chakraborty

INTRODUCTION

This program is a general purpose regression neural network model which can be used to predict target values using data in a variety of quantitative applications. This model is designed to capture and learn from complex relationships and patterns between quantitative data in order to develop a robust predictive ability beyond traditional regression. Users can provide the model training data and several hyperparameters based on their specific predictive application which the model will train on. Users can run their custom trained model on input features in order to obtain predictive values for their respective use-case or application. The model is designed to be fine-tuned to a user’s application and preferences, while providing for optimized regression training and data analysis.

This model is inspired by a previous neural network model I developed, which was designed to specialize in the prediction of housing prices. I had fixed, adjusted, and built upon core feature of that model in the development of this program, providing more robust and accurate functionality for this program.

FUNCTIONALITY

Program Structure:
This model was developed using object-oriented programming in Python
This programming structure allows users to generate custom model(s) (or model object(s)) trained on the specific data and hyperparameters provided by the user and which can run and develop predictions on new input data as many times as needed by the user

Model Architecture:

The number of layers in the model is determined by the user
The number of neurons in each layer is exponentially determined based on the base rule given by the user and the layer number (ending at 0).
For example, a model with a model size of 5 and a base rule of 2 provided by the user will have a layout of 2^4 X 2^3 X 2^2 X 2^1 X 2^0 = 16 X 8 X 4 X 2 X 1 (5 layers, 31 total neurons)
Hyper-Parameters (Provided By Users in Addition to Dataset)
Text Reference (Default Value - None) - An optional dictionary of text values to respective numerical values the user can put in if the data they provide contains text values - necessary for quantitative processing and analysis)
Model Size (Default Value - 5) - The number of layers in the neuron
Neuron Size Base (Default Value - 2) - The base rule for the number of neurons in each layer
Training Epochs (Default Value - 50) - The number of epochs (times) the model will train on the training data
Training Data Proportion - The portion of data provided by the user which will be the training data (with the remainder being used for validation)
Delta (Default Value - 1.0) - A threshold value for loss which determines whether the cost function should switch to the absolute value of loss from squared loss during Gradient Descent (under the Huber Loss Function)
Learning Rate (Default Value - 0.001) - The rate at which gradients for weight updates are updated during Gradient Descent 
Learning Rate Decay Rate (Default Value - 10.0 ^ -4) - The rate at which the learning rate is decayed each training instance 
Momentum Factor (Default Value - 0.9) - The portion (decimal value) of accumulated gradients from past updates that is added to the weight update 
Max Norm Percentile Threshold (Default Value - 90) - The percentile (percent value) of gradients that will be used to determine the max norm (maximum value a gradient can be) for gradients in each neuron during each training instance
The use of default values provides the user the added convenience of not having to input values for all the hyperparameters if they choose to stick with the default value
Each hyperparameter has an associated getter method which can be used by the user to retrieve its value

Model Process

During the training stage of the model as data and hyper parameters are passed into it, the model will pass in labeled data points (with a target value) from the training dataset - a specific portion of the data passed into it based on the Training Data Proportion hyperparameter value imputed by the user, with the remainder being used for the model’s validation process - and in the several processes described below (Z Score Normalization to Forward Pass), will process, analyze, and develop a predicted output value based on the input feature values of each data point
After the model develops a predicted output value for each data point passed in from the training dataset, it will then take several steps to update its parameters based on the difference of its predicted value and the actual target (or label value) of the data point (Backward Propagation to Gradient Descent)
The model will process all data points from the training dataset for a given number of Epochs based on the Training Epochs hyperparameter provided by the user
The training dataset is shuffled after each Epoch in order to ensure the model does not try to memorize the order of data
When the model takes in labeled data points from the validation dataset - for the validation process which occurs through the training stage - or new and unlabeled data points (without a target value) to predict on - as the model is ran during the inference stage aftering being trained, the model will take the same process it takes in developing its predicted value for each data point, however, its parameters will not be adjusted

Z-Score Normalization

Input feature or target (or label) values in each data point in the training and validation datasets from the data received by the model during the training stage is normalized (or scaled) to a z-score based on the mean and standard deviation of the values of the respective feature or target in the training dataset
This normalization serves to ensure that potential differences in the magnitude of range between input features and target values in the data the model receives do not destabilize the model’s training or learning
Keeping the values of the input features and targets in the data points of the validation dataset normalized with respect to the statistics of the training dataset ensures that the model does not have undue information about unseen (validation) data while it is training, which may compromise the its ability to evaluate its performance during validation
As such, the model is trained on predicting z-scores of target values based on z-scores of input feature values
Similarly, input feature values in new data points received by the model during the inference stage is normalized (or scaled) based on the mean and standard deviation of the values of the respective feature in the training dataset
This is given the fact the model is trained on predicting the z-scores of target values based on the z score of input feature values with respect to the statistics of the training data rather than the raw values themselves
The output of the model during the inference stage is rescaled in order to give the user a final output in the correct scaling

Weights

Each neuron will have a number of input weights corresponding to the number of inputs it receives (such that each input received by the neuron can be weighted by its respective input weight) in addition to a bias and a bias weight
Kaiming He Weight Initialization - Weights in each neuron are initialized using the Kaiming He initialization method based on the number of input/neuron connections received by the respective layer of the neuron (except bias, which is initialized to 0)
Each weight in a given neuron is randomly and uniformly initialized to a value within the following range [-6inputs received by neuron , 6inputs received by neuron  ]
This initialization method addresses the potential issue of Vanishing Gradients

Forward Pass

For each data point received by the model, neurons in the the input layer (initial layer) will in input feature values from the data point, and each neuron in each consecutive layer in the model will take in inputs consisting of each of the neuron outputs from the previous layer
The inputs of each neuron in the input layer represent the input features in the data point received by the model
The inputs of each neuron in the layers after the input layer represent the final neuron outputs from the previous layer
In each layer, each neuron will calculate a linear transformed weighted sum consisting of the sum of each input taken by the neuron multiplied by the neuron’s respective input weight for that input, plus the neuron’s bias multiplied by its bias weight

Layer Normalization - In each of the hidden layers (all layers excluding the input (initial) and output (final) layer) of the model, the linear transformed (or combined) weighted sum of each neuron is normalized (or scaled) to a z-score based on the mean and standard deviation of the weighted sums of all the neurons in the layer
This serves to strengthen the model’s training during the training stage and lead to smoother convergence (towards minimized cost - or error) for the model
The purpose of not applying Layer Normalization or a non-linear transformation on the output layer is to ensure output values of the model do not become distorted
The layer normalized weighted sum of each neuron in the input and output layer receive not non-linear transformation, effectively being processed by the Linear activation function

Activation Function - In each of the hidden layers of the model, non-linear transformation is applied to the layer normalized weighted sum of each neuron via the Rectified Linear Unit (ReLU) activation function, resulting in a non-linearly transformed final output by each neuron
The purpose of not applying Layer Normalization or a non-linear transformation on the input layer is to maintain consistent treatment of data, given the input data neurons in the input layer receive are the input feature values of the data point passed in to the model, while the input data neurons in the layers after receive are outputs from neurons from the previous layer
For neurons in the hidden layers, the final (ReLU) activation output is as follows:

Final Output = Weighted Sum for Weighted Sum > 0
Final Output = 0 for Weight Sum <= 0

For neurons in the input and output layer, the final (Linear) activation output is as follows:
Final Output = Weighted Sum 
The final activation outputs from each neuron in a given layer is passed on as the inputs to the neurons in the next layer until that output layer (which only contains one neuron), where the final activation output of the layer’s neuron will be the predicted value of the model of a given data point processed by the model

Backward Propagation
After the model computes a predicted value based on the input feature values of a data point from the training dataset, it will compute the gradients for each weight within each neuron within each layer of the model based on the cost function -

L2 Regularization - The loss function for the model for each labeled data point from the training dataset is equal to the difference between the target value of the data point and the model’s predicted value for the data point plus the model’s L2 Regularization - which consists of the sum of the squared values of all the input weights in the model

loss = target - prediction + summation(input_weight^2)

The purpose of adding the model’s L2 Regularization to its loss function is to stabilize the values of input weights with respect to their input, which serves to stabilize training and lead to smoother convergence, accounting for the Exploding Gradients problem

Huber Loss (Cost) Function - The cost function for the model based on its loss function is derived from the Huber Loss (Cost) Function
The gradient for each weight within each neuron (going from the output layer to input layer) is computed using chain rule based on the cost function of the model derived from the Huber Loss Function
The cost function for the model after processing a given data point is as follows:

cost = 0.5(loss)^2 for loss <= delta

cost = delta (|loss| - 0.5delta) for otherwise

Gradient for input weight =  cost derivative with respect to loss * loss derivative with respect to neuron activation output (final neuron output) * neuron activation output derivative with respect to neuron weighted sum * neuron weighted sum derivative with respect to input weight (equals the respective input of the input weight)

Gradient for bias weight = cost derivative with respect to loss * loss derivative with respect to neuron activation output (final neuron output) * neuron activation output derivative with respect to neuron weighted sum * neuron weighted sum derivative with respect to bias weight (equals bias)

General Formula For All Weights:

dCdW=dCdL*dLdA*dAdS*dSdW

Gradient for bias = cost derivative with respect to loss * loss derivative with respect to neuron activation output (final neuron output) * neuron activation output derivative with respect to neuron weighted sum * neuron weighted sum derivative with respect to bias (equals 1)

dCdB=dCdL*dLdA*dAdS*dSdB

Gradient Descent
The model utilizes Stochastic Gradient Descent (SGD), updating its weights and biases as it processes each data point from the training dataset
This provides for a more dynamic updating mechanism, better modeling the biological structure of neuron processing
SGD functionality for this model was fixed and adjusted from previous neural network developments in order to ensure that weights are properly and accurately updated.
After the model computes a gradient for a given weight or bias (parameter) in the model, the model will take several steps to adjust the parameter using the computed gradient for it

Gradient Clipping - The gradients computed for all weights and bias in a given neuron will be regulated (or clipped) through the gradient clipping process
The L2 Norm (Euclidean Norm) of all the gradients in the neuron are computed
The Max Norm of the gradients is determined as a percentile of the gradients based on the Max Norm Percentile Threshold hyperparameter 
If the L2 Norm of the gradients in the neuron is greater than their Max Norm, then all of the gradients in the neuron will be scaled down by a factor of (Max Norm/L2 Norm) such that none of the gradients will be greater than the Max Norm - otherwise they will remain the same

Final Gradient = Gradient (Max NormL2 Norm) for L2 Norm > Max Norm

Final Gradient = Gradient for otherwise

This process serves to stabilize training and lead to smoother convergence, accounting for the Exploding Gradients problem

Learning Rate Decay - The learning rate of the model by which gradients are scaled before being used to adjust weights will decay through each iteration of updates to parameters (or each time the model processes a data point from the training dataset) using the Learning Rate Decay Rate hyperparameter provided by the user
The adjusted learning rate used by the model at a given iteration of updates towards weights and biases is as follows
Adjusted Learning Rate = Original Learning Rate1 + (Learning Rate Decay Rate)(Iteration Step)
This process helps to provide for faster convergence as the model keeps learning from the training data

Momentum-Based Parameter Updating - After the gradient and adjusted learning rate for a given parameter is computed, the update (or velocity) to the parameter will be regulated based on previous updates given to that parameter
The factor by which the parameter update will be regulated is based on the Momentum Factor hyperparameter provided by the user
This process is an additional mechanism to stabilize training and lead to smoother convergence, accounting for the Exploding Gradients problem
The final update for a given parameter is as follows

Final Velocity (Update) = (Previous Velocity * MomentumFactor) - (Gradient * Adjusted Learning Rate)

The computed final update for the given parameter is added to the parameter as follows:

Parameter Value = Previous Parameter Value + Final Velocity 

Validation 
For each epoch in the training stage, after the model has processed and updated its parameters for all the data points in the training data, the model will test on the validation dataset with the updated parameters the model developed as it processed each data point in the training dataset for that epoch
The model will process all of the labeled data points in the validation dataset (determining a predicted value as it did, determine the cost for each data point processed, and then determine the average cost across all data points in the validation dataset - a metric for how well the model can generalize towards unseen data

Best Weight Restoration - Once the model has completed all epochs of training, the model will determine the epoch in which it produced the lowest average cost on the validation dataset, and will restore its parameters to the version it had at that epoch 
By restoring parameters to the epoch version which worked which fitted best to the validation data the model did not have access to when updating parameters through the course of processing the training data, the model is able to reduce overfitting to training data and better be able to generalize to new data, critical to the model’s performance

After Training
Inference Stage - After the training stage, once the model has trained on the training data for the full number of training epochs and executed Best Weight Restoration through Validation, the trained model will be able to run and process new unlabeled data points, developing prediction values based on the input feature values of a given data point, without having to anymore update its parameters
As such Backpropagation, Gradient Descent, and Validation will no longer be executed during this stage, with the other functionality necessary for the model’s prediction mechanism remaining

SPECIAL CONSIDERATIONS

Importance of Consistency In Data
It is highly important that each of the data points provided to the model for the training stage have the same number of input feature value in the same feature order, with target values at the end
Each data point provided to the model for prediction during the inference stage much have the same number of input feature values in the same feature order as each of the data points provided to the model for the training stage

Potential Model Limitations
Smaller training datasets can limit the ability for the model to learn the relationships and patterns between input feature and target values, possibly limiting the model from optimal performance
Large outliers or differences in scales in new data points the model is run on during the Inference Stage which was not present in data points in the training data and which the model did not subsequently train on could undermine the model’s predictive performance

APPLICATIONS AND NEXT STEPS

Having developed this model, I now plan to study how to utilize it for robust quantitative analysis, specifically in areas such as financial and economic modeling, which are highly relevant and meaningful application areas. Specifically, I aim to study how to fine-tune hyperparameters in achieving optimal predictive performance as well as how to determine and analyze the weighting of different input features in target values. As a sample application, I plan to employ the model in medical insurance cost data. This process will entail two primary steps. First, fine-tuning the model’s hyperparameters and potentially functionality to see how the model can optimally train, generalize, and predict on medical insurance cost data. Second, to conduct an ablation analysis to examine the weighting of different kinds of features (such as demographics, geography, health/smoker status metrics, etc.) on medical insurance costs. I plan to to extend what I learn in this particular application towards other areas of analysis, particularly in the Digital Economy with areas such as the Content Economy and Social Commerce.
