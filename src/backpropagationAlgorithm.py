#!/usr/bin/env python
# coding: utf-8

# # Build ANN classifier using backpropagation algorithm

# In[235]:


# import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# training60000.csv: 60,000 images of hand-written digits
# training60000_labels.csv: 60,000 labels (for the images in training60000.csv)
# testing10000.csv: 10,000 images of hand-written digits
# testing10000_labels.csv: 10,000 labels (for the images in testing10000.csv)

# training60000.csv and training60000_labels.csv for building/training your neural network model.

# testing10000.csv and testing10000_labels.csv for testing your neural network model.

# Load the training data and their corresponding labels into DataFrames
mnist_data= pd.read_csv("data/training60000.csv")
mnist_labels= pd.read_csv("data/training60000_labels.csv")


# Print their sizes
print("Training sizes:\n")
print(mnist_data.shape)
print(mnist_labels.shape,"\n")

# Load the testing data and their corresponding labels into DataFrames
mnist_testing_data= pd.read_csv("data/testing10000.csv")
mnist_testing_labels= pd.read_csv("data/testing10000_labels.csv")

# Print their sizes
print("Testing sizes: \n")
print(mnist_testing_data.shape)
print(mnist_testing_labels.shape,"\n")


# In[236]:


import numpy as np

# For the stopping criteria, I decide to iterate 60 times (as 60000:6000= 60)
epoch= 100

# The learning rate affects how aggressive we go down the hill.
# That is, do we move slowly or make big jumps.
learning_rate= 0.05

# Create all the function that will need to be used for the algorithm

# Logistic function for the hidden layers
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Softmax function for the outer layer
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define cross-entropy function (error)
# For output layer:
def cross_entropy_output(predictions, true_labels):
    num_instances = true_labels.shape[0]
    
    # Clip predictions to avoid log(0) issues
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    
    # Compute the errors for the output layer
    delta = predictions.copy()
    delta[range(num_instances), true_labels] -= 1.0
    
    return delta / num_instances

# For hidden layer:
def cross_entropy_hidden(hidden_output, weights_output, outer_error):
    
    # Compute the errors for the hidden layer
    delta = np.dot(outer_error, weights_output.T) * (hidden_output * (1 - hidden_output))

    return delta


# In[249]:


# Forward pass
def forward_pass(z, weights_hidden, bias_hidden, weights_output, bias_output):
    # Compute hidden layer input and output
    hidden_input = np.dot(z, weights_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    # Compute outer layer input and output
    outer_input = np.dot(hidden_output, weights_output) + bias_output
    outer_output = softmax(outer_input)
    
    return hidden_input, hidden_output, outer_input, outer_output

# Backward pass
def backward_pass(z, true_labels, hidden_input, hidden_output, outer_input, outer_output,
                  weights_hidden, bias_hidden, weights_output, bias_output, learning_rate):
    num_instances = true_labels.shape[0]

    # Compute errors for the output layer
    outer_error = cross_entropy_output(outer_output, true_labels)

    # Update outer layer parameters
    weights_output -= learning_rate * np.dot(hidden_output.T, outer_error)
    bias_output -= learning_rate * np.sum(outer_error, axis=0, keepdims=True)

    # Compute errors for the hidden layer
    hidden_error = cross_entropy_hidden(hidden_output, weights_output, outer_error)

    # Update hidden layer parameters
    weights_hidden -= learning_rate * np.dot(z.T, hidden_error)
    bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

# Training loop
def train(z, training_labels, epochs, learning_rate, batch_size=300):
    num_instances, input_size = z.shape
    num_classes = len(np.unique(y))
    
    # Initialize weights (He) and biases  with random values
    
    weights_hidden = np.random.randn(input_size, num_classes) * np.sqrt(2 / input_size)
    bias_hidden = np.zeros((1, num_classes))
    
    weights_output = np.random.randn(num_classes, num_classes) * np.sqrt(2 / num_classes)
    bias_output = np.zeros((1, num_classes))
    
    # Calculate the number of batches
    num_batches = num_instances // batch_size
    
    # Change epoch should start from 1, not 0
    for epoch in range(1, epochs + 1): 
        for batch in range(num_batches):
            # Extract a batch of data
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            z_batch = z[batch_start:batch_end]
            tlabels_batch = training_labels[batch_start:batch_end]
            
            # Forward pass
            hidden_input, hidden_output, outer_input, outer_output = forward_pass(z_batch, weights_hidden, bias_hidden, weights_output, bias_output)
            
            # Compute errors for the output layer
            outer_error = cross_entropy_output(outer_output, tlabels_batch)
            
            # Total loss of the batch
            loss = np.sum(outer_error)  
            
            # Backward pass
            backward_pass(z_batch, tlabels_batch, hidden_input, hidden_output, outer_input, outer_output,
                          weights_hidden, bias_hidden, weights_output, bias_output, learning_rate)
            
        
        # Print total loss every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: delta = {loss}")


    return weights_hidden, bias_hidden, weights_output, bias_output

# Access the labels column for training
training_labels = mnist_labels.iloc[:, 0].values

# Train the model
trained_data = train(mnist_data, training_labels, epoch, learning_rate)


# In[250]:


# Testing function
def test(z, weights_hidden, bias_hidden, weights_output, bias_output):
    # Forward pass on the testing data
    hidden_input, hidden_output, outer_input, predictions = forward_pass(z, weights_hidden, bias_hidden, weights_output, bias_output)
    
    # Apply softmax to obtain probabilities
    predictions = softmax(predictions)
    
    # Predicted class is the one with the highest probability
    predicted_labels = np.argmax(predictions, axis=1)
    
    return predicted_labels

# Use the trained parameters to make predictions on the testing data
predicted_labels = test(mnist_testing_data, trained_data[0], trained_data[1], trained_data[2], trained_data[3])


# In[251]:


def calculate_network_properties(predicted_labels, true_labels, weights_hidden, weights_output):
    # Number of neurons in the input and output layers
    input_neurons = weights_hidden.shape[0]
    output_neurons = weights_output.shape[1]

    # Number of correct and incorrect classifications
    correct = np.sum(predicted_labels == true_labels)
    incorrect = len(true_labels) - correct

    # Accuracy
    accuracy = correct / len(true_labels) * 100.0

    # Print network properties
    print("===== Result:")
    print(f"Network properties: Input: {input_neurons}, Hidden: X, Output: {output_neurons}")
    print(f"Correct classifications: {correct}")
    print(f"Number of incorrect classifications: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return input_neurons, output_neurons, correct, incorrect, accuracy

# Access the labels column
true_labels = mnist_testing_labels.iloc[:, 0].values

# Calculate network properties
input_neurons, output_neurons, correct, incorrect, accuracy = calculate_network_properties(predicted_labels, true_labels, trained_data[0], trained_data[2])

