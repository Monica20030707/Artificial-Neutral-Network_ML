# Artificial_Neutral_Network-ML
Our goal is to construct a Artificial Neural Network (ANN) classifier using the MNIST handwritten digits dataset. The task involves classifying these hand-written images into their respective digit categories (0-9). Notably, we will implement the BACKPROPAGATION algorithm from scratch, as outlined in the book by Kelleher et al. Importantly, we are explicitly prohibited from using ready-made open-source or commercial Machine Learning packages such as TensorFlow or PyTorch.

## Data
Enclosed in the "data" subfolder are four files:

- **training60000.csv**: Contains 60,000 training images.
- **training60000_labels.csv**: Labels corresponding to the training images.
- **testing10000.csv**: Contains 10,000 testing images.
- **testing10000_labels.csv**: Labels corresponding to the testing images.
  
The data has been normalized using Range Normalization with high=1 and low=0.01.

## Implementations
Our ANN will consist of three layers:
- Input Layer: Comprising 784 input neurons (one for each pixel in the 28x28 image).
- Hidden Layer(s): We’ll utilized with one hidden layer (X neurons).
- Output Layer: Consisting of 10 softmax units, corresponding to the 10 possible digit classes.
  
Activation functions:
- Use the logistic function for the HIDDEN layer(s).
- Use the softmax function for the OUTER layer.
  
Error Function and δ Calculation:

We’ll use the cross-entropy error function with a modification in the cross-entropy delta for the output layer to prevent errors due to log(0).
For the output layer, the δ values are calculated as follows:
- δ = -(1 – Pk) for the output neuron corresponding to the one-hot encoded target label.
- δ = Pk for all other output neurons.
  
## Training Details
The network is trained with the following parameters:
- Epochs: The network is trained for 100 epochs.
- Learning Rate: The learning rate used for training is 0.05.
- Mini-batch Size: The size of the mini-batch used for training is 300.
- Weight Initialization: The weights of the network are initialized using He Initialization to prevent vanishing/exploding gradient problems.

## Result
After training, test your model with the test datasets (testing10000.csv and testing10000_labels.csv). Print out the network properties and the classification accuracy. My code achieved 91.95% accuracy using the specified training parameters.

