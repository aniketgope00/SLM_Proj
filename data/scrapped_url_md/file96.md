- [Data Science](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Data Science Projects](https://www.geeksforgeeks.org/top-data-science-projects/)
- [Data Analysis](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Data Visualization](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-projects/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Computer Vision](https://www.geeksforgeeks.org/computer-vision/)
- [Artificial Intelligence](https://www.geeksforgeeks.org/artificial-intelligence/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/?type%3Darticle%26id%3D703096&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Deep Neural net with forward and back propagation from scratch - Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)

# Multi-Layer Perceptron Learning in Tensorflow

Last Updated : 05 Feb, 2025

Comments

Improve

Suggest changes

23 Likes

Like

Report

**Multi-Layer Perceptron (MLP)** is an artificial neural network widely used for solving classification and regression tasks.

MLP consists of fully connected dense layers that transform input data from one dimension to another. It is called _“multi-layer”_ because it contains an input layer, one or more hidden layers, and an output layer. The purpose of an MLP is to model complex relationships between inputs and outputs, making it a powerful tool for various machine learning tasks.

**Pre-requisites:** [_Neural Network_](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) _,_ [_Artificial Neural Network_](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/) _,_ [_Perceptron_](https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/)

## **Key Components of Multi-Layer Perceptron (MLP)**

- **Input Layer**: Each neuron (or node) in this layer corresponds to an input feature. For instance, if you have three input features, the input layer will have three neurons.
- **Hidden Layers**: An MLP can have any number of hidden layers, with each layer containing any number of nodes. These layers process the information received from the input layer.
- **Output Layer**: The output layer generates the final prediction or result. If there are multiple outputs, the output layer will have a corresponding number of neurons.

![](https://media.geeksforgeeks.org/wp-content/uploads/nodeNeural.jpg)

Every connection in the diagram is a representation of the fully connected nature of an MLP. This means that every node in one layer connects to every node in the next layer. As the data moves through the network, each layer transforms it until the final output is generated in the output layer.

## Working of Multi-Layer Perceptron

Let’s delve in to the working of the multi-layer perceptron. The key mechanisms such as forward propagation, loss function, backpropagation, and optimization.

### Step 1: Forward Propagation

In **forward propagation**, the data flows from the input layer to the output layer, passing through any hidden layers. Each neuron in the hidden layers processes the input as follows:

1. **Weighted Sum**: The neuron computes the weighted sum of the inputs:
   - z=∑iwixi+bz = \\sum\_{i} w\_i x\_i + bz=∑i​wi​xi​+b
     - Where:
       - xix\_ixi​​ is the input feature.
       - wiw\_iwi​​ is the corresponding weight.
       - bbb is the bias term.
2. [**Activation Function**](https://www.geeksforgeeks.org/activation-functions-neural-networks/): The weighted sum z is passed through an activation function to introduce non-linearity. Common activation functions include:
   - **Sigmoid**: σ(z)=11+e−z\\sigma(z) = \\frac{1}{1 + e^{-z}}σ(z)=1+e−z1​
   - **ReLU (Rectified Linear Unit)**: f(z)=max⁡(0,z)f(z) = \\max(0, z)f(z)=max(0,z)
   - **Tanh (Hyperbolic Tangent)**: tanh⁡(z)=21+e−2z–1\\tanh(z) = \\frac{2}{1 + e^{-2z}} – 1tanh(z)=1+e−2z2​–1

### Step 2: Loss Function

Once the network generates an output, the next step is to calculate the loss using a [loss function](https://www.geeksforgeeks.org/loss-functions-in-deep-learning/). In supervised learning, this compares the predicted output to the actual label.

For a classification problem, the commonly used [**binary cross-entropy**](https://www.geeksforgeeks.org/binary-cross-entropy-log-loss-for-binary-classification/) loss function is:

L=−1N∑i=1N\[yilog⁡(y^i)+(1–yi)log⁡(1–y^i)\]L = -\\frac{1}{N} \\sum\_{i=1}^{N} \\left\[ y\_i \\log(\\hat{y}\_i) + (1 – y\_i) \\log(1 – \\hat{y}\_i) \\right\]L=−N1​∑i=1N​\[yi​log(y^​i​)+(1–yi​)log(1–y^​i​)\]

Where:

- yiy\_iyi​ is the actual label.
- y^i\\hat{y}\_iy^​i​ is the predicted label.
- NNN is the number of samples.

For regression problems, the [**mean squared error (MSE)**](https://www.geeksforgeeks.org/python-mean-squared-error/) is often used:

MSE=1N∑i=1N(yi–y^i)2MSE = \\frac{1}{N} \\sum\_{i=1}^{N} (y\_i – \\hat{y}\_i)^2MSE=N1​∑i=1N​(yi​–y^​i​)2

### Step 3: Backpropagation

The goal of training an MLP is to minimize the loss function by adjusting the network’s weights and biases. This is achieved through [**backpropagation**](https://www.geeksforgeeks.org/backpropagation-in-neural-network/):

1. **Gradient Calculation**: The gradients of the loss function with respect to each weight and bias are calculated using the chain rule of calculus.
2. **Error Propagation**: The error is propagated back through the network, layer by layer.
3. [**Gradient Descent**](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/): The network updates the weights and biases by moving in the opposite direction of the gradient to reduce the loss: w=w–η⋅∂L∂ww = w – \\eta \\cdot \\frac{\\partial L}{\\partial w}w=w–η⋅∂w∂L​
   - Where:
     - www is the weight.
     - η\\etaη is the learning rate.
     - ∂L∂w​\\frac{\\partial L}{\\partial w}​∂w∂L​​ is the gradient of the loss function with respect to the weight.

### Step 4: Optimization

MLPs rely on optimization algorithms to iteratively refine the weights and biases during training. Popular optimization methods include:

- [**Stochastic Gradient Descent (SGD)**](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/): Updates the weights based on a single sample or a small batch of data: w=w–η⋅∂L∂ww = w – \\eta \\cdot \\frac{\\partial L}{\\partial w}w=w–η⋅∂w∂L​

- [**Adam Optimizer**](https://www.geeksforgeeks.org/adam-optimizer/): An extension of SGD that incorporates momentum and adaptive learning rates for more efficient training:
  - mt=β1mt−1+(1–β1)⋅gtm\_t = \\beta\_1 m\_{t-1} + (1 – \\beta\_1) \\cdot g\_tmt​=β1​mt−1​+(1–β1​)⋅gt​
  - vt=β2vt−1+(1–β2)⋅gt2v\_t = \\beta\_2 v\_{t-1} + (1 – \\beta\_2) \\cdot g\_t^2
    vt​=β2​vt−1​+(1–β2​)⋅gt2​
- Here, gtg\_tgt​​ represents the gradient at time ttt, and β1,β2\\beta\_1, \\beta\_2β1​,β2​ are decay rates.

Now that we are done with the theory part of multi-layer perception, let’s go ahead and implement some code in python using the TensorFlow library.

## Implementing Multi Layer Perceptron using TensorFlow

In this section, we will guide through building a neural network using TensorFlow.

### Step 1: Import Required Modules and Load Dataset

First, we import necessary libraries such as **TensorFlow**, **NumPy**, and **Matplotlib** for visualizing the data. We also load the [**MNIST dataset**](https://www.geeksforgeeks.org/mnist-dataset/).

Python`
# Importing necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
`

### **Step 2:** Load and Normalize Image Data

Next, we normalize the image data by dividing by **255** (since pixel values range from 0 to 255), which helps in faster convergence during training.

Python`
# Normalize image pixel values by dividing by 255 (grayscale)
gray_scale = 255
x_train = x_train.astype('float32') / gray_scale
x_test = x_test.astype('float32') / gray_scale
# Checking the shape of feature and target matrices
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)
`

**Output:**

```
Feature matrix (x_train): (60000, 28, 28)
Target matrix (x_test): (10000, 28, 28)
Feature matrix (y_train): (60000,)
Target matrix (y_test): (10000,)
```

### **Step** 3: Visualizing Data

To understand the data better, we plot the first 100 training samples, each representing a digit.

Python`
# Visualizing 100 images from the training data
fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto')
        k += 1
plt.show()
`

**Output:**

![Sample-Images](https://media.geeksforgeeks.org/wp-content/uploads/20250205023853302691/Sample-Images.png)

### Step 4: Building the Neural Network Model

Here, we build a **Sequential neural network model**. The model consists of:

- **Flatten Layer**: Reshapes 2D input (28×28 pixels) into a 1D array of 784 elements.
- **Dense Layers**: Fully connected layers with 256 and 128 neurons, both using the relu activation function.
- **Output Layer**: The final layer with 10 neurons representing the 10 classes of digits (0-9) with **sigmoid** activation.

Python`
# Building the Sequential neural network model
model = Sequential([\
    Flatten(input_shape=(28, 28)),\
    Dense(256, activation='sigmoid'),\
    Dense(128, activation='sigmoid'),\
    Dense(10, activation='softmax'),\
])
`

### Step 5: Compiling the Model

Once the model is defined, we compile it by specifying:

- **Optimizer**: Adam, for efficient weight updates.
- **Loss Function**: Sparse categorical crossentropy, which is suitable for multi-class classification.
- **Metrics**: Accuracy, to evaluate model performance.

Python`
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
`

### Step 6: Training the Model

We train the model on the training data using 10 epochs and a batch size of 2000. We also use 20% of the training data for validation to monitor the model’s performance on unseen data during training.

Python`
model.fit(x_train, y_train, epochs=10,
          batch_size=2000,
          validation_split=0.2)
`

**Output:**

```
Epoch 1/10
24/24 ━━━━━━━━━━━━━━━━━━━━ 5s 141ms/step - accuracy: 0.2382 - loss: 2.2627 - val_accuracy: 0.6237 - val_loss: 1.7715
Epoch 2/10
24/24 ━━━━━━━━━━━━━━━━━━━━ 3s 55ms/step - accuracy: 0.6948 - loss: 1.6032 - val_accuracy: 0.8134 - val_loss: 1.0656
. . .
Epoch 9/10
24/24 ━━━━━━━━━━━━━━━━━━━━ 2s 54ms/step - accuracy: 0.9194 - loss: 0.3018 - val_accuracy: 0.9250 - val_loss: 0.2774
Epoch 10/10
24/24 ━━━━━━━━━━━━━━━━━━━━ 1s 57ms/step - accuracy: 0.9234 - loss: 0.2805 - val_accuracy: 0.9285 - val_loss: 0.2623
```

### Step 7: Evaluating the Model

After training, we evaluate the model on the test dataset to determine its performance.

Python`
# Evaluating the model on test data
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)
`

**Output:**

```
Test loss, Test accuracy: [0.2682029604911804, 0.9257000088691711]
```

We got the accuracy of our model 92% by using **model.evaluate()** on the test samples.

### Complete Code

Python`
## Importing necessary modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize image pixel values by dividing by 255 (grayscale)
gray_scale = 255
x_train = x_train.astype('float32') / gray_scale
x_test = x_test.astype('float32') / gray_scale
# Checking the shape of feature and target matrices
print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (y_train):", y_train.shape)
print("Feature matrix (x_test):", x_test.shape)
print("Target matrix (y_test):", y_test.shape)
# Visualizing 100 images from the training data
fig, ax = plt.subplots(10, 10)
k = 0
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(x_train[k].reshape(28, 28), aspect='auto', cmap='gray')
        ax[i][j].axis('off')  # Hide axes for better visualization
        k += 1
plt.suptitle("Sample Images from MNIST Dataset", fontsize=16)
plt.show()
# Building the Sequential neural network model
model = Sequential([\
    # Flatten input from 28x28 images to 784 (28*28) vector\
    Flatten(input_shape=(28, 28)),\
\
    # Dense layer 1 (256 neurons)\
    Dense(256, activation='sigmoid'),\
\
    # Dense layer 2 (128 neurons)\
    Dense(128, activation='sigmoid'),\
\
    # Output layer (10 classes)\
    Dense(10, activation='softmax'),\
])
# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Training the model with training data
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=2000,
                    validation_split=0.2)
# Evaluating the model on test data
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, Test accuracy:', results)
# Visualization of Training and Validation Accuracy/Loss
plt.figure(figsize=(12, 5))
# Plotting Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True)
# Plotting Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.suptitle("Model Training Performance", fontsize=16)
plt.tight_layout()
plt.show()
`

**Output:**

![download](https://media.geeksforgeeks.org/wp-content/uploads/20250205024417151121/download.png)

The model is learning effectively on the training set, but the validation accuracy and loss levels off, which might indicate that the model is starting to overfit (where it performs well on training data but not as well on unseen data).

## Advantages of Multi Layer Perceptron

- **Versatility**: MLPs can be applied to a variety of problems, both classification and regression.
- **Non-linearity**: Thanks to activation functions, MLPs can model complex, non-linear relationships in data.
- **Parallel Computation**: With the help of GPUs, MLPs can be trained quickly by taking advantage of parallel computing.

## Disadvantages of Multi Layer Perceptron

- **Computationally Expensive**: MLPs can be slow to train, especially on large datasets with many layers.
- **Prone to Overfitting**: Without proper regularization techniques, MLPs can overfit the training data, leading to poor generalization.
- **Sensitivity to Data Scaling**: MLPs require properly normalized or scaled data for optimal performance.

The **Multilayer Perceptron** has the ability to learn complex patterns from data makes it a valuable tool in machine learning. Whether you’re working with structured data, images, or text, understanding how MLP works can open doors to solving a wide range of problems.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)

[Deep Neural net with forward and back propagation from scratch - Python](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)

[S](https://www.geeksforgeeks.org/user/swarnimrai/)

[swarnimrai](https://www.geeksforgeeks.org/user/swarnimrai/)

Follow

23

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [Tensorflow](https://www.geeksforgeeks.org/tag/tensorflow/)

### Similar Reads

[Deep Learning Tutorial\\
\\
\\
Deep Learning tutorial covers the basics and more advanced topics, making it perfect for beginners and those with experience. Whether you're just starting or looking to expand your knowledge, this guide makes it easy to learn about the different technologies of Deep Learning. Deep Learning is a bran\\
\\
5 min read](https://www.geeksforgeeks.org/deep-learning-tutorial/)

## Introduction to Deep Learning

- [Introduction to Deep Learning\\
\\
\\
Deep Learning is transforming the way machines understand, learn, and interact with complex data. Deep learning mimics neural networks of the human brain, it enables computers to autonomously uncover patterns and make informed decisions from vast amounts of unstructured data. Deep Learning leverages\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-deep-learning/)

* * *

- [Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning\\
\\
\\
Artificial Intelligence is basically the mechanism to incorporate human intelligence into machines through a set of rules(algorithm). AI is a combination of two words: "Artificial" meaning something made by humans or non-natural things and "Intelligence" meaning the ability to understand or think ac\\
\\
14 min read](https://www.geeksforgeeks.org/difference-between-artificial-intelligence-vs-machine-learning-vs-deep-learning/)

* * *


## Basic Neural Network

- [Difference between ANN and BNN\\
\\
\\
Do you ever think of what it's like to build anything like a brain, how these things work, or what they do? Let us look at how nodes communicate with neurons and what are some differences between artificial and biological neural networks. 1. Artificial Neural Network: Artificial Neural Network (ANN)\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-ann-and-bnn/)

* * *

- [Single Layer Perceptron in TensorFlow\\
\\
\\
Single Layer Perceptron is inspired by biological neurons and their ability to process information. To understand the SLP we first need to break down the workings of a single artificial neuron which is the fundamental building block of neural networks. An artificial neuron is a simplified computatio\\
\\
4 min read](https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/)

* * *

- [Multi-Layer Perceptron Learning in Tensorflow\\
\\
\\
Multi-Layer Perceptron (MLP) is an artificial neural network widely used for solving classification and regression tasks. MLP consists of fully connected dense layers that transform input data from one dimension to another. It is called "multi-layer" because it contains an input layer, one or more h\\
\\
9 min read](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)

* * *

- [Deep Neural net with forward and back propagation from scratch - Python\\
\\
\\
This article aims to implement a deep neural network from scratch. We will implement a deep neural network containing two input layers, a hidden layer with four units and one output layer. The implementation will go from scratch and the following steps will be implemented. Algorithm:1. Loading and v\\
\\
6 min read](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)

* * *

- [Understanding Multi-Layer Feed Forward Networks\\
\\
\\
Let's understand how errors are calculated and weights are updated in backpropagation networks(BPNs). Consider the following network in the below figure. The network in the above figure is a simple multi-layer feed-forward network or backpropagation network. It contains three layers, the input layer\\
\\
7 min read](https://www.geeksforgeeks.org/understanding-multi-layer-feed-forward-networks/)

* * *

- [List of Deep Learning Layers\\
\\
\\
Deep learning (DL) is characterized by the use of neural networks with multiple layers to model and solve complex problems. Each layer in the neural network plays a unique role in the process of converting input data into meaningful and insightful outputs. The article explores the layers that are us\\
\\
7 min read](https://www.geeksforgeeks.org/ml-list-of-deep-learning-layers/)

* * *


## Activation Functions

- [Activation Functions\\
\\
\\
To put it in simple terms, an artificial neuron calculates the 'weighted sum' of its inputs and adds a bias, as shown in the figure below by the net input. Mathematically, Net Input=∑(Weight×Input)+Bias\\text{Net Input} =\\sum \\text{(Weight} \\times \\text{Input)+Bias}Net Input=∑(Weight×Input)+Bias Now the value of net input can be any anything from -\\
\\
3 min read](https://www.geeksforgeeks.org/activation-functions/)

* * *

- [Types Of Activation Function in ANN\\
\\
\\
The biological neural network has been modeled in the form of Artificial Neural Networks with artificial neurons simulating the function of a biological neuron. The artificial neuron is depicted in the below picture: Each neuron consists of three major components:Â  A set of 'i' synapses having weigh\\
\\
4 min read](https://www.geeksforgeeks.org/types-of-activation-function-in-ann/)

* * *

- [Activation Functions in Pytorch\\
\\
\\
In this article, we will Understand PyTorch Activation Functions. What is an activation function and why to use them?Activation functions are the building blocks of Pytorch. Before coming to types of activation function, let us first understand the working of neurons in the human brain. In the Artif\\
\\
5 min read](https://www.geeksforgeeks.org/activation-functions-in-pytorch/)

* * *

- [Understanding Activation Functions in Depth\\
\\
\\
In artificial neural networks, the activation function of a neuron determines its output for a given input. This output serves as the input for subsequent neurons in the network, continuing the process until the network solves the original problem. Consider a binary classification problem, where the\\
\\
6 min read](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

* * *


## Artificial Neural Network

- [Artificial Neural Networks and its Applications\\
\\
\\
As you read this article, which organ in your body is thinking about it? It's the brain, of course! But do you know how the brain works? Well, it has neurons or nerve cells that are the primary units of both the brain and the nervous system. These neurons receive sensory input from the outside world\\
\\
9 min read](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)

* * *

- [Gradient Descent Optimization in Tensorflow\\
\\
\\
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function. In other words, gradient descent is an iterative algorithm that helps to find the optimal solution to a given problem. In this blog, we will discuss gr\\
\\
15+ min read](https://www.geeksforgeeks.org/gradient-descent-optimization-in-tensorflow/)

* * *

- [Choose Optimal Number of Epochs to Train a Neural Network in Keras\\
\\
\\
One of the critical issues while training a neural network on the sample data is Overfitting. When the number of epochs used to train a neural network model is more than necessary, the training model learns patterns that are specific to sample data to a great extent. This makes the model incapable t\\
\\
6 min read](https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/)

* * *


## Classification

- [Python \| Classify Handwritten Digits with Tensorflow\\
\\
\\
Classifying handwritten digits is the basic problem of the machine learning and can be solved in many ways here we will implement them by using TensorFlowUsing a Linear Classifier Algorithm with tf.contrib.learn linear classifier achieves the classification of handwritten digits by making a choice b\\
\\
4 min read](https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/)

* * *

- [Train a Deep Learning Model With Pytorch\\
\\
\\
Neural Network is a type of machine learning model inspired by the structure and function of human brain. It consists of layers of interconnected nodes called neurons which process and transmit information. Neural networks are particularly well-suited for tasks such as image and speech recognition,\\
\\
6 min read](https://www.geeksforgeeks.org/train-a-deep-learning-model-with-pytorch/)

* * *


## Regression

- [Linear Regression using PyTorch\\
\\
\\
Linear Regression is a very commonly used statistical method that allows us to determine and study the relationship between two continuous variables. The various properties of linear regression and its Python implementation have been covered in this article previously. Now, we shall find out how to\\
\\
4 min read](https://www.geeksforgeeks.org/linear-regression-using-pytorch/)

* * *

- [Linear Regression Using Tensorflow\\
\\
\\
We will briefly summarize Linear Regression before implementing it using TensorFlow. Since we will not get into the details of either Linear Regression or Tensorflow, please read the following articles for more details: Linear Regression (Python Implementation)Introduction to TensorFlowIntroduction\\
\\
6 min read](https://www.geeksforgeeks.org/linear-regression-using-tensorflow/)

* * *


## Hyperparameter tuning

- [Hyperparameter tuning\\
\\
\\
Machine Learning model is defined as a mathematical model with several parameters that need to be learned from the data. By training a model with existing data we can fit the model parameters. However there is another kind of parameter known as hyperparameters which cannot be directly learned from t\\
\\
8 min read](https://www.geeksforgeeks.org/hyperparameter-tuning/)

* * *


## Introduction to Convolution Neural Network

- [Introduction to Convolution Neural Network\\
\\
\\
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs are widely us\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)

* * *

- [Digital Image Processing Basics\\
\\
\\
Digital Image Processing means processing digital image by means of a digital computer. We can also say that it is a use of computer algorithms, in order to get enhanced image either to extract some useful information. Digital image processing is the use of algorithms and mathematical models to proc\\
\\
7 min read](https://www.geeksforgeeks.org/digital-image-processing-basics/)

* * *

- [Difference between Image Processing and Computer Vision\\
\\
\\
Image processing and Computer Vision both are very exciting field of Computer Science. Computer Vision: In Computer Vision, computers or machines are made to gain high-level understanding from the input digital images or videos with the purpose of automating tasks that the human visual system can do\\
\\
2 min read](https://www.geeksforgeeks.org/difference-between-image-processing-and-computer-vision/)

* * *

- [CNN \| Introduction to Pooling Layer\\
\\
\\
Pooling layer is used in CNNs to reduce the spatial dimensions (width and height) of the input feature maps while retaining the most important information. It involves sliding a two-dimensional filter over each channel of a feature map and summarizing the features within the region covered by the fi\\
\\
5 min read](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/)

* * *

- [CIFAR-10 Image Classification in TensorFlow\\
\\
\\
Prerequisites:Image ClassificationConvolution Neural Networks including basic pooling, convolution layers with normalization in neural networks, and dropout.Data Augmentation.Neural Networks.Numpy arrays.In this article, we are going to discuss how to classify images using TensorFlow. Image Classifi\\
\\
8 min read](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/)

* * *

- [Implementation of a CNN based Image Classifier using PyTorch\\
\\
\\
Introduction: Introduced in the 1980s by Yann LeCun, Convolution Neural Networks(also called CNNs or ConvNets) have come a long way. From being employed for simple digit classification tasks, CNN-based architectures are being used very profoundly over much Deep Learning and Computer Vision-related t\\
\\
9 min read](https://www.geeksforgeeks.org/implementation-of-a-cnn-based-image-classifier-using-pytorch/)

* * *

- [Convolutional Neural Network (CNN) Architectures\\
\\
\\
Convolutional Neural Network(CNN) is a neural network architecture in Deep Learning, used to recognize the pattern from structured arrays. However, over many years, CNN architectures have evolved. Many variants of the fundamental CNN Architecture This been developed, leading to amazing advances in t\\
\\
11 min read](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-architectures/)

* * *

- [Object Detection vs Object Recognition vs Image Segmentation\\
\\
\\
Object Recognition: Object recognition is the technique of identifying the object present in images and videos. It is one of the most important applications of machine learning and deep learning. The goal of this field is to teach machines to understand (recognize) the content of an image just like\\
\\
5 min read](https://www.geeksforgeeks.org/object-detection-vs-object-recognition-vs-image-segmentation/)

* * *

- [YOLO v2 - Object Detection\\
\\
\\
In terms of speed, YOLO is one of the best models in object recognition, able to recognize objects and process frames at the rate up to 150 FPS for small networks. However, In terms of accuracy mAP, YOLO was not the state of the art model but has fairly good Mean average Precision (mAP) of 63% when\\
\\
6 min read](https://www.geeksforgeeks.org/yolo-v2-object-detection/)

* * *


## Recurrent Neural Network

- [Natural Language Processing (NLP) Tutorial\\
\\
\\
Natural Language Processing (NLP) is the branch of Artificial Intelligence (AI) that gives the ability to machine understand and process human languages. Human languages can be in the form of text or audio format. Applications of NLPThe applications of Natural Language Processing are as follows: Voi\\
\\
5 min read](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)

* * *

- [Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging\\
\\
\\
Natural Language Toolkit (NLTK) is one of the largest Python libraries for performing various Natural Language Processing tasks. From rudimentary tasks such as text pre-processing to tasks like vectorized representation of text - NLTK's API has covered everything. In this article, we will accustom o\\
\\
5 min read](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/)

* * *

- [Word Embeddings in NLP\\
\\
\\
Word Embeddings are numeric representations of words in a lower-dimensional space, capturing semantic and syntactic information. They play a vital role in Natural Language Processing (NLP) tasks. This article explores traditional and neural approaches, such as TF-IDF, Word2Vec, and GloVe, offering i\\
\\
15+ min read](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)

* * *

- [Introduction to Recurrent Neural Networks\\
\\
\\
Recurrent Neural Networks (RNNs) work a bit different from regular neural networks. In neural network the information flows in one direction from input to output. However in RNN information is fed back into the system after each step. Think of it like reading a sentence, when you're trying to predic\\
\\
12 min read](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)

* * *

- [Recurrent Neural Networks Explanation\\
\\
\\
Today, different Machine Learning techniques are used to handle different types of data. One of the most difficult types of data to handle and the forecast is sequential data. Sequential data is different from other types of data in the sense that while all the features of a typical dataset can be a\\
\\
8 min read](https://www.geeksforgeeks.org/recurrent-neural-networks-explanation/)

* * *

- [Sentiment Analysis with an Recurrent Neural Networks (RNN)\\
\\
\\
Recurrent Neural Networks (RNNs) excel in sequence tasks such as sentiment analysis due to their ability to capture context from sequential data. In this article we will be apply RNNs to analyze the sentiment of customer reviews from Swiggy food delivery platform. The goal is to classify reviews as\\
\\
3 min read](https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/)

* * *

- [Short term Memory\\
\\
\\
In the wider community of neurologists and those who are researching the brain, It is agreed that two temporarily distinct processes contribute to the acquisition and expression of brain functions. These variations can result in long-lasting alterations in neuron operations, for instance through act\\
\\
5 min read](https://www.geeksforgeeks.org/short-term-memory/)

* * *

- [What is LSTM - Long Short Term Memory?\\
\\
\\
Long Short-Term Memory (LSTM) is an enhanced version of the Recurrent Neural Network (RNN) designed by Hochreiter & Schmidhuber. LSTMs can capture long-term dependencies in sequential data making them ideal for tasks like language translation, speech recognition and time series forecasting. Unli\\
\\
7 min read](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)

* * *

- [Long Short Term Memory Networks Explanation\\
\\
\\
Prerequisites: Recurrent Neural Networks To solve the problem of Vanishing and Exploding Gradients in a Deep Recurrent Neural Network, many variations were developed. One of the most famous of them is the Long Short Term Memory Network(LSTM). In concept, an LSTM recurrent unit tries to "remember" al\\
\\
7 min read](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/)

* * *

- [LSTM - Derivation of Back propagation through time\\
\\
\\
Long Short-Term Memory (LSTM) are a type of neural network designed to handle long-term dependencies by handling the vanishing gradient problem. One of the fundamental techniques used to train LSTMs is Backpropagation Through Time (BPTT) where we have sequential data. In this article we summarize ho\\
\\
4 min read](https://www.geeksforgeeks.org/lstm-derivation-of-back-propagation-through-time/)

* * *

- [Text Generation using Recurrent Long Short Term Memory Network\\
\\
\\
LSTMs are a type of neural network that are well-suited for tasks involving sequential data such as text generation. They are particularly useful because they can remember long-term dependencies in the data which is crucial when dealing with text that often has context that spans over multiple words\\
\\
6 min read](https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network/)

* * *


Like23

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)

Improvement

Suggest changes

Suggest Changes

Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.

![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)

Create Improvement

Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.

![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)

Suggest Changes

min 4 words, max Words Limit:1000

## Thank You!

Your suggestions are valuable to us.

## What kind of Experience do you want to share?

[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=389092059.1745056725&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1948265680)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

Sign In

By creating this account, you agree to our [Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/) & [Cookie Policy.](https://www.geeksforgeeks.org/legal/privacy-policy/#:~:text=the%20appropriate%20measures.-,COOKIE%20POLICY,-A%20cookie%20is)

# Create Account

Already have an account ?Log in

Continue with Google

or

Username or Email

Password

Institution / Organization

```

```

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password