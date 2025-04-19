- [Python Tutorial](https://www.geeksforgeeks.org/python-programming-language-tutorial/)
- [Interview Questions](https://www.geeksforgeeks.org/python-interview-questions/)
- [Python Quiz](https://www.geeksforgeeks.org/python-quizzes/)
- [Python Glossary](https://www.geeksforgeeks.org/python-glossary/)
- [Python Projects](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/)
- [Practice Python](https://www.geeksforgeeks.org/python-exercises-practice-questions-and-solutions/)
- [Data Science With Python](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Python Web Dev](https://www.geeksforgeeks.org/python-web-development-django/)
- [DSA with Python](https://www.geeksforgeeks.org/python-data-structures-and-algorithms/)
- [Python OOPs](https://www.geeksforgeeks.org/python-oops-concepts/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/?type%3Darticle%26id%3D926471&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Multi-Layer Perceptron Learning in Tensorflow\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)

# Single Layer Perceptron in TensorFlow

Last Updated : 08 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

[**Single Layer Perceptron**](https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/) is inspired by biological neurons and their ability to process information. To understand the SLP we first need to break down the workings of a single artificial neuron which is the fundamental building block of neural networks. An **artificial neuron** is a simplified computational model that mimics the behavior of a biological neuron. It takes inputs, processes them and produces an output. Here’s how it works step by step:

- Receive signal from outside.
- Process the signal and decide whether we need to send information or not.
- Communicate the signal to the target cell, which can be another neuron or gland.

![Structure of a biological neuron](https://media.geeksforgeeks.org/wp-content/uploads/20221219111342/Biological-Neuron-and-similarity-with-neural-network.png)

Structure of a biological neuron

In the same way neural networks also work.

![Neural Network in Machine Learning](https://media.geeksforgeeks.org/wp-content/uploads/nodeNeural.jpg)

Neural Network in Machine Learning

## **Single Layer Perceptron**

It is one of the oldest and first introduced neural networks. It was proposed by **Frank Rosenblatt** in **1958**. Perceptron is also known as an artificial neural network. Perceptron is mainly used to compute the [logical gate](https://www.geeksforgeeks.org/logic-gates-definition-types-uses/) like **AND, OR and NOR** which has binary input and binary output.

The main functionality of the perceptron is:-

- Takes input from the input layer
- Weight them up and sum it up.
- Pass the sum to the nonlinear function to produce the output.

![Single-layer neural network](https://media.geeksforgeeks.org/wp-content/uploads/20221219111343/Single-Layer-Perceptron.png)

Single-layer neural network

Here activation functions can be anything like **sigmoid, tanh, relu** based on the requirement we will be choosing the most appropriate nonlinear [activation function](https://www.geeksforgeeks.org/activation-functions-neural-networks/) to produce the better result. Now let us implement a single-layer perceptron.

## **Implementation of Single-layer Perceptron**

Let’s build a simple **single-layer perceptron** to recognize handwritten digits from the [**MNIST dataset**](https://www.geeksforgeeks.org/mnist-dataset/) using **TensorFlow**. This model will help you understand how neural networks work at the most basic level.

### **Step1: Import necessary libraries**

- [**Numpy**](https://www.geeksforgeeks.org/python-numpy/)– Numpy arrays are very fast and can perform large computations in a very short time.
- [**Matplotlib**](https://www.geeksforgeeks.org/matplotlib-tutorial/)– This library is used to draw visualizations.
- [**TensorFlow**](https://www.geeksforgeeks.org/introduction-to-tensorflow/) – This is an open-source library that is used for Machine Learning and Artificial intelligence and provides a range of functions to achieve complex functionalities with single lines of code.

Python`
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
`

### **Step 2: Load the Dataset**

Now we load the dataset using “Keras” from the imported version of tensor flow.

Python`
(x_train, y_train),\
    (x_test, y_test) = keras.datasets.mnist.load_data()
`

Now display the shape and image of the single image in the dataset. The image size contains a 28\*28 matrix and length of the training set is 60,000 and the testing set is 10,000. Each image is **28×28 pixels**.

- The pixel values range from **0 (black)** to **255 (white)**.
- **`matshow()`** displays the image in a grid format.

Python`
len(x_train)
len(x_test)
x_train[0].shape
plt.matshow(x_train[0])
`

**Output:**

![Sample image from the training dataset](https://media.geeksforgeeks.org/wp-content/uploads/20221213191643/31.png)

Sample image from the training dataset

### **Step 3: Normalize the Dataset**

Now normalize the dataset in order to compute the calculations in a fast and accurate manner. [**Normalization**](https://www.geeksforgeeks.org/what-is-data-normalization/) helps the model train faster and improves accuracy. Neural networks expect input in a **1D format** so we **flatten** the 2D 28×28 images into 784-length vectors.

Python`
x_train = x_train/255
x_test = x_test/255
# Flatting the dataset in order
# to compute for model building
x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)
`

### **Step 4: Building a neural network**

Now we will build neural network with single-layer perception. Here we can observe as the model is a single-layer perceptron that only contains one input layer and one output layer there is no presence of the hidden layers.

- 10 output neurons (one for each digit from 0 to 9).
- `input_shape=(784)` matches the flattened image shape.
- We use **sigmoid activation** which outputs probabilities between 0 and 1 for each digit class.

Python`
model = keras.Sequential([\
    keras.layers.Dense(10, input_shape=(784,),\
                       activation='sigmoid')\
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train_flatten, y_train, epochs=5)
`

**Output:**

![Training progress per epoch](https://media.geeksforgeeks.org/wp-content/uploads/20221213192253/32.png)

Training progress per epoch

### **Step 5: Model Evaluation**

After training we test the model’s performance on unseen data. This gives the **loss** and **accuracy** on the test dataset.

Python`
model.evaluate(x_test_flatten, y_test)
`

**Output:**

![Models performance on the testing data](https://media.geeksforgeeks.org/wp-content/uploads/20221213192509/33.png)

Models performance on the testing data

Even with such a **simple model** we achieved **over 92% accuracy.** That’s quite impressive for a neural network with just one layer. However for even better results we could add **hidden layers** or use more complex architectures like [CNNs (Convolutional Neural Networks)](https://www.geeksforgeeks.org/introduction-convolution-neural-network/).

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)

[Multi-Layer Perceptron Learning in Tensorflow](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)

[D](https://www.geeksforgeeks.org/user/dominicajay74/)

[dominicajay74](https://www.geeksforgeeks.org/user/dominicajay74/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Python](https://www.geeksforgeeks.org/category/programming-language/python/)
- [Technical Scripter](https://www.geeksforgeeks.org/category/technical-scripter/)
- [Technical Scripter 2022](https://www.geeksforgeeks.org/tag/technical-scripter-2022/)
- [Tensorflow](https://www.geeksforgeeks.org/tag/tensorflow/)

+1 More

Practice Tags :

- [python](https://www.geeksforgeeks.org/explore?category=python)

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
To put it in simple terms, an artificial neuron calculates the 'weighted sum' of its inputs and adds a bias, as shown in the figure below by the net input. Mathematically, \[Tex\]\\text{Net Input} =\\sum \\text{(Weight} \\times \\text{Input)+Bias}\[/Tex\] Now the value of net input can be any anything from -\\
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


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1162339795.1745056722&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=239798192)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056722180&cv=11&fst=1745056722180&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb858768136&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fsingle-layer-perceptron-in-tensorflow%2F&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Single%20Layer%20Perceptron%20in%20TensorFlow%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=585835744.1745056722&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)