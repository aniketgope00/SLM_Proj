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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/?type%3Darticle%26id%3D266624&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Long Short Term Memory Networks Explanation\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/)

# What is LSTM – Long Short Term Memory?

Last Updated : 05 Apr, 2025

Comments

Improve

Suggest changes

39 Likes

Like

Report

Long Short-Term Memory (LSTM) is an enhanced version of the [Recurrent Neural Network (RNN)](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) designed by Hochreiter & Schmidhuber. LSTMs can capture long-term dependencies in sequential data making them ideal for tasks like language translation, speech recognition and time series forecasting.

Unlike traditional RNNs which use a single hidden state passed through time LSTMs introduce a memory cell that holds information over extended periods addressing the challenge of learning long-term dependencies.

## Problem with Long-Term Dependencies in RNN

Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining a hidden state that captures information from previous time steps. However they often face challenges in learning long-term dependencies where information from distant time steps becomes crucial for making accurate predictions for current state. This problem is known as the vanishing gradient or exploding gradient problem.

- **Vanishing Gradient**: When training a model over time, the gradients (which help the model learn) can shrink as they pass through many steps. This makes it hard for the model to learn long-term patterns since earlier information becomes almost irrelevant.
- **Exploding Gradient**: Sometimes, gradients can grow too large, causing instability. This makes it difficult for the model to learn properly, as the updates to the model become erratic and unpredictable.

Both of these issues make it challenging for standard RNNs to effectively capture long-term dependencies in sequential data.

## LSTM Architecture

LSTM architectures involves the memory cell which is controlled by three gates: the _input gate_, the _forget gate_ and the _output gate_. These gates decide what information to add to, remove from and output from the memory cell.

- **Input gate**: Controls what information is added to the memory cell.
- **Forget gate**: Determines what information is removed from the memory cell.
- **Output gate**: Controls what information is output from the memory cell.

This allows LSTM networks to selectively retain or discard information as it flows through the network which allows them to learn long-term dependencies. The network has a hidden state which is like its short-term memory. This memory is updated using the current input, the previous hidden state and the current state of the memory cell.

## Working of LSTM

LSTM architecture has a chain structure that contains four neural networks and different memory blocks called **cells**.

![gate_of_lstm](https://media.geeksforgeeks.org/wp-content/uploads/20250404172141987003/gate_of_lstm.webp)

LSTM Model

Information is retained by the cells and the memory manipulations are done by the **gates.** There are three gates –

### **Forget Gate**

The information that is no longer useful in the cell state is removed with the forget gate. Two inputs _x_ __t__ (input at the particular time) and _h_ __t-1__ (previous cell output) are fed to the gate and multiplied with weight matrices followed by the addition of bias. The resultant is passed through an activation function which gives a binary output. If for a particular cell state the output is 0, the piece of information is forgotten and for output 1, the information is retained for future use.

The equation for the forget gate is:

ft=σ(Wf⋅\[ht−1,xt\]+bf) f\_t = σ(W\_f · \[h\_{t-1}, x\_t\] + b\_f)ft​=σ(Wf​⋅\[ht−1​,xt​\]+bf​)

where:

- W\_f represents the weight matrix associated with the forget gate.
- \[h\_t-1, x\_t\] denotes the concatenation of the current input and the previous hidden state.
- b\_f is the bias with the forget gate.
- σ is the sigmoid activation function.

![forget_gate](https://media.geeksforgeeks.org/wp-content/uploads/20250404172507256364/forget_gate.jpg)

Forget Gate

### **Input gate**

The addition of useful information to the cell state is done by the input gate. First, the information is regulated using the sigmoid function and filter the values to be remembered similar to the forget gate using inputs _h_ __t-1__ and _x_ __t__. . Then, a vector is created using _tanh_ function that gives an output from -1 to +1, which contains all the possible values from ht-1 and _x_ __t__. At last, the values of the vector and the regulated values are multiplied to obtain the useful information. The equation for the input gate is:

it=σ(Wi⋅\[ht−1,xt\]+bi) i\_t = σ(W\_i · \[h\_{t-1}, x\_t\] + b\_i)it​=σ(Wi​⋅\[ht−1​,xt​\]+bi​)

C^t=tanh(Wc⋅\[ht−1,xt\]+bc)Ĉ\_t = tanh(W\_c · \[h\_{t-1}, x\_t\] + b\_c)C^t​=tanh(Wc​⋅\[ht−1​,xt​\]+bc​)

We multiply the previous state by ft, disregarding the information we had previously chosen to ignore. Next, we include it∗Ct. This represents the updated candidate values, adjusted for the amount that we chose to update each state value.

Ct=ft⊙Ct−1+it⊙C^tC\_t = f\_t ⊙ C\_{t-1} + i\_t ⊙ Ĉ\_tCt​=ft​⊙Ct−1​+it​⊙C^t​

where

-  ⊙ denotes element-wise multiplication
- tanh is tanh activation function

![input_gate](https://media.geeksforgeeks.org/wp-content/uploads/20250404172524079223/input_gate.jpg)

Input Gate

### **Output gate**

The task of extracting useful information from the current cell state to be presented as output is done by the output gate. First, a vector is generated by applying tanh function on the cell. Then, the information is regulated using the sigmoid function and filter by the values to be remembered using inputsht−1h\_{t-1} ht−1​and xtx\_txt​. At last, the values of the vector and the regulated values are multiplied to be sent as an output and input to the next cell. The equation for the output gate is:

ot=σ(Wo⋅\[ht−1,xt\]+bo)o\_t = σ(W\_o · \[h\_{t-1}, x\_t\] + b\_o)ot​=σ(Wo​⋅\[ht−1​,xt​\]+bo​)

![output_gate](https://media.geeksforgeeks.org/wp-content/uploads/20250404172543950041/output_gate.jpg)

Output Gate

## Bidirectional LSTM Model

[Bidirectional LSTM](https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/) (Bi LSTM/ BLSTM) is a variation of normal LSTM which processes sequential data in both forward and backward directions. This allows Bi LSTM to learn longer-range dependencies in sequential data than traditional LSTMs which can only process sequential data in one direction.

- Bi LSTMs are made up of two LSTM networks one that processes the input sequence in the forward direction and one that processes the input sequence in the backward direction.
- The outputs of the two LSTM networks are then combined to produce the final output.

> LSTM models including Bi LSTMs have demonstrated state-of-the-art performance across various tasks such as machine translation, speech recognition and text summarization.

LSTM networks can be stacked to form deeper models allowing them to learn more complex patterns in data. Each layer in the stack captures different levels of information and time-based relationships in the input.

## **Applications of LSTM**

Some of the famous applications of LSTM includes:

- **Language Modeling**: Used in tasks like language modeling, machine translation and text summarization. These networks learn the dependencies between words in a sentence to generate coherent and grammatically correct sentences.
- **Speech Recognition**: Used in transcribing speech to text and recognizing spoken commands. By learning speech patterns they can match spoken words to corresponding text.
- **Time Series Forecasting**: Used for predicting stock prices, weather and energy consumption. They learn patterns in time series data to predict future events.
- **Anomaly Detection**: Used for detecting fraud or network intrusions. These networks can identify patterns in data that deviate drastically and flag them as potential anomalies.
- **Recommender Systems**: In recommendation tasks like suggesting movies, music and books. They learn user behavior patterns to provide personalized suggestions.
- **Video Analysis**: Applied in tasks such as object detection, activity recognition and action classification. When combined with [Convolutional Neural Networks (CNNs)](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) they help analyze video data and extract useful information.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/)

[Long Short Term Memory Networks Explanation](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/)

[![author](https://media.geeksforgeeks.org/auth/profile/gkqgqisrn09d4pga5j4n)](https://www.geeksforgeeks.org/user/aakarshachug/)

[aakarshachug](https://www.geeksforgeeks.org/user/aakarshachug/)

Follow

39

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Computer Subject](https://www.geeksforgeeks.org/category/computer-subject/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

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


Like39

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=832787522.1745057223&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102015666~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1310371515)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057222711&cv=11&fst=1745057222711&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3h1v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102015666~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fdeep-learning-introduction-to-long-short-term-memory%2F&_ng=1&hn=www.googleadservices.com&frm=0&tiba=What%20is%20LSTM%20%E2%80%93%20Long%20Short%20Term%20Memory%3F%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=586881170.1745057223&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)