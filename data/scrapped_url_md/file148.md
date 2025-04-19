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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/?type%3Darticle%26id%3D910015&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Word Embeddings in NLP\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)

# Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging

Last Updated : 01 May, 2024

Comments

Improve

Suggest changes

5 Likes

Like

Report

**Natural Language Toolkit (NLTK)** is one of the largest Python libraries for performing various Natural Language Processing tasks. From rudimentary tasks such as text pre-processing to tasks like vectorized representation of text – **NLTK’s API** has covered everything. In this article, we will accustom ourselves to the basics of NLTK and perform some crucial NLP tasks: Tokenization, Stemming, Lemmatization, and POS Tagging.

Table of Content

- [What is the Natural Language Toolkit (NLTK)?](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#what-is-the-natural-language-toolkit-nltk)
- [Tokenization](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#tokenization)
- [Stemming and Lemmatization](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#stemming-and-lemmatization)
- [Stemming](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#stemming)
- [Lemmatization](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#lemmatization)
- [Part of Speech Tagging](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/#part-of-speech-tagging)

## What is the Natural Language Toolkit (NLTK)?

As discussed earlier, NLTK is [Python’s](https://www.geeksforgeeks.org/python-programming-language/) API library for performing an array of tasks in human language. It can perform a variety of operations on textual data, such as classification, tokenization, stemming, tagging, Leparsing, [semantic reasoning](https://www.geeksforgeeks.org/understanding-semantic-analysis-nlp/), etc.

**Installation:**

NLTK can be installed simply using pip or by running the following code.

```
! pip install nltk

```

**Accessing Additional Resources:**

To incorporate the usage of additional resources, such as recourses of languages other than English – you can run the following in a python script. It has to be done only once when you are running it for the first time in your system.

Python`
import nltk
nltk.download('all')
`

Now, having installed NLTK successfully in our system, let’s perform some basic operations on text data using NLTK.

## Tokenization

Tokenization refers to break down the text into smaller units. It entails splitting paragraphs into sentences and sentences into words. It is one of the initial steps of any NLP pipeline. Let us have a look at the two major kinds of tokenization that NLTK provides:

### **Work Tokenization**

It involves breaking down the text into words.

```
 "I study Machine Learning on GeeksforGeeks." will be word-tokenized as
  ['I', 'study', 'Machine', 'Learning', 'on', 'GeeksforGeeks', '.'].

```

### **Sentence Tokenization**

It involves breaking down the text into individual sentences.

```
Example:
"I study Machine Learning on GeeksforGeeks. Currently, I'm studying NLP"
 will be sentence-tokenized as
 ['I study Machine Learning on GeeksforGeeks.', 'Currently, I'm studying NLP.']

```

In Python, both these tokenizations can be implemented in NLTK as follows:

Python`
# Tokenization using NLTK
from nltk import word_tokenize, sent_tokenize
sent = "GeeksforGeeks is a great learning platform.\
It is one of the best for Computer Science students."
print(word_tokenize(sent))
print(sent_tokenize(sent))
`

**Output:**

```
['GeeksforGeeks', 'is', 'a', 'great', 'learning', 'platform', '.',\
'It', 'is', 'one', 'of', 'the', 'best', 'for', 'Computer', 'Science', 'students', '.']
['GeeksforGeeks is a great learning platform.',\
 'It is one of the best for Computer Science students.']

```

## **Stemming and Lemmatization**

When working with Natural Language, we are not much interested in the form of words – rather, we are concerned with the meaning that the words intend to convey. Thus, we try to map every word of the language to its root/base form. This process is called canonicalization.

**E.g**. The words ‘play’, ‘plays’, ‘played’, and ‘playing’ convey the same action – hence, we can map them all to their base form i.e. ‘play’.

Now, there are two widely used canonicalization techniques: [**Stemming**](https://www.geeksforgeeks.org/introduction-to-stemming/) and **Lemmatization.**

## Stemming

Stemming generates the base word from the inflected word by removing the affixes of the word. It has a set of pre-defined rules that govern the dropping of these affixes. It must be noted that stemmers might not always result in semantically meaningful base words.  Stemmers are faster and computationally less expensive than lemmatizers.

In the following code, we will be stemming words using Porter Stemmer – one of the most widely used stemmers:

Python`
from nltk.stem import PorterStemmer
# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("play"))
print(porter.stem("playing"))
print(porter.stem("plays"))
print(porter.stem("played"))
`

**Output:**

```
play
play
play
play

```

We can see that all the variations of the word ‘play’ have been reduced to the same word  – ‘play’. In this case, the output is a meaningful word, ‘play’. However, this is not always the case. Let us take an example.

Please note that these groups are stored in the lemmatizer; there is no removal of affixes as in the case of a stemmer.

Python`
from nltk.stem import PorterStemmer
# create an object of class PorterStemmer
porter = PorterStemmer()
print(porter.stem("Communication"))
`

**Output:**

```
commun

```

The stemmer reduces the word ‘communication’ to a base word ‘commun’ which is meaningless in itself.

## Lemmatization

Lemmatization involves grouping together the inflected forms of the same word. This way, we can reach out to the base form of any word which will be meaningful in nature. The base from here is called the Lemma.

Lemmatizers are slower and computationally more expensive than stemmers.

```
Example:
'play', 'plays', 'played', and 'playing' have 'play' as the lemma.

```

In Python, both these tokenizations can be implemented in NLTK as follows:

Python`
from nltk.stem import WordNetLemmatizer
# create an object of class WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("plays", 'v'))
print(lemmatizer.lemmatize("played", 'v'))
print(lemmatizer.lemmatize("play", 'v'))
print(lemmatizer.lemmatize("playing", 'v'))
`

**Output:**

```
play
play
play
play
```

Please note that in lemmatizers, we need to pass the Part of Speech of the word along with the word as a function argument.

Also, lemmatizers always result in meaningful base words. Let us take the same example as we took in the case for stemmers.

Python`
from nltk.stem import WordNetLemmatizer
# create an object of class WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("Communication", 'v'))
`

**Output:**

```
Communication

```

## Part of Speech Tagging

Part of Speech (POS) tagging refers to assigning each word of a sentence to its part of speech. It is significant as it helps to give a better syntactic overview of a sentence.

```
Example:
"GeeksforGeeks is a Computer Science platform."
Let's see how NLTK's POS tagger will tag this sentence.

```

In Python, both these tokenizations can be implemented in NLTK as follows:

Python`
from nltk import pos_tag
from nltk import word_tokenize
text = "GeeksforGeeks is a Computer Science platform."
tokenized_text = word_tokenize(text)
tags = tokens_tag = pos_tag(tokenized_text)
tags
`

**Output:**

```
[('GeeksforGeeks', 'NNP'),\
 ('is', 'VBZ'),\
 ('a', 'DT'),\
 ('Computer', 'NNP'),\
 ('Science', 'NNP'),\
 ('platform', 'NN'),\
 ('.', '.')]

```

## Conclusion

In conclusion, the Natural Language Toolkit (NLTK) works as a powerful Python library that a wide range of tools for Natural Language Processing (NLP). From fundamental tasks like text pre-processing to more advanced operations such as semantic reasoning, NLTK provides a versatile API that caters to the diverse needs of language-related tasks.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Stemming & Lemmatisation

[Visit Course![explore course icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/Vector11.svg)](https://www.geeksforgeeks.org/courses/data-science-live?utm_source=geeksforgeeks&utm_medium=article_tab_visit_course&utm_campaign=gfg_datascience)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)

[Word Embeddings in NLP](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)

[![author](https://media.geeksforgeeks.org/auth/profile/ryy64fbirebgxy9hlgc7)](https://www.geeksforgeeks.org/user/suvratarora06/)

[suvratarora06](https://www.geeksforgeeks.org/user/suvratarora06/)

Follow

5

Improve

Article Tags :

- [Python](https://www.geeksforgeeks.org/category/programming-language/python/)
- [Technical Scripter](https://www.geeksforgeeks.org/category/technical-scripter/)
- [Technical Scripter 2022](https://www.geeksforgeeks.org/tag/technical-scripter-2022/)

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


Like5

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=761182558.1745057508&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1017730201)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=5w4hocau6gbt)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=9e83rq9seoi0)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=invisible&cb=5u4saqicj5yk)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)