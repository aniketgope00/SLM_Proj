- [Deep Learning Tutorial](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Data Analysis Tutorial](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Python â€“ Data visualization tutorial](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [R](https://www.geeksforgeeks.org/r-tutorial/)
- [Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/)
- [Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning-projects/)_)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)_)
- [Machine Learning Mathematics](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Deep Learning Project](https://www.geeksforgeeks.org/5-deep-learning-project-ideas-for-beginners/)_)
- [Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning-interview-questions/)_)
- [Computer Vision Tutorial](https://www.geeksforgeeks.org/computer-vision/)
- [Computer Vision Projects](https://www.geeksforgeeks.org/computer-vision-projects/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [NLP Project](https://www.geeksforgeeks.org/nlp-project-ideas-for-beginners/))
- [NLP Interview Questions](https://www.geeksforgeeks.org/nlp-interview-questions/))
- [Statistics with Python](https://www.geeksforgeeks.org/statistics-with-python/)
- [100 Days of Machine Learning](https://www.geeksforgeeks.org/100-days-of-machine-learning/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/feedforward-neural-network/?type%3Darticle%26id%3D1272287&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Feedforward Neural Networks (FNNs) in R\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/feedforward-neural-networks-fnns-in-r/)

# Feedforward Neural Network

Last Updated : 19 Mar, 2025

Comments

Improve

Suggest changes

5 Likes

Like

Report

Feedforward Neural Network (FNN) is a type of artificial [neural network](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) in which information flows in a single direction—from the input layer through hidden layers to the output layer—without loops or feedback. It is mainly used for pattern recognition tasks like image and speech classification.

> For example in a credit scoring system banks use an FNN which analyze users' financial profiles—such as income, credit history and spending habits—to determine their creditworthiness.
>
> Each piece of information flows through the network’s layers where various calculations are made to produce a final score.

## Structure of a Feedforward Neural Network

Feedforward Neural Networks have a structured layered design where data flows sequentially through each layer.

1. **Input Layer**: The input layer consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
2. **Hidden Layers**: One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
3. **Output Layer**: The output layer provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.

Each connection between neurons in these layers has an associated weight that is adjusted during the training process to minimize the error in predictions.

![FNN](https://media.geeksforgeeks.org/wp-content/uploads/20240601001059/FNN.jpg)Feed Forward Neural Network

## Activation Functions

[**Activation functions**](https://www.geeksforgeeks.org/activation-functions-neural-networks/) introduce non-linearity into the network enabling it to learn and model complex data patterns.

Common activation functions include:

- [**Sigmoid**](https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/): σ(x)=σ(x)=11+e−x\\sigma(x) = \\frac{1}{1 + e^{-x}}σ(x)=1+e−x1​
- [**Tanh**](https://www.geeksforgeeks.org/tanh-activation-in-neural-network/): tanh(x)=ex−e−xex+e−x\\text{tanh}(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
tanh(x)=ex+e−xex−e−x​
- [**ReLU**](https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/): ReLU(x)=max⁡(0,x)\\text{ReLU}(x) = \\max(0, x)ReLU(x)=max(0,x)

## Training a Feedforward Neural Network

Training a Feedforward Neural Network involves adjusting the weights of the neurons to minimize the error between the predicted output and the actual output. This process is typically performed using backpropagation and gradient descent.

1. **Forward Propagation**: During forward propagation the input data passes through the network and the output is calculated.
2. **Loss Calculation**: The loss (or error) is calculated using a loss function such as Mean Squared Error (MSE) for regression tasks or Cross-Entropy Loss for classification tasks.
3. **Backpropagation**: In backpropagation the error is propagated back through the network to update the weights. The gradient of the loss function with respect to each weight is calculated and the weights are adjusted using gradient descent.

![es](https://media.geeksforgeeks.org/wp-content/uploads/20240601001819/es.png)Forward Propagation

## Gradient Descent

[**Gradient Descent**](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/) is an optimization algorithm used to minimize the loss function by iteratively updating the weights in the direction of the negative gradient. Common variants of gradient descent include:

- **Batch Gradient Descent**: Updates weights after computing the gradient over the entire dataset.
- **Stochastic Gradient Descent (SGD)**: Updates weights for each training example individually.
- **Mini-batch Gradient Descent**: It Updates weights after computing the gradient over a small batch of training examples.

## Evaluation of Feedforward neural network

Evaluating the performance of the trained model involves several metrics:

- **Accuracy**: The proportion of correctly classified instances out of the total instances.
- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix:** A table used to describe the performance of a classification model, showing the true positives, true negatives, false positives, and false negatives.

## Code Implementation of Feedforward neural network

This code demonstrates the process of building, training and evaluating a neural network model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

The model architecture is defined using the **Sequential API** consisting of:

- a Flatten layer to convert the 2D image input into a 1D array
- a Dense layer with 128 neurons and ReLU activation
- a final Dense layer with 10 neurons and softmax activation to output probabilities for each digit class.

Model is compiled with the Adam optimizer, SparseCategoricalCrossentropy loss function and SparseCategoricalAccuracy metric and then trained for 5 epochs on the training data.

Python`
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Build the model
model = Sequential([\
    Flatten(input_shape=(28, 28)),\
    Dense(128, activation='relu'),\
    Dense(10, activation='softmax')\
])
# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=[SparseCategoricalAccuracy()])
# Train the model
model.fit(x_train, y_train, epochs=5)
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc}')
`

**Output:**

> **Test accuracy:** 0.9767000079154968

By understanding their architecture, activation functions, and training process, one can appreciate the capabilities and limitations of these networks. Continuous advancements in optimization techniques and activation functions have made feedforward networks more efficient and effective, contributing to the broader field of artificial intelligence.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/feedforward-neural-networks-fnns-in-r/)

[Feedforward Neural Networks (FNNs) in R](https://www.geeksforgeeks.org/feedforward-neural-networks-fnns-in-r/)

[![author](https://media.geeksforgeeks.org/auth/profile/1vqxkri54ygmulv49l3v)](https://www.geeksforgeeks.org/user/gouravlohar/)

[gouravlohar](https://www.geeksforgeeks.org/user/gouravlohar/)

Follow

5

Improve

Article Tags :

- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [NLP](https://www.geeksforgeeks.org/category/ai-ml-ds/nlp/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Deep-Learning](https://www.geeksforgeeks.org/tag/deep-learning/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+1 More

### Similar Reads

[Feedforward neural network\\
\\
\\
Feedforward Neural Network (FNN) is a type of artificial neural network in which information flows in a single directionâ€”from the input layer through hidden layers to the output layerâ€”without loops or feedback. It is mainly used for pattern recognition tasks like image and speech classification. For\\
\\
6 min read](https://www.geeksforgeeks.org/feedforward-neural-network/?ref=ml_lbp)
[Feedforward Neural Networks (FNNs) in R\\
\\
\\
Feedforward Neural Networks (FNNs) are a type of artificial neural network where connections between nodes do not form a cycle. This means that data moves in one directionâ€”forwardâ€”from the input layer through the hidden layers to the output layer. These networks are often used for tasks such as clas\\
\\
6 min read](https://www.geeksforgeeks.org/feedforward-neural-networks-fnns-in-r/?ref=ml_lbp)
[Hopfield Neural Network\\
\\
\\
The Hopfield Neural Networks, invented by Dr John J. Hopfield consists of one layer of 'n' fully connected recurrent neurons. It is generally used in performing auto-association and optimization tasks. It is calculated using a converging interactive process and it generates a different response than\\
\\
7 min read](https://www.geeksforgeeks.org/hopfield-neural-network/?ref=ml_lbp)
[Feedback System in Neural Networks\\
\\
\\
A feedback system in neural networks is a mechanism where the output is fed back into the network to influence subsequent outputs, often used to enhance learning and stability. This article provides an overview of the working of the feedback loop in Neural Networks. Understanding Feedback SystemIn d\\
\\
6 min read](https://www.geeksforgeeks.org/feedback-system-in-neural-networks/?ref=ml_lbp)
[Multilayer Feed-Forward Neural Network in Data Mining\\
\\
\\
Multilayer Feed-Forward Neural Network(MFFNN) is an interconnected Artificial Neural Network with multiple layers that has neurons with weights associated with them and they compute the result using activation functions. It is one of the types of Neural Networks in which the flow of the network is f\\
\\
3 min read](https://www.geeksforgeeks.org/multilayer-feed-forward-neural-network-in-data-mining/?ref=ml_lbp)
[Understanding Multi-Layer Feed Forward Networks\\
\\
\\
Let's understand how errors are calculated and weights are updated in backpropagation networks(BPNs). Consider the following network in the below figure. The network in the above figure is a simple multi-layer feed-forward network or backpropagation network. It contains three layers, the input layer\\
\\
7 min read](https://www.geeksforgeeks.org/understanding-multi-layer-feed-forward-networks/?ref=ml_lbp)
[Deep Neural Network With L - Layers\\
\\
\\
This article aims to implement a deep neural network with an arbitrary number of hidden layers each containing different numbers of neurons. We will be implementing this neural net using a few helper functions and at last, we will combine these functions to make the L-layer neural network model.L -\\
\\
11 min read](https://www.geeksforgeeks.org/deep-neural-network-with-l-layers/?ref=ml_lbp)
[What is a Neural Network?\\
\\
\\
Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making. In this article, we will explore the fundament\\
\\
14 min read](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/?ref=ml_lbp)
[What are Graph Neural Networks?\\
\\
\\
Graph Neural Networks (GNNs) are a neural network specifically designed to work with data represented as graphs. Unlike traditional neural networks, which operate on grid-like data structures like images (2D grids) or text (sequential), GNNs can model complex, non-Euclidean relationships in data, su\\
\\
13 min read](https://www.geeksforgeeks.org/what-are-graph-neural-networks/?ref=ml_lbp)
[Neural Network Node\\
\\
\\
In the realm of artificial intelligence and machine learning particularly within the neural networks the concept of a "node" is fundamental. Nodes, often referred to as neurons in the context of neural networks are the core computational units that drive the learning process. They play a crucial rol\\
\\
5 min read](https://www.geeksforgeeks.org/neural-network-node/?ref=ml_lbp)

Like5

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/feedforward-neural-network/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=335399289.1745056914&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1857778597)

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

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)