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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/layers-in-artificial-neural-networks-ann/?type%3Darticle%26id%3D1284346&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Numba vs. Cython: A Technical Comparison\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/numba-vs-cython-a-technical-comparison/)

# Layers in Artificial Neural Networks (ANN)

Last Updated : 01 Mar, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

In Artificial Neural Networks (ANNs), data flows from the input layer to the output layer through one or more hidden layers. Each layer consists of neurons that receive input, process it, and pass the output to the next layer. The layers work together to extract features, transform data, and make predictions.

**An** [**Artificial Neural Networks (ANNs)**](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/) **consists of three primary types of layers:**

- **Input Layer**
- **Hidden Layers**
- **Output Layer**

Each layer is composed of nodes (neurons) that are interconnected. The layers work together to process data through a series of transformations.

![Layers-in-ANN](https://media.geeksforgeeks.org/wp-content/uploads/20240719174919/Layers-in-ANN.webp)ANN Layers

## Basic Layers in ANN

### 1\. Input Layer

Input layer is the first layer in an ANN and is responsible for receiving the raw input data. This layer's neurons correspond to the features in the input data. For example, in image processing, each neuron might represent a pixel value. The input layer doesn't perform any computations but passes the data to the next layer.

**Key Points:**

- **Role**: Receives raw data.
- **Function**: Passes data to the hidden layers.
- **Example**: For an image, the input layer would have neurons for each pixel value.

![Input-Layer-in-an-ANN](https://media.geeksforgeeks.org/wp-content/uploads/20240719175629/Input-Layer-in-an-ANN.webp)Input Layer in ANN

### 2\. Hidden Layers

**Hidden Layers** are the intermediate layers between the input and output layers. They perform most of the computations required by the network. Hidden layers can vary in number and size, depending on the complexity of the task.

Each hidden layer applies a set of weights and biases to the input data, followed by an activation function to introduce non-linearity.

### 3\. **Output Layer**

**Output Layer** is the final layer in an ANN. It produces the output predictions. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.

The activation function used in the output layer depends on the type of problem:

- **Softmax** for multi-class classification
- **Sigmoid** for binary classification
- **Linear** for regression

> For better understanding of the activation functions, Refer to the article - [Activation functions in Neural Networks](https://www.geeksforgeeks.org/activation-functions-neural-networks/)

## Types of Hidden Layers in Artificial Neural Networks

Till now we have covered the basic layers: input, hidden, and output. Let’s now dive into the specific types of hidden layers.

### 1\. Dense (Fully Connected) Layer

[**Dense (Fully Connected) Layer**](https://www.geeksforgeeks.org/what-is-fully-connected-layer-in-deep-learning/) is the most common type of hidden layer in an ANN. Every neuron in a dense layer is connected to every neuron in the previous and subsequent layers. This layer performs a weighted sum of inputs and applies an activation function to introduce non-linearity. The activation function (like ReLU, Sigmoid, or Tanh) helps the network learn complex patterns.

- **Role**: Learns representations from input data.
- **Function**: Performs weighted sum and activation.

### 2\. Convolutional Layer

[**Convolutional layers**](https://www.geeksforgeeks.org/what-are-convolution-layers/) are used in Convolutional Neural Networks (CNNs) for image processing tasks. They apply convolution operations to the input, capturing spatial hierarchies in the data. Convolutional layers use filters to scan across the input and generate feature maps. This helps in detecting edges, textures, and other visual features.

- **Role**: Extracts spatial features from images.
- **Function**: Applies convolution using filters.

### 3\. Recurrent Layer

[Recurrent layers](https://www.geeksforgeeks.org/recurrent-layers-in-tensorflow/) are used in Recurrent Neural Networks (RNNs) for sequence data like time series or natural language. They have connections that loop back, allowing information to persist across time steps. This makes them suitable for tasks where context and temporal dependencies are important.

- **Role**: Processes sequential data with temporal dependencies.
- **Function**: Maintains state across time steps.

### 4\. Dropout Layer

[**Dropout layers**](https://www.geeksforgeeks.org/dropout-in-neural-networks/) are a regularization technique used to prevent overfitting. They randomly drop a fraction of the neurons during training, which forces the network to learn more robust features and reduces dependency on specific neurons. During training, each neuron is retained with a probability p.

- **Role**: Prevents overfitting.
- **Function**: Randomly drops neurons during training.

### **5\. Pooling Layer**

[**Pooling Layer**](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/) is used to reduce the spatial dimensions of the data, thereby decreasing the computational load and controlling overfitting. Common types of pooling include Max Pooling and Average Pooling.

**Use Cases:** Dimensionality reduction in CNNs

### **6\. Batch Normalization Layer**

A [**Batch Normalization Layer**](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/) normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. This helps in accelerating the training process and improving the performance of the network.

**Use Cases:** Stabilizing and speeding up training

Understanding the different types of layers in an ANN is essential for designing effective neural networks. Each layer has a specific role, from receiving input data to learning complex patterns and producing predictions. By combining these layers, we can build powerful models capable of solving a wide range of tasks.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/numba-vs-cython-a-technical-comparison/)

[Numba vs. Cython: A Technical Comparison](https://www.geeksforgeeks.org/numba-vs-cython-a-technical-comparison/)

[I](https://www.geeksforgeeks.org/user/indrasingh52/)

[indrasingh52](https://www.geeksforgeeks.org/user/indrasingh52/)

Follow

Improve

Article Tags :

- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

### Similar Reads

[Building Artificial Neural Networks (ANN) from Scratch\\
\\
\\
Artificial Neural Networks (ANNs)are a collection of interconnected layers of neurons. Key components of an ANN include: Input Layer: Receives input features.Hidden Layers: Process information through weighted connections and activation functions.Output Layer: Produces the final prediction.Weights a\\
\\
6 min read](https://www.geeksforgeeks.org/building-artificial-neural-networks-ann-from-scratch/)
[Artificial Neural Network in TensorFlow\\
\\
\\
Artificial Neural Networks (ANNs) compose layers of nodes (neurons), where each node processes information and passes it to the next layer. TensorFlow, an open-source machine learning framework developed by Google, provides a powerful environment for implementing and training ANNs. Layers in Artific\\
\\
5 min read](https://www.geeksforgeeks.org/artificial-neural-network-in-tensorflow/)
[Introduction to Artificial Neural Networks \| Set 1\\
\\
\\
Artificial Neural Networks (ANNs) are computational models inspired by the human brain. They are widely used for solving complex tasks such as pattern recognition, speech processing, and decision-making. By mimicking the interconnected structure of biological neurons, ANNs can learn patterns and mak\\
\\
5 min read](https://www.geeksforgeeks.org/introduction-to-artificial-neutral-networks/)
[Implementing Models of Artificial Neural Network\\
\\
\\
1\. McCulloch-Pitts Model of Neuron The McCulloch-Pitts neural model, which was the earliest ANN model, has only two types of inputs â€” Excitatory and Inhibitory. The excitatory inputs have weights of positive magnitude and the inhibitory weights have weights of negative magnitude. The inputs of the M\\
\\
7 min read](https://www.geeksforgeeks.org/implementing-models-of-artificial-neural-network/)
[Introduction to Artificial Neural Network \| Set 2\\
\\
\\
Artificial Neural Networks contain artificial neurons which are called units. These units are arranged in a series of layers that together constitute the whole Artificial Neural Network in a system. This article provides the outline for understanding the Artificial Neural Network. Characteristics of\\
\\
3 min read](https://www.geeksforgeeks.org/introduction-artificial-neural-network-set-2/)
[Machine Learning vs Neural Networks\\
\\
\\
Neural Networks and Machine Learning are two terms closely related to each other; however, they are not the same thing, and they are also different in terms of the level of AI. Artificial intelligence, on the other hand, is the ability of a computer system to display intelligence and most importantl\\
\\
12 min read](https://www.geeksforgeeks.org/machine-learning-vs-neural-networks/)
[Semantic Networks in Artificial Intelligence\\
\\
\\
Semantic networks are a powerful tool in the field of artificial intelligence (AI), used to represent knowledge and understand relationships between different concepts. They are graphical representations that connect nodes (representing concepts) with edges (representing relationships). Semantic net\\
\\
10 min read](https://www.geeksforgeeks.org/semantic-networks-in-artificial-intelligence/)
[Deep Neural Network With L - Layers\\
\\
\\
This article aims to implement a deep neural network with an arbitrary number of hidden layers each containing different numbers of neurons. We will be implementing this neural net using a few helper functions and at last, we will combine these functions to make the L-layer neural network model.L -\\
\\
11 min read](https://www.geeksforgeeks.org/deep-neural-network-with-l-layers/)
[Convolutional Neural Networks (CNNs) in R\\
\\
\\
Convolutional Neural Networks (CNNs) are a specialized type of neural network designed to process and analyze visual data. They are particularly effective for tasks involving image recognition and classification due to their ability to automatically and adaptively learn spatial hierarchies of featur\\
\\
11 min read](https://www.geeksforgeeks.org/convolutional-neural-networks-cnns-in-r/)
[Architecture and Learning process in neural network\\
\\
\\
In order to learn about Backpropagation, we first have to understand the architecture of the neural network and then the learning process in ANN. So, let's start about knowing the various architectures of the ANN: Architectures of Neural Network: ANN is a computational system consisting of many inte\\
\\
9 min read](https://www.geeksforgeeks.org/ml-architecture-and-learning-process-in-neural-network/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/layers-in-artificial-neural-networks-ann/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=298771983.1745056728&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026~103130498~103130500&z=111714011)

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