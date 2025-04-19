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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/variational-autoencoders/?type%3Darticle%26id%3D450399&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Contractive Autoencoder (CAE)\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/contractive-autoencoder-cae/)

# Variational AutoEncoders

Last Updated : 04 Mar, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Variational Autoencoders (VAEs) are generative models in machine learning (ML) that create new data similar to the input they are trained on. Along with data generation they also perform common autoencoder tasks like **denoising.** Like all autoencoders VAEs consist of:

- **Encoder:** Learns important patterns (latent variables) from input data.
- **Decoder:** It uses those latent variables to reconstruct the input.

Unlike traditional [autoencoders](https://www.geeksforgeeks.org/auto-encoders/) that encode afixedrepresentation VAEs learn a continuous probabilistic representation of latent space. This allows them to reconstruct input data accurately and generate new data samples that resemble the original input.

## Architecture of Variational Autoencoder

VAE is a special kind of autoencoder that can generate new data instead of just compressing and reconstructing it. It has three main parts:

### **1\. Encoder (Understanding the Input)**

- The encoder takes the input data like an image or text and tries to understand its most important features.
- Instead of creating a fixed compressed version like a normal autoencoder it creates two things:
  - **Mean (μ):** A central value representing the data.
  - **Standard Deviation (σ):** It is a measure of how much the values can vary.
- These two values define a range of possibilities instead of a single number.

### **2\. Latent Space (Adding Some Randomness)**

- Instead of encoding input into a fixed number VAEs introduce randomness to create variations.
- The model picks a point from the range to create different variations of the data.
- This is what makes VAEs great for generating new slightly different but realistic data.

### **3\. Decoder (Reconstructing or Creating New Data)**

- The **decoder** takes this sampled value and tries to reconstruct the original input.
- Since the encoder creates a range of possibilities instead of a fixed number the decoder can generate new similar data instead of just memorizing the input.

![Variational-AutoEncoder](https://media.geeksforgeeks.org/wp-content/uploads/20231201153426/Variational-AutoEncoder.png)

Variational Autoencoder

### **Mathematics behind Variational Autoencoder**

Variational autoencoder uses KL-divergence as its loss function the goal of this is to minimize the difference between a supposed distribution and original distribution of dataset.

Suppose we have a distribution z and we want to generate the observation x from it.  In other words we want to calculate p(z∣x)p\\left( {z\|x} \\right) p(z∣x)

We can do it by following way:

p(z∣x)=p(x∣z)p(z)p(x)p\\left( {z\|x} \\right) = \\frac{{p\\left( {x\|z} \\right)p\\left( z \\right)}}{{p\\left( x \\right)}} p(z∣x)=p(x)p(x∣z)p(z)​

But, the calculation of p(x) can be quite difficult:

p(x)=∫p(x∣z)p(z)dzp\\left( x \\right) = \\int {p\\left( {x\|z} \\right)p\\left(z\\right)dz} p(x)=∫p(x∣z)p(z)dz

This usually makes it an intractable distribution. Hence, we need to approximate p(z\|x) to q(z\|x) to make it a tractable distribution. To better approximate p(z\|x) to q(z\|x), we will minimize the KL-divergence loss which calculates how similar two distributions are:

min⁡KL(q(z∣x)∣∣p(z∣x))\\min KL\\left( {q\\left( {z\|x} \\right)\|\|p\\left( {z\|x} \\right)} \\right) minKL(q(z∣x)∣∣p(z∣x))

By simplifying, the above minimization problem is equivalent to the following maximization problem :

Eq(z∣x)log⁡p(x∣z)–KL(q(z∣x)∣∣p(z)){E\_{q\\left( {z\|x} \\right)}}\\log p\\left( {x\|z} \\right) – KL\\left( {q\\left( {z\|x} \\right)\|\|p\\left( z \\right)} \\right) Eq(z∣x)​logp(x∣z)–KL(q(z∣x)∣∣p(z))

The first term represents the reconstruction likelihood and the other term ensures that our learned distribution q is similar to the true prior distribution p.

Thus our total loss consists of two terms one is reconstruction error and other is KL-divergence loss:

Loss=L(x,x^)+∑jKL(qj(z∣x)∣∣p(z))Loss = L\\left( {x, \\hat x} \\right) + \\sum\\limits\_j {KL\\left( {{q\_j}\\left( {z\|x} \\right)\|\|p\\left( z \\right)} \\right)} Loss=L(x,x^)+j∑​KL(qj​(z∣x)∣∣p(z))

## Implementing Variational Autoencoder

In this implementation we will be using the Fashion-MNIST dataset this dataset is already available in **keras.datasets** API so we don’t need to add or upload manually. You can also find the implementation in the from an.

### Step 1: Importing Libraries

In this first we need to import the necessary packages to our python environment. we will be using Keras package with TensorFlow as a backend.

python`
import numpy as np
import tensorflow as tf
import keras
from keras import layers
`

### Step 2: Creating a Sampling Layer

For Variational autoencoders we need to define the architecture of two parts encoder and decoder but first we will define the bottleneck layer of architecture the sampling layer.

python`
class Sampling(layers.Layer):
    """Uses (mean, log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
`

### Step 3: Define Encoder Block

Now we define the architecture of encoder part of our autoencoder this part takes images as input and encodes their representation in the Sampling layer.

Python`
latent_dim = 2
encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
mean = layers.Dense(latent_dim, name="mean")(x)
log_var = layers.Dense(latent_dim, name="log_var")(x)
z = Sampling()([mean, log_var])
encoder = keras.Model(encoder_inputs, [mean, log_var, z], name="encoder")
encoder.summary()
`

**Output:**

![encoder](https://media.geeksforgeeks.org/wp-content/uploads/20250304155021501999/encoder.jpg)

### Step 4: Define Decoder Block

Now we define the architecture of decoder part of our autoencoder this part takes the output of the sampling layer as input and output an image of size (28, 28, 1) .

python`
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
`

**Output:**

![decoder](https://media.geeksforgeeks.org/wp-content/uploads/20250304155138331250/decoder.PNG)

### Step 5: Define the VAE Model

In this step we combine the model and define the training procedure with loss functions.

python`
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    @property
    def metrics(self):
        return [\
            self.total_loss_tracker,\
            self.reconstruction_loss_tracker,\
            self.kl_loss_tracker,\
        ]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean,log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
`

### Step 6: Train the VAE

Now it’s the right time to train our Variational autoencoder model we will train it for 10 epochs.  But first we need to import the fashion MNIST dataset.

python`
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
fashion_mnist = np.concatenate([x_train, x_test], axis=0)
fashion_mnist = np.expand_dims(fashion_mnist, -1).astype("float32") / 255
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(fashion_mnist, epochs=10, batch_size=128)
`

**Output:**

![training-output](https://media.geeksforgeeks.org/wp-content/uploads/20250304161226504321/training-output.PNG)

### Step 7: Display Sampled Images

In this step we display training result and we will be displaying these results according to their values in latent space vectors.

python`
import matplotlib.pyplot as plt
def plot_latent_space(vae, n=10, figsize=5):

    img_size = 28
    scale = 0.5
    figure = np.zeros((img_size * n, img_size * n))

    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(sample, verbose=0)
            images = x_decoded[0].reshape(img_size, img_size)
            figure[\
                i * img_size : (i + 1) * img_size,\
                j * img_size : (j + 1) * img_size,\
            ] = images
    plt.figure(figsize=(figsize, figsize))
    start_range = img_size // 2
    end_range = n * img_size + start_range
    pixel_range = np.arange(start_range, end_range, img_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
plot_latent_space(vae)
`

**Output:**

![Image-Display](https://media.geeksforgeeks.org/wp-content/uploads/20231128171148/Image-Display.png)

### Step 8: Display Latent Space Clusters

To get a more clear view of our representational latent vectors values we will be plotting the scatter plot of training data on the basis of their values of corresponding latent dimensions generated from the encoder .

python`
def plot_label_clusters(encoder, decoder, data, test_lab):
	z_mean, _, _ = encoder.predict(data)
	plt.figure(figsize =(12, 10))
	sc = plt.scatter(z_mean[:, 0], z_mean[:, 1], c = test_lab)
	cbar = plt.colorbar(sc, ticks = range(10))
	cbar.ax.set_yticklabels([labels.get(i) for i in range(10)])
	plt.xlabel("z[0]")
	plt.ylabel("z[1]")
	plt.show()
labels = {0 :"T-shirt / top",
1: "Trouser",
2: "Pullover",
3: "Dress",
4: "Coat",
5: "Sandal",
6: "Shirt",
7: "Sneaker",
8: "Bag",
9: "Ankle boot"}
(x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
plot_label_clusters(encoder, decoder, x_train, y_train)
`

**Output:**

![Latent-Space](https://media.geeksforgeeks.org/wp-content/uploads/20231128171646/Latent-Space.png)

#### What is better GANs or VAE?

> For image generation GANs is a better option as it generates high quality samples and VAE is a better option to use in signal analysis.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/contractive-autoencoder-cae/)

[Contractive Autoencoder (CAE)](https://www.geeksforgeeks.org/contractive-autoencoder-cae/)

[P](https://www.geeksforgeeks.org/user/pawangfg/)

[pawangfg](https://www.geeksforgeeks.org/user/pawangfg/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
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


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/variational-autoencoders/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1376413560.1745057329&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=1728632107)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)