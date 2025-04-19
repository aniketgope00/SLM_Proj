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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/conditional-generative-adversarial-network/?type%3Darticle%26id%3D1090830&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Top 10 Tableau Project Ideas For Data Science\[2025\]\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/tableau-project-ideas-for-data-science/)

# Conditional Generative Adversarial Network

Last Updated : 27 Feb, 2025

Comments

Improve

Suggest changes

2 Likes

Like

Report

Imagine you have to generate images of cats that match your ideal vision or create landscapes in a specific artistic style. CGANs make this possible by generating data based on specific conditions such as class labels or descriptions. _**A Conditional Generative Adversarial Network (CGAN) is an advanced type of**_ [_**GAN (Generative Adversarial Network)**_](https://www.geeksforgeeks.org/generative-adversarial-network-gan/) _**where output generation is controlled using a condition.**_

For example if you want to generate only Mercedes cars from a dataset of various cars CGANs allow you to specify "Mercedes" as a condition. The generation of data in a CGAN is conditional on specific input information which could be labels, class information or any other relevant features. This conditioning enables more precise and targeted data generation.

## Architecture and Working of CGANs

CGANs are an enhanced version of GANs where both the generator and discriminator are conditioned on some additional information. This helps guide the generation process making it more controlled and targeted.

**1\. Generator in CGANs**: The Generator is the part of the CGAN that creates synthetic data like images, text or videos. It takes two key inputs:

- **Random Noise (z)**: This is a vector of random values that provides variety to the generated data.
- **Conditioning Information (y)**: This is extra data e.g., labels or additional context that controls what the generator creates. It could be a class label like "cat" or "dog", or any other relevant data.

The generator combines both random noise (z) and conditioning information (y) to generate realistic data. For example when y is 'cat' it generates an image of a cat.

**2\. Discriminator in CGANs**: The Discriminator is a binary classifier that tries to figure out if the data is real or fake. It takes two inputs:

- **Real Data (x)**: This is actual data sampled from the true dataset.
- **Conditioning Information (y)**: The same extra information ( **y**) that was given to the generator.

The discriminator uses both real data (x) and conditioning information (y) to distinguish between real and fake data. Like if the input data is a "real cat" image the discriminator checks whether the image is genuinely a cat.

**3\. How Generator and Discriminator Work Together**: The Generator and Discriminator are trained simultaneously in a process called adversarial training. Here’s how the interaction works:

- The **Generator** creates fake data based on noise ( **z**) and extra information ( **y**).
- The **Discriminator** then checks if the data is real or fake using the same conditioning variable ( **y**).

The goal of the adversarial process is:

- **Generator**: To create data that **fools** the discriminator into thinking it's real.
- **Discriminator**: To correctly distinguish between real and fake data.

**4\. Loss Function and Training in CGANs:** The Loss Function for CGANs is key to how the system learns:

- The Generator aims to minimize its ability to be caught by the discriminator meaning it wants to create **more realistic** data.
- The Discriminator aims to maximize its ability to detect fake data improving its accuracy in distinguishing real vs. fake.

The system improves over time as both the generator and discriminator get better at their tasks. The loss function for CGANs is expressed as:

minGmaxDV(D,G)=Ex∼pdata(x)\[logD(x∣y)\]+Ez∼pz(z)\[log(1−D(G(z∣y)))\] min\_G max\_D V(D,G) = \\mathbb{E}\_{x \\sim p\_{data} (x)}\[logD(x\|y)\] + \\mathbb{E}\_{z \\sim p\_{z}}(z)\[log(1- D(G(z∣y)))\]minG​maxD​V(D,G)=Ex∼pdata​(x)​\[logD(x∣y)\]+Ez∼pz​​(z)\[log(1−D(G(z∣y)))\]

Here,

- E\\mathbb{E}

E represents the expected operator. It is used to denote the expected value of a random variable. In this context, Ex∼pdata\\mathbb{E}\_{x \\sim p\_{data}}

Ex∼pdata​​represents the expected value with respect to the real data distribution pdata(x)p\_{data} (x)

pdata​(x), and Ez∼pz(z) \\mathbb{E}\_{z \\sim p\_{z}}(z)

Ez∼pz​​(z) represents the expected value with respect to the prior noise distribution pz(z)p\_z (z)

pz​(z).
- The objective is to simultaneously minimize the generator's ability to fool the discriminator and maximize the discriminator's ability to correctly classify real and generated samples.
- The first term (logD(x∣y))(logD(x∣y))

(logD(x∣y)) encourages the discriminator to correctly classify real samples.
- The second term (log(1−D(G(z∣y))))(log(1−D(G(z∣y))))

(log(1−D(G(z∣y)))) encourages the generator to produce samples that are classified as real by the discriminator.

Over time as both the generator and discriminator improve this adversarial process leads to more realistic data generated by the CGAN.

![Conditional-GANs](https://media.geeksforgeeks.org/wp-content/uploads/20231117113724/Conditional-GANs.png)Conditional Generative Adversarial Network

### Implementing CGAN on CiFAR-10

Let us see the working of CGAN on CIFAR-10 Dataset. Follow along these steps to have a better understanding about how the CGAN model works.

#### Import libraries

We will start with importing the necessary libraries.

Python`
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from keras.preprocessing import image
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
`

#### Loading the data and variable declaring

After this we will first load the dataset and then declare some necessary global variables for the CGAN modelling. (that includes variables like epoch counts, image size, batch size, etc.)

Python`
batch_size = 16
epoch_count = 50
noise_dim = 100
n_class = 10
tags = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
img_size = 32
(X_train, y_train), (_, _) = cifar10.load_data()
X_train = (X_train - 127.5) / 127.5
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
`

#### Visualizing the data

Now we will visualize the images from the dataset loaded in tf.data.Dataset .

Python`
plt.figure(figsize=(2,2))
idx = np.random.randint(0,len(X_train))
img = image.array_to_img(X_train[idx], scale=True)
plt.imshow(img)
plt.axis('off')
plt.title(tags[y_train[idx][0]])
plt.show()
`

**Output**:

![cat](https://media.geeksforgeeks.org/wp-content/uploads/20231028010214/cat.png)

#### Defining Loss function and Optimizers

In the next step we need to define the Loss function and optimizer for the discriminator and generator networks in a Conditional Generative Adversarial Network(CGANS).

- Binary Cross-Entropy Loss (bce loss) is suitable for distinguishing between real and fake data in GANs.
- The discriminator loss function take two arguments, real and fake.
- The binary entropy calculates two losses:
1. real\_loss: the loss when the discriminator tries to classify real data as real
2. fake\_loss : the loss when the discriminator tries to classify fake data as fake
- The total loss is the sum of real\_loss and fake\_loss which represents how well the discriminator is at distinguishing between real and fake data
- The generator\_loss function calculates the bce loss for the generator. The aim of the generator is discriminate real and fake data.
- d\_optimizer and g\_optimizer are used to update the trainable parameters of the discriminator and generator during training. Adam optimizer is used to update the trainable parameters.

Python`
bce_loss = tf.keras.losses.BinaryCrossentropy()
def discriminator_loss(real, fake):
    real_loss = bce_loss(tf.ones_like(real), real)
    fake_loss = bce_loss(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(preds):
    return bce_loss(tf.ones_like(preds), preds)

d_optimizer=Adam(learning_rate=0.0002, beta_1 = 0.5)
g_optimizer=Adam(learning_rate=0.0002, beta_1 = 0.5)
`

#### Building the Generator Model

Now let us begin with building the generator model. The generator takes a label and noise as input and generates data based on the label. Since we are giving a condition i.e., our label we will use an embedding layer to change each label into a vector representation of size 50. And after building the model we will check the architecture of the model.

- the input layer is used to provide the label as input to the generators
- the embedding layer converts the label i.e., single value into vector representation of size 50
- the input layer is used to provide the noise (latent space input) to the generator. The latent space input goes through a series of dense layers with large number of nodes and [LeakyRelu](https://www.geeksforgeeks.org/activation-functions-in-neural-networks-set2/) activation function.
- the label are reshaped and the concatenated with the processed latent space.
- the merged data goes to a series of convolution transpose layers:
  - the first 'Conv2DTranspose' layer doubles the spatial size to ' 16x16x128' and LeakyReLU activation is applied.
  - the second 'Conv2DTranspose' layer double the spatial size to '32x32x128', and LeakyReLU activation is applied.
- The final output layer convolutional layer with 3 channels (for RGB color), using a kernel size (8,8) and activation function 'tanh'. it produces an image with size '32x32x3' as the desired output.

Python`
def build_generator():
# label input
    in_label = tf.keras.layers.Input(shape=(1,))
    # create an embedding layer for all the 10 classes in the form of a vector
    # of size 50
    li = tf.keras.layers.Embedding(n_class, 50)(in_label)
    n_nodes = 8 * 8
    li = tf.keras.layers.Dense(n_nodes)(li)
    # reshape the layer
    li = tf.keras.layers.Reshape((8, 8, 1))(li)
    # image generator input
    in_lat = tf.keras.layers.Input(shape=(noise_dim,))
    n_nodes = 128 * 8 * 8
    gen = tf.keras.layers.Dense(n_nodes)(in_lat)
    gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = tf.keras.layers.Reshape((8, 8, 128))(gen)
    # merge image gen and label input
    merge = tf.keras.layers.Concatenate()([gen, li])
    gen = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(merge)  # 16x16x128
    gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
    gen = tf.keras.layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(gen)  # 32x32x128
    gen = tf.keras.layers.LeakyReLU(alpha=0.2)(gen)
    out_layer = tf.keras.layers.Conv2D(
        3, (8, 8), activation='tanh', padding='same')(gen)  # 32x32x3
    model = Model([in_lat, in_label], out_layer)
    return model
g_model = build_generator()
g_model.summary()
`

**Output**:

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_2 (InputLayer)        [(None, 100)]                0         []

 input_1 (InputLayer)        [(None, 1)]                  0         []

 dense_1 (Dense)             (None, 8192)                 827392    ['input_2[0][0]']

 embedding (Embedding)       (None, 1, 50)                500       ['input_1[0][0]']

 leaky_re_lu (LeakyReLU)     (None, 8192)                 0         ['dense_1[0][0]']

 dense (Dense)               (None, 1, 64)                3264      ['embedding[0][0]']

 reshape_1 (Reshape)         (None, 8, 8, 128)            0         ['leaky_re_lu[0][0]']

 reshape (Reshape)           (None, 8, 8, 1)              0         ['dense[0][0]']

 concatenate (Concatenate)   (None, 8, 8, 129)            0         ['reshape_1[0][0]',\
                                                                     'reshape[0][0]']

 conv2d_transpose (Conv2DTr  (None, 16, 16, 128)          264320    ['concatenate[0][0]']
 anspose)

 leaky_re_lu_1 (LeakyReLU)   (None, 16, 16, 128)          0         ['conv2d_transpose[0][0]']

 conv2d_transpose_1 (Conv2D  (None, 32, 32, 128)          262272    ['leaky_re_lu_1[0][0]']
 Transpose)

 leaky_re_lu_2 (LeakyReLU)   (None, 32, 32, 128)          0         ['conv2d_transpose_1[0][0]']

 conv2d (Conv2D)             (None, 32, 32, 3)            24579     ['leaky_re_lu_2[0][0]']

==================================================================================================
Total params: 1382327 (5.27 MB)
Trainable params: 1382327 (5.27 MB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________

```

Now we will do the same for Discriminator model as well.

We have defined the discriminator model for CGAN the discriminator takes both an image and a label as input and is responsible for distinguishing between and generated data.

- Input layer takes single value which represents a label or class information. It's embedded into a 50-dimensional vector using an embedded layer
- The label input is embedded into vector and embedding allows the discriminator to take into account the class information during the discrimination process
- The label is reshaped then concatenated with the input. The combination of label information with the image is a way to provide conditional information to the discriminator
- Two convolutional layers are applied to concatenated input. LeakyReLU activation is used with an alpha value of 0.2 after each convolutional layer. These layers extract the features.
- The feature maps are flattened and dropout layer is applied to prevent overfitting. The flattened and dropout processed features are connected to the dense layer with a single neuron and a sigmoid activation function. The output is the probability score representing the input data is real or fake.
- A model is created using TensorFlow API, considering both image and label as input and producing the discriminator's output as the final layer.

Python`
def build_discriminator():

# label input
in_label = tf.keras.layers.Input(shape=(1,))
#This vector of size 50 will be learnt by the discriminator
li = tf.keras.layers.Embedding(n_class, 50)(in_label)

n_nodes = img_size * img_size
li = tf.keras.layers.Dense(n_nodes)(li)

li = tf.keras.layers.Reshape((img_size, img_size, 1))(li)
# image input
in_image = tf.keras.layers.Input(shape=(img_size, img_size, 3))

merge = tf.keras.layers.Concatenate()([in_image, li])

#We will combine input label with input image and supply as inputs to the model.
fe = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
fe = tf.keras.layers.LeakyReLU(alpha=0.2)(fe)

fe = tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
fe = tf.keras.layers.LeakyReLU(alpha=0.2)(fe)

fe = tf.keras.layers.Flatten()(fe)

fe = tf.keras.layers.Dropout(0.4)(fe)

out_layer = tf.keras.layers.Dense(1, activation='sigmoid')(fe)
# define model the model.
model = Model([in_image, in_label], out_layer)

return model
d_model = build_discriminator()
d_model.summary()
`

**Output**:

```
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_3 (InputLayer)        [(None, 1)]                  0         []

 embedding_1 (Embedding)     (None, 1, 50)                500       ['input_3[0][0]']

 dense_2 (Dense)             (None, 1, 1024)              52224     ['embedding_1[0][0]']

 input_4 (InputLayer)        [(None, 32, 32, 3)]          0         []

 reshape_2 (Reshape)         (None, 32, 32, 1)            0         ['dense_2[0][0]']

 concatenate_1 (Concatenate  (None, 32, 32, 4)            0         ['input_4[0][0]',\
 )                                                                   'reshape_2[0][0]']

 conv2d_1 (Conv2D)           (None, 16, 16, 128)          4736      ['concatenate_1[0][0]']

 leaky_re_lu_3 (LeakyReLU)   (None, 16, 16, 128)          0         ['conv2d_1[0][0]']

 conv2d_2 (Conv2D)           (None, 8, 8, 128)            147584    ['leaky_re_lu_3[0][0]']

 leaky_re_lu_4 (LeakyReLU)   (None, 8, 8, 128)            0         ['conv2d_2[0][0]']

 flatten (Flatten)           (None, 8192)                 0         ['leaky_re_lu_4[0][0]']

 dropout (Dropout)           (None, 8192)                 0         ['flatten[0][0]']

 dense_3 (Dense)             (None, 1)                    8193      ['dropout[0][0]']

==================================================================================================
Total params: 213237 (832.96 KB)
Trainable params: 213237 (832.96 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________

```

Now we will create a train step function for training our GAN model together using Gradient Tape. Gradient Tape allows use to use custom loss functions, update weights or not and also helps in training faster.

The code provided below defines a complete training step for a GAN, where the generator and discriminator are updated alternately. The **tf.function** makes sure the training step can be executed efficiently in a TensorFlow graph.

Python`
# Compiles the train_step function into a callable TensorFlow graph
@tf.function
def train_step(dataset):

    real_images, real_labels = dataset
    # Sample random points in the latent space and concatenate the labels.
    random_latent_vectors = tf.random.normal(shape=(batch_size, noise_dim))
    generated_images = g_model([random_latent_vectors, real_labels])
    # Train the discriminator.
    with tf.GradientTape() as tape:
        pred_fake = d_model([generated_images, real_labels])
        pred_real = d_model([real_images, real_labels])

        d_loss = discriminator_loss(pred_real, pred_fake)

    grads = tape.gradient(d_loss, d_model.trainable_variables)
    # print(grads)
    d_optimizer.apply_gradients(zip(grads, d_model.trainable_variables))
    #-----------------------------------------------------------------#

    # Sample random points in the latent space.
    random_latent_vectors = tf.random.normal(shape=(batch_size, noise_dim))

    # Train the generator
    with tf.GradientTape() as tape:
        fake_images = g_model([random_latent_vectors, real_labels])
        predictions = d_model([fake_images, real_labels])
        g_loss = generator_loss(predictions)

    grads = tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, g_model.trainable_variables))

    return d_loss, g_loss
`

Also we will create a helper code for visualizing the output after each epoch ends for each class. The examples of images generated for each class. demonstrating how well the generator can produce images conditioned on specific labels or classes.

Python`
# Helper function to plot generated images
def show_samples(num_samples, n_class, g_model):
    fig, axes = plt.subplots(10,num_samples, figsize=(10,20))
    fig.tight_layout()
    fig.subplots_adjust(wspace=None, hspace=0.2)
    for l in np.arange(10):
      random_noise = tf.random.normal(shape=(num_samples, noise_dim))
      label = tf.ones(num_samples)*l
      gen_imgs = g_model.predict([random_noise, label])
      for j in range(gen_imgs.shape[0]):
        img = image.array_to_img(gen_imgs[j], scale=True)
        axes[l,j].imshow(img)
        axes[l,j].yaxis.set_ticks([])
        axes[l,j].xaxis.set_ticks([])
        if j ==0:
          axes[l,j].set_ylabel(tags[l])
    plt.show()
`

#### Training the model and Visualize the output

At the final step, we will start training the model.

Python`
def train(dataset, epochs=epoch_count):
    for epoch in range(epochs):
        print('Epoch: ', epochs)
        d_loss_list = []
        g_loss_list = []
        q_loss_list = []
        start = time.time()

        itern = 0
        for image_batch in tqdm(dataset):
            d_loss, g_loss = train_step(image_batch)
            d_loss_list.append(d_loss)
            g_loss_list.append(g_loss)
            itern=itern+1

        show_samples(3, n_class, g_model)

        print (f'Epoch: {epoch} -- Generator Loss: {np.mean(g_loss_list)}, Discriminator Loss: {np.mean(d_loss_list)}\n')
        print (f'Took {time.time()-start} seconds. \n\n')


train(dataset, epochs=epoch_count)
`

**Output**:

![clasess](https://media.geeksforgeeks.org/wp-content/uploads/20231028014517/clasess.jpg)

We can see some details in these pictures. For better result we can try to run this for more epochs.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/tableau-project-ideas-for-data-science/)

[Top 10 Tableau Project Ideas For Data Science\[2025\]](https://www.geeksforgeeks.org/tableau-project-ideas-for-data-science/)

[C](https://www.geeksforgeeks.org/user/chikubhaiya322/)

[chikubhaiya322](https://www.geeksforgeeks.org/user/chikubhaiya322/)

Follow

2

Improve

Article Tags :

- [Geeks Premier League](https://www.geeksforgeeks.org/category/geeksforgeeks-initiatives/geeks-premier-league/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Deep-Learning](https://www.geeksforgeeks.org/tag/deep-learning/)
- [Geeks Premier League 2023](https://www.geeksforgeeks.org/tag/geeks-premier-league-2023/)

+1 More

### Similar Reads

[Generative Adversarial Network (GAN)\\
\\
\\
Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow and his colleagues in 2014. GANs are a class of neural networks that autonomously learn patterns in the input data to generate new examples resembling the original dataset. GAN's architecture consists of two neural networks: Ge\\
\\
12 min read](https://www.geeksforgeeks.org/generative-adversarial-network-gan/?ref=ml_lbp)
[Generative Adversarial Networks (GANs) vs Diffusion Models\\
\\
\\
Generative Adversarial Networks (GANs) and Diffusion Models are powerful generative models designed to produce synthetic data that closely resembles real-world data. Each model has distinct architectures, strengths, and limitations, making them uniquely suited for various applications. This article\\
\\
5 min read](https://www.geeksforgeeks.org/generative-adversarial-networks-gans-vs-diffusion-models/?ref=ml_lbp)
[Generative Adversarial Networks (GANs) with R\\
\\
\\
Generative Adversarial Networks (GANs) are a type of neural network architecture introduced by Ian Goodfellow and his colleagues in 2014. GANs are designed to generate new data samples that resemble a given dataset. They can produce high-quality synthetic data across various domains. Working of GANs\\
\\
15 min read](https://www.geeksforgeeks.org/generative-adversarial-networks-gans-with-r/?ref=ml_lbp)
[Basics of Generative Adversarial Networks (GANs)\\
\\
\\
GANs is an approach for generative modeling using deep learning methods such as CNN (Convolutional Neural Network). Generative modeling is an unsupervised learning approach that involves automatically discovering and learning patterns in input data such that the model can be used to generate new exa\\
\\
3 min read](https://www.geeksforgeeks.org/basics-of-generative-adversarial-networks-gans/?ref=ml_lbp)
[Generative Adversarial Networks (GANs) in PyTorch\\
\\
\\
Generative Adversarial Networks (GANs) have revolutionized the field of machine learning by enabling models to generate realistic data. In this comprehensive tutorial, weâ€™ll show you how to implement GANs using the PyTorch framework. Why PyTorch for GANs?PyTorch is one of the most popular deep learn\\
\\
9 min read](https://www.geeksforgeeks.org/generative-adversarial-networks-gans-in-pytorch/?ref=ml_lbp)
[What is so special about Generative Adversarial Network (GAN)\\
\\
\\
Fans are ecstatic for a variety of reasons, including the fact that GANs were the first generative algorithms to produce convincingly good results, as well as the fact that they have opened up many new research directions. In the last several years, GANs are considered to be the most prominent machi\\
\\
5 min read](https://www.geeksforgeeks.org/what-is-so-special-about-generative-adversarial-network-gan/?ref=ml_lbp)
[Image Generation using Generative Adversarial Networks (GANs) using TensorFlow\\
\\
\\
Generative Adversarial Networks (GANs) represent a revolutionary approach to artificial intelligence particularly for generating images. Introduced in 2014 GANs have significantly advanced the ability to create realistic and high-quality images from random noise. In this article, we are going to tra\\
\\
5 min read](https://www.geeksforgeeks.org/image-generation-using-generative-adversarial-networks-gans/?ref=ml_lbp)
[Architecture of Super-Resolution Generative Adversarial Networks (SRGANs)\\
\\
\\
Super-Resolution Generative Adversarial Networks (SRGANs) are advanced deep learning models designed to upscale low-resolution images to high-resolution outputs with remarkable detail. This article aims to provide a comprehensive overview of SRGANs, focusing on their architecture, key components, an\\
\\
9 min read](https://www.geeksforgeeks.org/architecture-of-super-resolution-generative-adversarial-networks-srgans/?ref=ml_lbp)
[Conditional GANs (cGANs) for Image Generation\\
\\
\\
Traditional GANs, however, operate without any specific guidance, producing images based purely on the data they are trained on. Conditional GANs (cGANs) extend this capability by incorporating additional information to generate more targeted and specific images. This article explores the concept of\\
\\
7 min read](https://www.geeksforgeeks.org/conditional-gans-cgans-for-image-generation/?ref=ml_lbp)
[Wasserstein Generative Adversarial Networks (WGANs) Convergence and Optimization\\
\\
\\
Wasserstein Generative Adversarial Network (WGANs) is a variation of Deep Learning GAN with little modification in the algorithm. Generative Adversarial Network (GAN) is a method for constructing an efficient generative model. Martin Arjovsky, Soumith Chintala, and LÃ©on Bottou developed this network\\
\\
9 min read](https://www.geeksforgeeks.org/wasserstein-generative-adversarial-networks-wgans-convergence-and-optimization/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/conditional-generative-adversarial-network/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1058070393.1745057274&gtm=45je54h0h2v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=374009741)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057274583&cv=11&fst=1745057274583&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54h0h2v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=720&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fconditional-generative-adversarial-network%2F&hn=www.googleadservices.com&frm=0&tiba=Conditional%20Generative%20Adversarial%20Network%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=937785411.1745057275&uaa=x86&uab=64&uafvl=Chromium%3B131.0.6778.33%7CNot_A%2520Brand%3B24.0.0.0&uamb=0&uam=&uap=Windows&uapv=10.0&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=rfl29r6kowvh)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=vsb8x7g1tcqx)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=invisible&cb=9a7n33e0dgkr)