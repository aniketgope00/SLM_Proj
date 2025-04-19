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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/?type%3Darticle%26id%3D1242475&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
PowerTransformer in scikit-learn\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/powertransformer-in-scikit-learn/)

# What is Batch Normalization in CNN?

Last Updated : 13 May, 2024

Comments

Improve

Suggest changes

2 Likes

Like

Report

Batch Normalization is a technique used to improve the training and performance of neural networks, particularly CNNs. The article aims to provide an overview of batch normalization in CNNs along with the implementation in PyTorch and TensorFlow.

Table of Content

- [Overview of Batch Normalization](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#overview-of-batch-normalization)
- [Need for Batch Normalization in CNN model](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#need-for-batch-normalization-in-cnn-model)
- [How Does Batch Normalization Work in CNN?](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#how-does-batch-normalization-work-in-cnn)

  - [1\. Normalization within Mini-Batch](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#1-normalization)
  - [2\. Scaling and Shifting](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#2-scaling-and-shifting)
  - [3\. Learnable Parameters](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#3-differentiability)
  - [4\. Applying Batch Normalization](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#4-applying-batch-normalization)
  - [5\. Training and Inference](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#5-training-and-inference)

- [Applying Batch Normalization in CNN model using TensorFlow](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#applying-batch-normalization-in-cnn-model-using-tensorflow)
- [Applying Batch Normalization in CNN model using PyTorch](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#applying-batch-normalization-in-cnn-model-using-pytorch)
- [Advantages of Batch Normalization in CNN](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/#advantages-of-batch-normalization-in-cnn)

## Overview of Batch Normalization

[_**Batch normalization**_](https://www.geeksforgeeks.org/what-is-batch-normalization-in-deep-learning/) is a technique to improve the training of [deep neural networks](https://www.geeksforgeeks.org/introduction-deep-learning/) by stabilizing and accelerating the learning process. Introduced by Sergey Ioffe and Christian Szegedy in 2015, it addresses the issue known as _"internal covariate shift"_ where the distribution of each layer's inputs changes during training, as the parameters of the previous layers change.

## Need for Batch Normalization in CNN model

Batch Normalization in CNN addresses several challenges encountered during training. There are following reasons highlight the need for batch normalization in CNN:

1. **Addressing Internal Covariate Shift:** Internal covariate shift occurs when the distribution of network activations changes as parameters are updated during training. Batch normalization addresses this by normalizing the activations in each layer, maintaining consistent mean and variance across inputs throughout training. This stabilizes training and speeds up convergence.
2. **Improving Gradient Flow:** Batch normalization contributes to stabilizing the gradient flow during backpropagation by reducing the reliance of gradients on parameter scales. As a result, training becomes faster and more stable, enabling effective training of deeper networks without facing issues like vanishing or exploding gradients.
3. **Regularization Effect:** During training, batch normalization introduces noise to the network activations, serving as a regularization technique. This noise aids in averting overfitting by injecting randomness and decreasing the network's sensitivity to minor fluctuations in the input data.

## How Does Batch Normalization Work in CNN?

Batch normalization works in [convolutional neural networks (CNNs)](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) by normalizing the activations of each layer across mini-batch during training. The working is discussed below:

### 1\. Normalization within Mini-Batch

In a CNN, each layer receives inputs from multiple channels (feature maps) and processes them through convolutional filters. Batch Normalization operates on each feature map separately, normalizing the activations across the mini-batch.

During training, batch normalization (BN) standardizes the activations of each layer by subtracting the mean and dividing by the standard deviation of each mini-batch.

- Mean Calculation: μB=1m∑i=1mxiμ\_B = \\frac{1}{m}\\sum\_{i=1}^{m}{x\_i}
μB​=m1​∑i=1m​xi​
- Variance Calculation: σB2=1m∑i=1m(xi−μB)2\\sigma\_{B}^{2} = \\frac{1}{m} \\sum\_{i=1}^{m}{(x\_i - \\mu\_B)^2}σB2​=m1​∑i=1m​(xi​−μB​)2
- Normalization: xi^=xi−μBσB2+ϵ\\widehat{x\_i} = \\frac{x\_i - \\mu\_B}{\\sqrt{\\sigma\_{B}^{2} + \\epsilon}}xi​​=σB2​+ϵ​xi​−μB​​

### 2\. Scaling and Shifting

After normalization, BN adjusts the normalized activations using learned scaling and shifting parameters. These parameters enable the network to adaptively scale and shift the activations, thereby maintaining the network's ability to represent complex patterns in the data.

- Scaling: γxi^\\gamma \\widehat{x\_i}γxi​​
- Shifting: zi=yi+βz\_i = y\_i + \\betazi​=yi​+β

### **3\. Learnable Parameters**

The parameters γ\\gammaγ and β\\betaβ are learned during training through backpropagation. This allows the network to adaptively adjust the normalization and ensure that the activations are in the appropriate range for learning.

### **4\. Applying Batch Normalization**

Batch Normalization is typically applied after the convolutional and activation layers in a CNN, before passing the outputs to the next layer. It can also be applied before or after the activation function, depending on the network architecture.

### **5\. Training and Inference**

During training, Batch Normalization calculates the mean and variance of each mini-batch. During inference (testing), it uses the aggregated mean and variance calculated during training to normalize the activations. This ensures consistent normalization between training and inference.

## Applying Batch Normalization in CNN model using TensorFlow

In this section, we have provided a pseudo code, to illustrate how can we apply batch normalization in CNN model using TensorFlow. For applying batch normalization layers after the convolutional layers and before the activation functions, we use _**'tf.keras.layers.BatchNormalization()'**_.

```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

# Build the CNN model
model = Sequential([\
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),\
    BatchNormalization(),  # Add batch normalization layer\
    MaxPooling2D((2, 2)),\
    Conv2D(64, (3, 3), activation='relu'),\
    BatchNormalization(),  # Add batch normalization layer\
    MaxPooling2D((2, 2)),\
    Flatten(),\
    Dense(64, activation='relu'),\
    Dense(10, activation='softmax')\
])
```

## Applying Batch Normalization in CNN model using PyTorch

In PyTorch, we can easily apply batch normalization in a CNN model.

For applying BN in 1D Convolutional Neural Network model, we use ' _**nn.BatchNorm1d()'**_.

```
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * 28, 10)  # Example fully connected layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 32 * 28)  # Reshape for fully connected layer
        x = self.fc(x)
        return x

# Instantiate the model
model = CNN1D()
```

For applying Batch Normalization in 2D Convolutional Neural Network model, we use ' _**nn.BatchNorm2d()'**_.

```
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 28 * 28, 10)  # Example fully connected layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 32 * 28 * 28)  # Reshape for fully connected layer
        x = self.fc(x)
        return x

# Instantiate the model
model = CNN()
```

> For more detailed explanation regarding the implementation, refer to
>
> - [Batch Normalization Implementation in PyTorch](https://www.geeksforgeeks.org/batch-normalization-implementation-in-pytorch/)
> - [Applying Batch Normalization in Keras using BatchNormalization Class](https://www.geeksforgeeks.org/applying-batch-normalization-in-keras-using-batchnormalization-class/)

## Advantages of Batch Normalization in CNN

- Fast Convergence
- Improved generalization
- reduced sensitivity
- Higher learning rates
- Improvement in model accuracy

## Conclusion

In conclusion, batch normalization stands as a pivotal technique in enhancing the training and performance of convolutional neural networks (CNNs). Its implementation addresses critical challenges such as internal covariate shift, thereby stabilizing training, accelerating convergence, and facilitating deeper network architectures.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/powertransformer-in-scikit-learn/)

[PowerTransformer in scikit-learn](https://www.geeksforgeeks.org/powertransformer-in-scikit-learn/)

[K](https://www.geeksforgeeks.org/user/kushagragupta209/)

[kushagragupta209](https://www.geeksforgeeks.org/user/kushagragupta209/)

Follow

2

Improve

Article Tags :

- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [Computer Vision](https://www.geeksforgeeks.org/category/ai-ml-ds/computer-vision/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+1 More

### Similar Reads

[What is Batch Normalization In Deep Learning?\\
\\
\\
Internal covariate shift is a major challenge encountered while training deep learning models. Batch normalization was introduced to address this issue. In this article, we are going to learn the fundamentals and need of Batch normalization. We are also going to perform batch normalization. What is\\
\\
5 min read](https://www.geeksforgeeks.org/what-is-batch-normalization-in-deep-learning/?ref=ml_lbp)
[What is Group Normalization?\\
\\
\\
Group Normalization (GN) is a technique introduced by Yuxin Wu and Kaiming He in 2018. It addresses some of the limitations posed by Batch Normalization, especially when dealing with small batch sizes that are common in high-resolution images or video processing tasks. Unlike Batch Normalization, wh\\
\\
4 min read](https://www.geeksforgeeks.org/what-is-group-normalization/?ref=ml_lbp)
[Batch Normalization Implementation in PyTorch\\
\\
\\
Batch Normalization (BN) is a critical technique in the training of neural networks, designed to address issues like vanishing or exploding gradients during training. In this tutorial, we will implement batch normalization using PyTorch framework. Table of Content What is Batch Normalization?How Bat\\
\\
7 min read](https://www.geeksforgeeks.org/batch-normalization-implementation-in-pytorch/?ref=ml_lbp)
[How to Effectively Use Batch Normalization in LSTM?\\
\\
\\
Batch Normalization (BN) has revolutionized the training of deep neural networks by normalizing input data across batches, stabilizing the learning process, and allowing faster convergence. While BN is widely used in feedforward neural networks, its application to recurrent neural networks (RNNs) li\\
\\
8 min read](https://www.geeksforgeeks.org/how-to-effectively-use-batch-normalization-in-lstm/?ref=ml_lbp)
[Instance Normalization vs Batch Normalization\\
\\
\\
Instance normalization and batch normalization are techniques used to make machine learning models train better by normalizing data, but they work differently. Instance normalization normalizes each input individually focusing only on its own features. This is more like giving personalized feedback\\
\\
5 min read](https://www.geeksforgeeks.org/instance-normalization-vs-batch-normalization/?ref=ml_lbp)
[Data Normalization Machine Learning\\
\\
\\
Normalization is an essential step in the preprocessing of data for machine learning models, and it is a feature scaling technique. Normalization is especially crucial for data manipulation, scaling down, or up the range of data before it is utilized for subsequent stages in the fields of soft compu\\
\\
9 min read](https://www.geeksforgeeks.org/what-is-data-normalization/?ref=ml_lbp)
[What is Zero Mean and Unit Variance Normalization\\
\\
\\
Answer: Zero Mean and Unit Variance normalization rescale data to have a mean of zero and a standard deviation of one.Explanation:Mean Centering: The first step of Zero Mean normalization involves subtracting the mean value of each feature from all data points. This centers the data around zero, mea\\
\\
2 min read](https://www.geeksforgeeks.org/what-is-zero-mean-and-unit-variance-normalization/?ref=ml_lbp)
[What is Standardization in Machine Learning\\
\\
\\
In Machine Learning we train our data to predict or classify things in such a manner that isn't hardcoded in the machine. So for the first, we have the Dataset or the input data to be pre-processed and manipulated for our desired outcomes. Any ML Model to be built follows the following procedure: Co\\
\\
6 min read](https://www.geeksforgeeks.org/what-is-standardization-in-machine-learning/?ref=ml_lbp)
[What is the use of SoftMax in CNN?\\
\\
\\
Answer: SoftMax is used in Convolutional Neural Networks (CNNs) to convert the network's final layer logits into probability distributions, ensuring that the output values represent normalized class probabilities, making it suitable for multi-class classification tasks.SoftMax is a crucial activatio\\
\\
2 min read](https://www.geeksforgeeks.org/why-should-softmax-be-used-in-cnn/?ref=ml_lbp)
[What is Padding in Neural Network?\\
\\
\\
As we know while building a neural network we are doing convolution to extract features with the help of kernels with respect to the current datasets which is the important part to make your network learn while convolving. For example, if you want to train your neural network to classify whether it\\
\\
9 min read](https://www.geeksforgeeks.org/what-is-padding-in-neural-network/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/what-is-batch-normalization-in-cnn/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=102938773.1745057170&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=445790750)

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