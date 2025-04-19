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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/what-is-forward-propagation-in-neural-networks/?type%3Darticle%26id%3D1224452&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Keras Input Layer\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/keras-input-layer/)

# What is Forward Propagation in Neural Networks?

Last Updated : 13 Apr, 2025

Comments

Improve

Suggest changes

6 Likes

Like

Report

Forward propagation is the fundamental process in a neural network where input data passes through multiple layers to generate an output. It is the process by which input data passes through each layer of neural network to generate output. In this article, we’ll more about forward propagation and see how it's implemented in practice.

## Understanding Forward Propogation

In **Forward propagation** input data moves through each layer of neural network where each neuron applies weighted sum, adds bias, passes the result through an [activation function](https://www.geeksforgeeks.org/activation-functions-neural-networks/) and making predictions. This process is crucial before [backpropagation](https://www.geeksforgeeks.org/backpropagation-in-neural-network/) updates the weights. It determines the output of neural network with a given set of inputs and current state of model parameters (weights and biases). Understanding this process helps in optimizing neural networks for various tasks like classification, regression and more. Below is the step by step working of forward propagation:

### 1\. **Input Layer**

- The input data is fed into the network through the input layer.
- Each feature in the input dataset represents a neuron in this layer.
- The input is usually normalized or standardized to improve model performance.

### 2\. **Hidden Layers**

- The input moves through one or more hidden layers where transformations occur.
- Each neuron in hidden layer computes a weighted sum of inputs and applies activation function to introduce non-linearity.
- Each neuron receives inputs, computes: Z=WX+bZ= W X + bZ=WX+b , where:
  - WWW is the weight matrix
  - XXX is the input vector
  - bbb is the bias term
- The activation function such as ReLU or sigmoid is applied.

### 3\. **Output Layer**

- The last layer in the network generates the final prediction.
- The activation function of this layer depends on the type of problem:
  - **Softmax** (for multi-class classification)
  - **Sigmoid** (for binary classification)
  - **Linear** (for regression tasks)

### 4\. **Prediction**

- The network produces an output based on current weights and biases.
- The loss function evaluates the error by comparing predicted output with actual values.

## Mathematical Explanation of Forward Propagation

Consider a neural network with one input layer, two hidden layers and one output layer.

![fpnn](https://media.geeksforgeeks.org/wp-content/uploads/20250411124740004542/fpnn.webp)architecture of a neural network

### **1\. Layer 1 (First Hidden Layer)**

The transformation is: A\[1\]=σ(W\[1\]X+b\[1\])A^{\[1\]} = \\sigma(W^{\[1\]}X + b^{\[1\]})A\[1\]=σ(W\[1\]X+b\[1\]) where:

- W\[1\]W^{\[1\]}W\[1\] is the weight matrix,
- XXX is the input vector,
- b\[1\]b^{\[1\]}b\[1\]is the bias vector,
- σ\\sigmaσ is the activation function.

### **2\. Layer 2 (Second Hidden Layer)**

A\[2\]=σ(W\[2\]A\[1\]+b\[2\])A^{\[2\]} = \\sigma(W^{\[2\]}A^{\[1\]} + b^{\[2\]})A\[2\]=σ(W\[2\]A\[1\]+b\[2\])

### **3\. Output Layer**

Y=σ(W\[3\]A\[2\]+b\[3\])Y = \\sigma(W^{\[3\]}A^{\[2\]} + b^{\[3\]})Y=σ(W\[3\]A\[2\]+b\[3\]) where YYY is the final output. Thus the complete equation for forward propagation is:

A\[3\]=σ(σ(σ(XW\[1\]+b\[1\])W\[2\]+b\[2\])W\[3\]+b\[3\])A^{\[3\]} = \\sigma(\\sigma(\\sigma(X W^{\[1\]} + b^{\[1\]}) W^{\[2\]} + b^{\[2\]}) W^{\[3\]} + b^{\[3\]})A\[3\]=σ(σ(σ(XW\[1\]+b\[1\])W\[2\]+b\[2\])W\[3\]+b\[3\])

This equation illustrates how data flows through the network:

- Weights (WWW) determine the importance of each input
- Biases (bbb) adjust activation thresholds
- Activation functions (σ\\sigmaσ) introduce non-linearity to enable complex decision boundaries.

## Implementation of Forward Propagation

### **1\. Import Required Libraries**

Here we will import [Numpy](https://www.geeksforgeeks.org/numpy-in-python-set-1-introduction/) and [pandas](https://www.geeksforgeeks.org/introduction-to-pandas-in-python/) library.

Python`
import numpy as np
import pandas as pd
`

### **2\. Create Sample Dataset**

- The dataset consists of CGPA, profile score and salary in LPA.
- XXX contains only input features.

Python`
data = {'cgpa': [8.5, 9.2, 7.8], 'profile_score': [85, 92, 78], 'lpa': [10, 12, 8]}
df = pd.DataFrame(data)
X = df[['cgpa', 'profile_score']].values
`

### **3\. Initialize Parameters**

When initilaizing parameters **Random initialization** avoids symmetry issues where neurons learn the same function.

Python`
def initialize_parameters():
    np.random.seed(1)
    W = np.random.randn(2, 1) * 0.01
    b = np.zeros((1, 1))
    return W, b
`

### **4\. Define Forward Propagation**

- Z=WX+BZ=WX+BZ=WX+B computes the linear transformation.
- Sigmoid activation ensures values remain between 0 and 1.

Python`
def forward_propagation(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
`

### **5\. Execute Forward Propagation**

Here we will execute the process of forward propagation using the above functions we created.

Python`
W, b = initialize_parameters()
A = forward_propagation(X, W, b)
print("Final Output:", A)
`

**Output:**

> Final Output:
>
> \[\[0.40566303\]\
>\
> \[0.39810287\]\
>\
> \[0.41326819\]\]

Each number represents the **model's predicted probability** before training for the given input. The values represent the **sigmoid activation output** which ranges between 0 and 1 indicating a probability like score for classification. Understanding forward propagation is crucial for building and optimizing deep learning models as it forms the basis for making predictions before weight adjustments occur during backpropagation

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/keras-input-layer/)

[Keras Input Layer](https://www.geeksforgeeks.org/keras-input-layer/)

[![author](https://media.geeksforgeeks.org/auth/profile/w7lvohlsdzcs4ymqlw6n)](https://www.geeksforgeeks.org/user/anshumanm2fja/)

[anshumanm2fja](https://www.geeksforgeeks.org/user/anshumanm2fja/)

Follow

6

Improve

Article Tags :

- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

### Similar Reads

[Feedforward Neural Networks (FNNs) in R\\
\\
\\
Feedforward Neural Networks (FNNs) are a type of artificial neural network where connections between nodes do not form a cycle. This means that data moves in one directionâ€”forwardâ€”from the input layer through the hidden layers to the output layer. These networks are often used for tasks such as clas\\
\\
6 min read](https://www.geeksforgeeks.org/feedforward-neural-networks-fnns-in-r/)
[What is a Neural Network?\\
\\
\\
Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making. In this article, we will explore the fundament\\
\\
14 min read](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)
[What is a Projection Layer in the Context of Neural Networks?\\
\\
\\
A projection layer in neural networks refers to a layer that transforms input data into a different space, typically either higher or lower-dimensional, depending on the design and goals of the neural network. This transformation is generally linear and is often achieved using a fully connected laye\\
\\
5 min read](https://www.geeksforgeeks.org/what-is-a-projection-layer-in-the-context-of-neural-networks/)
[Optimization Rule in Deep Neural Networks\\
\\
\\
In machine learning, optimizers and loss functions are two fundamental components that help improve a modelâ€™s performance. A loss function evaluates a model's effectiveness by computing the difference between expected and actual outputs. Common loss functions include log loss, hinge loss, and mean s\\
\\
5 min read](https://www.geeksforgeeks.org/optimization-rule-in-deep-neural-networks/)
[What are radial basis function neural networks?\\
\\
\\
Radial Basis Function (RBF) Neural Networks are a specialized type of Artificial Neural Network (ANN) used primarily for function approximation tasks. Known for their distinct three-layer architecture and universal approximation capabilities, RBF Networks offer faster learning speeds and efficient p\\
\\
8 min read](https://www.geeksforgeeks.org/what-are-radial-basis-function-neural-networks/)
[What is Dynamic Neural Network?\\
\\
\\
Dynamic Neural Networks are the upgraded version of Static Neural Networks. They have better decision algorithms and can generate better-quality results. The decision algorithm refers to the improvements to the network. It is responsible for making the right decisions accurately and with the right a\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-dynamic-neural-network/)
[Multi Layered Neural Networks in R Programming\\
\\
\\
A series or set of algorithms that try to recognize the underlying relationship in a data set through a definite process that mimics the operation of the human brain is known as a Neural Network. Hence, the neural networks could refer to the neurons of the human, either artificial or organic in natu\\
\\
6 min read](https://www.geeksforgeeks.org/multi-layered-neural-networks-in-r-programming/)
[Weights and Bias in Neural Networks\\
\\
\\
Machine learning, with its ever-expanding applications in various domains, has revolutionized the way we approach complex problems and make data-driven decisions. At the heart of this transformative technology lies neural networks, computational models inspired by the human brain's architecture. Neu\\
\\
13 min read](https://www.geeksforgeeks.org/the-role-of-weights-and-bias-in-neural-networks/)
[Optimization in Neural Networks and Newton's Method\\
\\
\\
In machine learning, optimizers and loss functions are two components that help improve the performance of the model. A loss function measures the performance of a model by measuring the difference between the output expected from the model and the actual output obtained from the model. Mean square\\
\\
12 min read](https://www.geeksforgeeks.org/optimization-in-neural-networks-and-newtons-method/)
[What is Perceptron \| The Simplest Artificial neural network\\
\\
\\
The Perceptron is one of the simplest artificial neural network architectures, introduced by Frank Rosenblatt in 1957. It is primarily used for binary classification. At that time, traditional methods like Statistical Machine Learning and Conventional Programming were commonly used for predictions.\\
\\
13 min read](https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/)

Like6

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/what-is-forward-propagation-in-neural-networks/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1514610390.1745056900&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102015666~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=748909714)