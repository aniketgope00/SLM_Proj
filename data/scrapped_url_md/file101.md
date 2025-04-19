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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/activation-functions-neural-networks/?type%3Darticle%26id%3D172507&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Implementation of K Nearest Neighbors\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)

# Activation functions in Neural Networks

Last Updated : 05 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

While building a neural network, one key decision is selecting the Activation Function for both the hidden layer and the output layer. Activation functions decide whether a neuron should be activated.

> _Before diving into the activation function, you should have prior knowledge of the following topics :_ [_**Neural Networks**_](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) _**,**_ [_**Backpropagation**_](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)

## What is an Activation Function?

An activation function is a mathematical function applied to the output of a neuron. It introduces non-linearity into the model, allowing the network to learn and represent complex patterns in the data. Without this non-linearity feature, a neural network would behave like a linear regression model, no matter how many layers it has.

Activation function decides whether a neuron should be activated by calculating the weighted sum of inputs and adding a bias term. This helps the model make complex decisions and predictions by introducing non-linearities to the output of each neuron.

## Introducing Non-Linearity in Neural Network

Non-linearity means that the relationship between input and output **is not a straight line**. In simple terms, the output **does not change proportionally** with the input. A common choice is the ReLU function, defined as σ(x)=max⁡(0,x)\\sigma(x) = \\max(0, x)σ(x)=max(0,x).

Imagine you want to classify **apples** and **bananas** based on their **shape and color**.

- If we use a **linear function**, it can only separate them using a **straight line**.
- But real-world data is often more complex (e.g., overlapping colors, different lighting).
- By adding a **non-linear activation function** (like **ReLU, Sigmoid, or Tanh**), the network can create **curved decision boundaries** to separate them correctly.

### Effect of Non-Linearity

The inclusion of the ReLU activation function \\sigma allows h\_1​ to introduce a non-linear decision boundary in the input space. This non-linearity enables the network to learn more complex patterns that are not possible with a purely linear model, such as:

- Modeling functions that are not linearly separable.
- Increasing the capacity of the network to form multiple decision boundaries based on the combination of weights and biases.

## Why is Non-Linearity Important in Neural Networks?

Neural networks consist of neurons that operate using **weights**, **biases**, and **activation functions**.

In the learning process, these weights and biases are updated based on the error produced at the output—a process known as **backpropagation**. Activation functions enable backpropagation by providing gradients that are essential for updating the weights and biases.

Without non-linearity, even deep networks would be limited to solving only simple, linearly separable problems. Activation functions empower neural networks to model highly complex data distributions and solve advanced deep learning tasks. Adding non-linear activation functions introduce flexibility and enable the network to learn more complex and abstract patterns from data.

## Mathematical Proof of Need of Non-Linearity in Neural Networks

To illustrate the need for non-linearity in neural networks with a specific example, let’s consider a network with two input nodes (i1and i2​)(i\_1 \\text{and } i\_2​)(i1​and i2​​), a single hidden layer containing neurons h1 and h2h\_1 \\text{ and } h\_2 h1​ and h2​, and an output neuron (out).

We will use w1,w2w\_1, w\_2w1​,w2​ as weights connecting the inputs to the hidden neuron, and w5w\_5w5​​ as the weight connecting the hidden neuron to the output. We’ll also include biases (b1b\_1b1​​ for the hidden neuron and b2b\_2b2​ for the output neuron) to complete the model.

1. **Input Layer**: Two inputs i1i\_1i1​ and i2​i\_2​i2​​.
2. **Hidden Layer**: Two neuron h1h\_1h1​​ and h2h\_2h2​
3. **Output Layer**: One output neuron.

![NeuralNetwok](https://media.geeksforgeeks.org/wp-content/uploads/20240503121617/NeuralNetwok.png)

The input to the hidden neuron h1h\_1h1​ is calculated as a weighted sum of the inputs plus a bias:

h1=i1.w1+i2.w3+b1{h\_1} = i\_1.w\_1 + i\_2.w\_3 + b\_1 h1​=i1​.w1​+i2​.w3​+b1​

h2=i1.w2+i2.w4+b2{h\_2} = i\_1.w\_2 + i\_2.w\_4 + b\_2 h2​=i1​.w2​+i2​.w4​+b2​

The output neuron is then a weighted sum of the hidden neuron’s output plus a bias:

output=h1.w5+h2.w6+bias \\text{output} = h\_1.w\_5 + h\_2.w\_6 + \\text{bias}output=h1​.w5​+h2​.w6​+bias

Here, h\_1 , h\_2 \\text{ and output} are linear expressions.

In order to add non-linearity, we will be using **sigmoid activation function** in the output layer:

σ(x)=11+e−x\\sigma(x) = \\frac{1}{1+e^{-x}}σ(x)=1+e−x1​

final output=σ(h1.w5+h2.w6+bias)\\text{final output} = \\sigma(h\_1.w\_5 + h\_2.w\_6 + \\text{bias})final output=σ(h1​.w5​+h2​.w6​+bias)

final output=11+e−(h1.w5+h2.w6+bias)\\text{final output} = \\frac{1}{1+e^{-(h\_1.w\_5+h\_2.w\_6 + \\text{bias})}}final output=1+e−(h1​.w5​+h2​.w6​+bias)1​

This gives the final output of the network after applying the sigmoid activation function in output layers, introducing the desired non-linearity.

## Types of Activation Functions in Deep Learning

### **1\. Linear Activation Function**

**Linear Activation Function** resembles straight line define by y=x. No matter how many layers the neural network contains, if they all use linear activation functions, the output is a linear combination of the input.

- The range of the output spans from (−∞ to +∞)(-\\infty \\text{ to } + \\infty)(−∞ to +∞).
- **Linear activation function** is used at just one place i.e. output layer.
- Using linear activation across all layers makes the network’s ability to learn complex patterns limited.

Linear activation functions are useful for specific tasks but must be combined with non-linear functions to enhance the neural network’s learning and predictive capabilities.

![Linear-Activation-Function](https://media.geeksforgeeks.org/wp-content/uploads/20241029115212560858/Linear-Activation-Function.png)

Linear Activation Function or Identity Function returns the input as the output

### 2\. Non-Linear Activation Functions

**1\. Sigmoid Function**

[**Sigmoid Activation Function**](https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/) is characterized by ‘S’ shape. It is mathematically defined asA=11+e−x A = \\frac{1}{1 + e^{-x}}A=1+e−x1​​. This formula ensures a smooth and continuous output that is essential for gradient-based optimization methods.

- It allows neural networks to handle and model complex patterns that linear equations cannot.
- The output ranges between 0 and 1, hence useful for binary classification.
- The function exhibits a steep gradient when x values are between -2 and 2. This sensitivity means that small changes in input x can cause significant changes in output y, which is critical during the training process.

![Sigmoid-Activation-Function](https://media.geeksforgeeks.org/wp-content/uploads/20241029120537926197/Sigmoid-Activation-Function.png)

Sigmoid or Logistic Activation Function Graph

**2\. Tanh Activation Function**

[**Tanh function**](https://www.geeksforgeeks.org/tanh-activation-in-neural-network/) **(hyperbolic tangent function),** is a shifted version of the sigmoid, allowing it to stretch across the y-axis. It is defined as:

f(x)=tanh⁡(x)=21+e−2x–1. f(x) = \\tanh(x) = \\frac{2}{1 + e^{-2x}} – 1. f(x)=tanh(x)=1+e−2x2​–1.

Alternatively, it can be expressed using the sigmoid function:

tanh⁡(x)=2×sigmoid(2x)–1 \\tanh(x) = 2 \\times \\text{sigmoid}(2x) – 1tanh(x)=2×sigmoid(2x)–1

- **Value Range**: Outputs values from -1 to +1.
- **Non-linear**: Enables modeling of complex data patterns.
- **Use in Hidden Layers**: Commonly used in hidden layers due to its zero-centered output, facilitating easier learning for subsequent layers.

![Tanh-Activation-Function](https://media.geeksforgeeks.org/wp-content/uploads/20241029120618881107/Tanh-Activation-Function.png)

Tanh Activation Function

**3\. ReLU** (Rectified Linear Unit) **Function**

[**ReLU activation**](https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/) is defined by A(x)=max⁡(0,x)A(x) = \\max(0,x)A(x)=max(0,x), this means that if the input x is positive, ReLU returns x, if the input is negative, it returns 0.

- **Value Range**: \[0,∞)\[0, \\infty)\[0,∞), meaning the function only outputs non-negative values.\
- **Nature**: It is a **non-linear** activation function, allowing neural networks to learn complex patterns and making backpropagation more efficient.\
- **Advantage over other Activation:** ReLU is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. At a time only a few neurons are activated making the network sparse making it efficient and easy for computation.\
\
![relu-activation-function](https://media.geeksforgeeks.org/wp-content/uploads/20241029120652402777/relu-activation-function.png)\
\
ReLU Activation Function\
\
### 3\. **Exponential Linear Units**\
\
**1\. Softmax Function**\
\
[**Softmax function**](https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) is designed to handle multi-class classification problems. It transforms raw output scores from a neural network into probabilities. It works by squashing the output values of each class into the range of 0 to 1, while ensuring that the sum of all probabilities equals 1.\
\
- Softmax is a **non-linear** activation function.\
- The Softmax function ensures that each class is assigned a probability, helping to identify which class the input belongs to.\
\
![softmax](https://media.geeksforgeeks.org/wp-content/uploads/20241029120724445438/softmax.png)\
\
Softmax Activation Function\
\
**2\. SoftPlus Function**\
\
[**Softplus function**](https://www.geeksforgeeks.org/softplus-function-in-neural-network/) is defined mathematically as: A(x)=log⁡(1+ex)A(x) = \\log(1 + e^x)A(x)=log(1+ex).\
\
This equation ensures that the output is always positive and differentiable at all points, which is an advantage over the traditional ReLU function.\
\
- **Nature**: The Softplus function is **non-linear**.\
- **Range**: The function outputs values in the range (0,∞)(0, \\infty)(0,∞), similar to ReLU, but without the hard zero threshold that ReLU has.\
- **Smoothness**: Softplus is a smooth, continuous function, meaning it avoids the sharp discontinuities of ReLU, which can sometimes lead to problems during optimization.\
\
![softplus](https://media.geeksforgeeks.org/wp-content/uploads/20241029120803898518/softplus.png)\
\
Softplus Activation Function\
\
## Impact of Activation Functions on Model Performance\
\
The choice of activation function has a direct impact on the performance of a neural network in several ways:\
\
1. **Convergence Speed:** Functions like **ReLU** allow faster training by avoiding the vanishing gradient problem, while **Sigmoid** and **Tanh** can slow down convergence in deep networks.\
2. **Gradient Flow:** Activation functions like **ReLU** ensure better gradient flow, helping deeper layers learn effectively. In contrast, **Sigmoid** can lead to small gradients, hindering learning in deep layers.\
3. **Model Complexity:** Activation functions like **Softmax** allow the model to handle complex multi-class problems, whereas simpler functions like **ReLU** or **Leaky ReLU** are used for basic layers.\
\
Activation functions are the backbone of neural networks, enabling them to capture non-linear relationships in data. From classic functions like Sigmoid and Tanh to modern variants like ReLU and Swish, each has its place in different types of neural networks. The key is to understand their behavior and choose the right one based on your model’s needs.\
\
[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)\
\
Comment\
\
\
More info\
\
[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)\
\
[Next Article](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)\
\
[Implementation of K Nearest Neighbors](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)\
\
[S](https://www.geeksforgeeks.org/user/Sakshi_Tiwari/)\
\
[Sakshi\_Tiwari](https://www.geeksforgeeks.org/user/Sakshi_Tiwari/)\
\
Follow\
\
Improve\
\
Article Tags :\
\
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)\
- [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/)\
\
Practice Tags :\
\
- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)\
\
### Similar Reads\
\
[Activation functions in Neural Networks \| Set2\\
\\
\\
The article Activation-functions-neural-networks will help to understand the use of activation function along with the explanation of some of its variants like linear, sigmoid, tanh, Relu and softmax. There are some other variants of the activation function like Elu, Selu, Leaky Relu, Softsign and S\\
\\
3 min read](https://www.geeksforgeeks.org/activation-functions-in-neural-networks-set2/)\
[Auto-associative Neural Networks\\
\\
\\
Auto associative Neural networks are the types of neural networks whose input and output vectors are identical. These are special kinds of neural networks that are used to simulate and explore the associative process. Association in this architecture comes from the instruction of a set of simple pro\\
\\
3 min read](https://www.geeksforgeeks.org/auto-associative-neural-networks/)\
[Types Of Activation Function in ANN\\
\\
\\
The biological neural network has been modeled in the form of Artificial Neural Networks with artificial neurons simulating the function of a biological neuron. The artificial neuron is depicted in the below picture: Each neuron consists of three major components:Â  A set of 'i' synapses having weigh\\
\\
4 min read](https://www.geeksforgeeks.org/types-of-activation-function-in-ann/)\
[Exploring Adaptive Filtering in Neural Networks\\
\\
\\
Adaptive filtering is a critical concept in neural networks, particularly in the context of signal processing, control systems, and error cancellation. This article delves into the adaptive filtering problem, its mathematical formulation, and the various techniques used to address it, with a focus o\\
\\
7 min read](https://www.geeksforgeeks.org/exploring-adaptive-filtering-in-neural-networks/)\
[Backpropagation in Neural Network\\
\\
\\
Backpropagation is also known as "Backward Propagation of Errors" and it is a method used to train neural network . Its goal is to reduce the difference between the modelâ€™s predicted output and the actual output by adjusting the weights and biases in the network. In this article we will explore what\\
\\
10 min read](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)\
[Introduction to Convolution Neural Network\\
\\
\\
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs are widely us\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)\
[Activation Functions\\
\\
\\
To put it in simple terms, an artificial neuron calculates the 'weighted sum' of its inputs and adds a bias, as shown in the figure below by the net input. Mathematically, Net Input=∑(Weight×Input)+Bias\\text{Net Input} =\\sum \\text{(Weight} \\times \\text{Input)+Bias}Net Input=∑(Weight×Input)+Bias Now the value of net input can be any anything from -\\
\\
3 min read](https://www.geeksforgeeks.org/activation-functions/)\
[Effect of Bias in Neural Network\\
\\
\\
Neural Network is conceptually based on actual neuron of brain. Neurons are the basic units of a large neural network. A single neuron passes single forward based on input provided. In Neural network, some inputs are provided to an artificial neuron, and with each input a weight is associated. Weigh\\
\\
3 min read](https://www.geeksforgeeks.org/effect-of-bias-in-neural-network/)\
[Dropout in Neural Networks\\
\\
\\
The concept of Neural Networks is inspired by the neurons in the human brain and scientists wanted a machine to replicate the same process. This craved a path to one of the most important topics in Artificial Intelligence. A Neural Network (NN) is based on a collection of connected units or nodes ca\\
\\
3 min read](https://www.geeksforgeeks.org/dropout-in-neural-networks/)\
[Artificial Neural Networks and its Applications\\
\\
\\
As you read this article, which organ in your body is thinking about it? It's the brain, of course! But do you know how the brain works? Well, it has neurons or nerve cells that are the primary units of both the brain and the nervous system. These neurons receive sensory input from the outside world\\
\\
9 min read](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)\
\
Like\
\
We use cookies to ensure you have the best browsing experience on our website. By using our site, you\
acknowledge that you have read and understood our\
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &\
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)\
Got It !\
\
\
![Lightbox](https://www.geeksforgeeks.org/activation-functions-neural-networks/)\
\
Improvement\
\
Suggest changes\
\
Suggest Changes\
\
Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.\
\
![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)\
\
Create Improvement\
\
Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.\
\
![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)\
\
Suggest Changes\
\
min 4 words, max Words Limit:1000\
\
## Thank You!\
\
Your suggestions are valuable to us.\
\
## What kind of Experience do you want to share?\
\
[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)\
\
[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=941849515.1745056903&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=1629325750)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056902856&cv=11&fst=1745056902856&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Factivation-functions-neural-networks%2F&hn=www.googleadservices.com&frm=0&tiba=Activation%20functions%20in%20Neural%20Networks%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1494229527.1745056903&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)\
\
[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)