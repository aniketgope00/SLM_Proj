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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/lenet-5-architecture/?type%3Darticle%26id%3D1251330&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is LeNet?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/what-is-lenet/)

# LeNet-5 Architecture

Last Updated : 24 May, 2024

Comments

Improve

Suggest changes

Like Article

Like

Report

In the late 1990s, Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner created a convolutional neural network (CNN) based architecture called LeNet. The LeNet-5 architecture was developed to recognize handwritten and machine-printed characters, a function that showcased the potential of deep learning in practical applications. This article provides an in-depth exploration of the LeNet-5 architecture, examining each component and its contribution in deep learning.

## Introduction to LeNet-5

LeNet-5 is a [convolutional neural network (CNN)](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) architecture that introduced several key features and innovations that have become standard in modern deep learning. It demonstrated the effectiveness of CNNs for image recognition tasks and introduced key concepts such as convolution, pooling, and hierarchical feature extraction that underpin modern deep learning models.

Originally designed for [handwritten digit recognition](https://www.geeksforgeeks.org/handwritten-digit-recognition-using-neural-network/), the principles behind LeNet-5 have been extended to various applications, including:

- Handwriting recognition in postal services and banking.
- Object and face recognition in images and videos.
- Autonomous driving systems for recognizing and interpreting road signs.

## Architecture of LeNet-5

![LeNet-5 ](https://media.geeksforgeeks.org/wp-content/uploads/20240524180818/lenet-min.PNG)LeNet-5 Architecture for Digit Recognition

The architecture of LeNet 5 contains 7 layers excluding the input layer. Here is a detailed breakdown of the LeNet-5 architecture:

### 1\. Input Layer

- **Input Size**: 32x32 pixels.
- The input is larger than the largest character in the database, which is at most 20x20 pixels, centered in a 28x28 field. The larger input size ensures that distinctive features such as stroke endpoints or corners can appear in the center of the receptive field of the highest-level feature detectors.
- **Normalization**: Input pixel values are normalized such that the background (white) corresponds to a value of 0, and the foreground (black) corresponds to a value of 1. This normalization makes the mean input roughly 0 and the variance roughly 1, which accelerates the learning process.

### 2\. **Layer C1 (Convolutional Layer)**

- **Feature Maps**: 6 feature maps.
- **Connections**: Each unit is connected to a 5x5 neighborhood in the input, producing 28x28 feature maps to prevent boundary effects.
- **Parameters**: 156 trainable parameters and 117,600 connections.

### **3\. Layer S2 (Subsampling Layer)**

- **Feature Maps**: 6 feature maps.
- **Size**: 14x14 (each unit connected to a 2x2 neighborhood in C1).
- **Operation**: Each unit adds four inputs, multiplies by a trainable coefficient, adds a bias, and applies a sigmoid function.
- **Parameters**: 12 trainable parameters and 5,880 connections.

> **Partial Connectivity**: C3 is not fully connected to S2, which limits the number of connections and breaks symmetry, forcing feature maps to learn different, complementary features.

### 4\. **Layer C3 (Convolutional Layer)**

- **Feature Maps**: 16 feature maps.
- **Connections**: Each unit is connected to several 5x5 neighborhoods at identical locations in a subset of S2’s feature maps.
- **Parameters and Connections**: Connections are partially connected to force feature maps to learn different features, with 1,516 trainable parameters and 151,600 connections.

### 5\. **Layer S4 (Subsampling Layer)**

- **Feature Maps**: 16 feature maps.
- **Size**: 7x7 (each unit connected to a 2x2 neighborhood in C3).
- **Parameters**: 32 trainable parameters and 2,744 connections.

### 6\. **Layer C5 (Convolutional Layer)**

- **Feature Maps**: 120 feature maps.
- **Size**: 1x1 (each unit connected to a 5x5 neighborhood on all 16 of S4’s feature maps, effectively fully connected due to input size).
- **Parameters**: 48,000 trainable parameters and 48,000 connections.

### 7\. **Layer F6 (Fully Connected Layer)**

- **Units**: 84 units.
- **Connections**: Each unit is fully connected to C5, resulting in 10,164 trainable parameters.
- **Activation**: Uses a scaled hyperbolic tangent function f(a) = A\\tan (Sa), where A = 1.7159 and S = 2/3

### 8\. **Output Layer**

In the output layer of LeNet, each class is represented by an Euclidean Radial Basis Function (RBF) unit. Here's how the output of each RBF unit y\_iis computed:

y\_i = \\sum\_{j} x\_j . w\_{ij}​

In this equation:

- x\_j represents the inputs to the RBF unit.
- w\_{ij} represents the weights associated with each input.
- The summation is over all inputs to the RBF unit.

In essence, the output of each RBF unit is determined by the Euclidean distance between its input vector and its parameter vector. The larger the distance between the input pattern and the parameter vector, the larger the RBF output. This output can be interpreted as a penalty term measuring the fit between the input pattern and the model of the class associated with the RBF unit.

## Detailed Explanation of the Layers

- **Convolutional Layers (Cx)**: These layers apply convolution operations to the input, using multiple filters to extract different features. The filters slide over the input image, computing the dot product between the filter weights and the input pixels. This process captures spatial hierarchies of features, such as edges and textures.
- **Subsampling Layers (Sx)**: These layers perform pooling operations (average pooling in the case of LeNet-5) to reduce the spatial dimensions of the feature maps. This helps to control overfitting, reduce the computational load, and make the representation more compact.
- **Fully Connected Layers (Fx)**: These layers are densely connected, meaning each neuron in these layers is connected to every neuron in the previous layer. This allows the network to combine features learned in previous layers to make final predictions.

The overall architecture of LeNet-5, with its combination of convolutional, subsampling, and fully connected layers, was designed to be both computationally efficient and effective at capturing the hierarchical structure of handwritten digit images. The careful normalization of input values and the structured layout of receptive fields contribute to the network's ability to learn and generalize from the training data effectively.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/what-is-lenet/)

[What is LeNet?](https://www.geeksforgeeks.org/what-is-lenet/)

[A](https://www.geeksforgeeks.org/user/alka1974/)

[alka1974](https://www.geeksforgeeks.org/user/alka1974/)

Follow

Improve

Article Tags :

- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [Computer Vision](https://www.geeksforgeeks.org/category/ai-ml-ds/computer-vision/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

### Similar Reads

[Architecture of a System\\
\\
\\
Architecture is a critical aspect of designing a system, as it sets the foundation for how the system will function and be built. It is the process of making high-level decisions about the organization of a system, including the selection of hardware and software components, the design of interfaces\\
\\
4 min read](https://www.geeksforgeeks.org/architecture-of-a-system/?ref=ml_lbp)
[Xilinx FPGA Architecture\\
\\
\\
FPGA, which stands for field-programmable gate array, can be used to solve any computable issue. It is an integrated circuit that may be customized after manufacturing by a client or a designer. To be more specific, FPGAs (Field Programmable Gate Arrays) are semiconductor devices that consist of a m\\
\\
5 min read](https://www.geeksforgeeks.org/xilinx-fpga-architecture/?ref=ml_lbp)
[Shared Nothing Architecture\\
\\
\\
In modern computing, scalability and resilience are very important. As applications and data volumes continue to grow exponentially, traditional architectures struggle to meet the demands of todayâ€™s dynamic digital landscape. Enter Shared Nothing Architecture (SNA), a design paradigm that promises t\\
\\
10 min read](https://www.geeksforgeeks.org/shared-nothing-architecture/?ref=ml_lbp)
[How to Evaluate a System's Architecture?\\
\\
\\
Understanding a system's architecture involves assessing its structure, components, and interactions to ensure alignment with functional and non-functional requirements. This article outlines essential criteria and methodologies for evaluating and optimizing system architectures effectively. Importa\\
\\
10 min read](https://www.geeksforgeeks.org/how-to-evaluate-a-systems-architecture/?ref=ml_lbp)
[Layered Architecture in Computer Networks\\
\\
\\
Layered architecture in computer networks refers to dividingÂ a network's functioning into different layers, each responsible for a certain communication component. The major goal of this layered architecture is to separate the complex network communication process into manageable, smaller activities\\
\\
10 min read](https://www.geeksforgeeks.org/layered-architecture-in-computer-networks/?ref=ml_lbp)
[Difference Between x64 and x86 Architecture\\
\\
\\
x64 and x86 are phrases used to explain different laptop fashions, commonly in terms of crucial processing gadgets (CPUs) and their processors. This time period is frequently used to differentiate between 32-bit and 64-bit architectures. In this article, here we provide an explanation for the creati\\
\\
9 min read](https://www.geeksforgeeks.org/difference-between-x64-and-x86-architecture/?ref=ml_lbp)
[Microkernel Architecture Pattern - System Design\\
\\
\\
The Microkernel Architecture Pattern is a system design approach where a small, core system the microkernel manages essential functions. It allows for flexibility by letting additional features and services be added as needed. This design makes the system adaptable and easier to maintain because new\\
\\
10 min read](https://www.geeksforgeeks.org/microkernel-architecture-pattern-system-design/?ref=ml_lbp)
[Multitiered Architectures in Distributed System\\
\\
\\
Multitiered Architectures in Distributed Systems explains how complex computer systems are organized into different layers or tiers to improve performance and manageability. Each tier has a specific role, such as handling user interactions, processing data, or storing information. By dividing tasks\\
\\
11 min read](https://www.geeksforgeeks.org/multitiered-architectures-in-distributed-system/?ref=ml_lbp)
[When to Choose Which Architecture for System Design\\
\\
\\
When to Choose Which Architecture for System Design guides you on selecting the right system architecture for your projects. It explains the differences between various architectures, like monolithic, microservices, and serverless, and when to use each. The article helps you understand the pros and\\
\\
11 min read](https://www.geeksforgeeks.org/when-to-choose-which-architecture-for-system-design/?ref=ml_lbp)
[ARM Interrupt Structure\\
\\
\\
A collection of reduced instruction set computer (RISC) instruction set architectures for computer processors that are tailored for different contexts is known as ARM (stylized in lowercase as an arm; originally an abbreviation for Advanced RISC Machines. System-on-a-chip (SoC) and system-on-module\\
\\
7 min read](https://www.geeksforgeeks.org/arm-interrupt-structure/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/lenet-5-architecture/)

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