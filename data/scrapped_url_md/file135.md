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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/restricted-boltzmann-machine/?type%3Darticle%26id%3D498721&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Find most similar sentence in the file to the input sentence \| NLP\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/find-most-similar-sentence-in-the-file-to-the-input-sentence-nlp/)

# Restricted Boltzmann Machine

Last Updated : 18 Mar, 2023

Comments

Improve

Suggest changes

Like Article

Like

Report

**Introduction :**

Restricted Boltzmann Machine (RBM) is a type of artificial neural network that is used for unsupervised learning. It is a type of generative model that is capable of learning a probability distribution over a set of input data.

RBM was introduced in the mid-2000s by Hinton and Salakhutdinov as a way to address the problem of unsupervised learning. It is a type of neural network that consists of two layers of neurons – a visible layer and a hidden layer. The visible layer represents the input data, while the hidden layer represents a set of features that are learned by the network.

The RBM is called “restricted” because the connections between the neurons in the same layer are not allowed. In other words, each neuron in the visible layer is only connected to neurons in the hidden layer, and vice versa. This allows the RBM to learn a compressed representation of the input data by reducing the dimensionality of the input.

The RBM is trained using a process called contrastive divergence, which is a variant of the stochastic gradient descent algorithm. During training, the network adjusts the weights of the connections between the neurons in order to maximize the likelihood of the training data. Once the RBM is trained, it can be used to generate new samples from the learned probability distribution.

RBM has found applications in a wide range of fields, including computer vision, natural language processing, and speech recognition. It has also been used in combination with other neural network architectures, such as deep belief networks and deep neural networks, to improve their performance.

**What are Boltzmann Machines?**

It is a network of neurons in which all the neurons are connected to each other. In this machine, there are two layers named visible layer or input layer and hidden layer. The visible layer is denoted as **v** and the hidden layer is denoted as the **h.** In Boltzmann machine, there is no output layer. Boltzmann machines are random and generative neural networks capable of learning internal representations and are able to represent and (given enough time) solve tough combinatoric problems.

The Boltzmann distribution (also known as **Gibbs Distribution**) which is an integral part of Statistical Mechanics and also explain the impact of parameters like Entropy and Temperature on the Quantum States in Thermodynamics. Due to this, it is also known as **Energy-Based Models (EBM)**. It was invented in 1985 by Geoffrey Hinton, then a Professor at Carnegie Mellon University, and Terry Sejnowski, then a Professor at Johns Hopkins University

![](https://media.geeksforgeeks.org/wp-content/uploads/20200927214842/Boltzmann-294x300.jpg)

**What are Restricted Boltzmann Machines (RBM)?**

A restricted term refers to that we are not allowed to connect the same type layer to each other. In other words, the two neurons of the input layer or hidden layer can’t connect to each other. Although the hidden layer and visible layer can be connected to each other.

As in this machine, there is no output layer so the question arises how we are going to identify, adjust the weights and how to measure the that our prediction is accurate or not. All the questions have one answer, that is Restricted Boltzmann Machine.

The RBM algorithm was proposed by Geoffrey Hinton (2007), which learns probability distribution over its sample training data inputs. It has seen wide applications in different areas of supervised/unsupervised machine learning such as feature learning, dimensionality reduction, classification, collaborative filtering, and topic modeling.

Consider the example movie rating discussed in the recommender system section.

Movies like Avengers, Avatar, and Interstellar have strong associations with the latest fantasy and science fiction factor. Based on the user rating RBM will discover latent factors that can explain the activation of movie choices. In short, RBM describes variability among correlated variables of input dataset in terms of a potentially lower number of unobserved variables.

The energy function is given by

![\mathrm{E}(\mathrm{v}, \mathrm{h})=-\mathrm{a}^{\mathrm{T}} \mathrm{v}-\mathrm{b}^{\mathrm{T}} \mathrm{h}-\mathrm{v}^{\mathrm{T}} \mathrm{Wh} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5d37d8cf57a8375d036b385a40ed6136_l3.svg)

**Applications of Restricted Boltzmann Machine**

Restricted Boltzmann Machines (RBMs) have found numerous applications in various fields, some of which are:

- **Collaborative filtering:** RBMs are widely used in collaborative filtering for recommender systems. They learn to predict user preferences based on their past behavior and recommend items that are likely to be of interest to the user.
- **Image and video processing:** RBMs can be used for image and video processing tasks such as object recognition, image denoising, and image reconstruction. They can also be used for tasks such as video segmentation and tracking.
- **Natural language processing:** RBMs can be used for natural language processing tasks such as language modeling, text classification, and sentiment analysis. They can also be used for tasks such as speech recognition and speech synthesis.
- **Bioinformatics:** RBMs have found applications in bioinformatics for tasks such as protein structure prediction, gene expression analysis, and drug discovery.
- **Financial modeling:** RBMs can be used for financial modeling tasks such as predicting stock prices, risk analysis, and portfolio optimization.
- **Anomaly detection:** RBMs can be used for anomaly detection tasks such as fraud detection in financial transactions, network intrusion detection, and medical diagnosis.
- It is used in Filtering.
- It is used in Feature Learning.
- It is used in Classification.
- It is used in Risk Detection.
- It is used in Business and Economic analysis.


**How do Restricted Boltzmann Machines work?**

In RBM there are two phases through which the entire RBM works:

**1st Phase:** In this phase, we take the input layer and using the concept of weights and biased we are going to activate the hidden layer. This process is said to be Feed Forward Pass. In Feed Forward Pass we are identifying the positive association and negative association.

Feed Forward Equation:

- **Positive Association** — When the association between the visible unit and the hidden unit is positive.
- **Negative Association** — When the association between the visible unit and the hidden unit is negative.

**2nd Phase:** As we don’t have any output layer. Instead of calculating the output layer, we are reconstructing the input layer through the activated hidden state. This process is said to be Feed Backward Pass. We are just backtracking the input layer through the activated hidden neurons. After performing this we have reconstructed Input through the activated hidden state. So, we can calculate the error and adjust weight in this way:

Feed Backward Equation:

- **Error =** Reconstructed Input Layer-Actual Input layer
- **Adjust Weight =** Input\*error\*learning rate (0.1)

After doing all the steps we get the pattern that is responsible to activate the hidden neurons. To understand how it works:

Let us consider an example in which we have some assumption that V1 visible unit activates the h1 and h2 hidden unit and V2 visible unit activates the h2 and h3 hidden. Now when any new visible unit let V5 has come into the machine and it also activates the h1 and h2 unit. So, we can back trace the hidden units easily and also identify that the characteristics of the new V5 neuron is matching with that of V1. This is because V1 also activated the same hidden unit earlier.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200927214428/RBM-277x300.jpg)

Restricted Boltzmann Machines

**Types of RBM :**

There are mainly two types of Restricted Boltzmann Machine (RBM) based on the types of variables they use:

1. **Binary RBM:** In a binary RBM, the input and hidden units are binary variables. Binary RBMs are often used in modeling binary data such as images or text.
2. **Gaussian RBM:** In a Gaussian RBM, the input and hidden units are continuous variables that follow a Gaussian distribution. Gaussian RBMs are often used in modeling continuous data such as audio signals or sensor data.

Apart from these two types, there are also variations of RBMs such as:

1. **Deep Belief Network (DBN):** A DBN is a type of generative model that consists of multiple layers of RBMs. DBNs are often used in modeling high-dimensional data such as images or videos.
2. **Convolutional RBM (CRBM):** A CRBM is a type of RBM that is designed specifically for processing images or other grid-like structures. In a CRBM, the connections between the input and hidden units are local and shared, which makes it possible to capture spatial relationships between the input units.
3. **Temporal RBM (TRBM):** A TRBM is a type of RBM that is designed for processing temporal data such as time series or video frames. In a TRBM, the hidden units are connected across time steps, which allows the network to model temporal dependencies in the data.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/find-most-similar-sentence-in-the-file-to-the-input-sentence-nlp/)

[Find most similar sentence in the file to the input sentence \| NLP](https://www.geeksforgeeks.org/find-most-similar-sentence-in-the-file-to-the-input-sentence-nlp/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Restricted Boltzmann Machine : How it works\\
\\
\\
A Restricted Boltzmann Machine (RBM), Introduced by Geoffrey Hinton and Terry Sejnowski in 1985, Since, It become foundational in unsupervised machine learning, particularly in the context of deep learning architectures. They are widely used for dimensionality reduction, classification, regression,\\
\\
5 min read](https://www.geeksforgeeks.org/restricted-boltzmann-machine-how-it-works/?ref=ml_lbp)
[Types of Boltzmann Machines\\
\\
\\
Deep Learning models are broadly classified into supervised and unsupervised models. Supervised DL models: Artificial Neural Networks (ANNs)Recurrent Neural Networks (RNNs)Convolutional Neural Networks (CNNs) Unsupervised DL models: Self Organizing Maps (SOMs)Boltzmann MachinesAutoencoders Let us le\\
\\
7 min read](https://www.geeksforgeeks.org/types-of-boltzmann-machines/?ref=ml_lbp)
[Contrastive Divergence in Restricted Boltzmann Machines\\
\\
\\
Contrastive Divergence (CD) is a fundamental technique in the realm of machine learning, particularly in the field of unsupervised learning and specifically in training Restricted Boltzmann Machines (RBMs). It serves as a crucial component in the learning process by approximating the gradient needed\\
\\
10 min read](https://www.geeksforgeeks.org/contrastive-divergence-in-restricted-boltzmann-machines/?ref=ml_lbp)
[Machine Learning with R\\
\\
\\
Machine Learning as the name suggests is the field of study that allows computers to learn and take decisions on their own i.e. without being explicitly programmed. These decisions are based on the available data that is available through experiences or instructions. It gives the computer that makes\\
\\
2 min read](https://www.geeksforgeeks.org/machine-learning-with-r/?ref=ml_lbp)
[Restricted Boltzmann Machine (RBM) with Practical Implementation\\
\\
\\
In the world of machine learning, one algorithm that has gained significant attention is the Restricted Boltzmann Machine (RBM). RBMs are powerful generative models that have been widely used for various applications, such as dimensionality reduction, feature learning, and collaborative filtering. I\\
\\
8 min read](https://www.geeksforgeeks.org/restricted-boltzmann-machine-rbm-with-practical-implementation/?ref=ml_lbp)
[Biopython - Machine Learning Overview\\
\\
\\
Machine Learning algorithms are useful in every aspect of life for analyzing data accurately. Bioinformatics can easily derive information using machine learning and without it, it is hard to analyze huge genetic information. Machine Learning algorithms are broadly classified into three parts: Super\\
\\
3 min read](https://www.geeksforgeeks.org/biopython-machine-learning-overview/?ref=ml_lbp)
[Machine Learning Tutorial\\
\\
\\
Machine learning is a subset of Artificial Intelligence (AI) that enables computers to learn from data and make predictions without being explicitly programmed. If you're new to this field, this tutorial will provide a comprehensive understanding of machine learning, its types, algorithms, tools, an\\
\\
8 min read](https://www.geeksforgeeks.org/machine-learning/?ref=ml_lbp)
[Machine Learning Models\\
\\
\\
Machine Learning models are very powerful resources that automate multiple tasks and make them more accurate and efficient. ML handles new data and scales the growing demand for technology with valuable insight. It improves the performance over time. This cutting-edge technology has various benefits\\
\\
14 min read](https://www.geeksforgeeks.org/machine-learning-models/?ref=ml_lbp)
[Supervised Machine Learning Examples\\
\\
\\
Supervised machine learning technology is a key in the world of the dramatic innovations of the modern AI. It is applied in numerous items, such as coat the email and the complicated one, self-driving carsOne of the most important tasks when it comes to supervised machine learning is making computer\\
\\
7 min read](https://www.geeksforgeeks.org/supervised-machine-learning-examples/?ref=ml_lbp)
[Steps to Build a Machine Learning Model\\
\\
\\
Machine learning models offer a powerful mechanism to extract meaningful patterns, trends, and insights from this vast pool of data, giving us the power to make better-informed decisions and appropriate actions. In this article, we will explore the Fundamentals of Machine Learning and the Steps to b\\
\\
9 min read](https://www.geeksforgeeks.org/steps-to-build-a-machine-learning-model/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/restricted-boltzmann-machine/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1009480509.1745057254&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1109051783)