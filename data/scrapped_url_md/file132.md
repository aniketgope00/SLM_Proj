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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-getting-started-with-alexnet/?type%3Darticle%26id%3D387423&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Python \| Customer Churn Analysis Prediction\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/)

# ML \| Getting Started With AlexNet

Last Updated : 26 Mar, 2020

Comments

Improve

Suggest changes

2 Likes

Like

Report

This article is focused on providing an introduction to the AlexNet architecture. Its name comes from one of the leading authors of the AlexNet [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)– Alex Krizhevsky. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 with a top-5 error rate of **_15.3%_** (beating the runner up which had a top-5 error rate of _26.2%_).

The most important features of the AlexNet paper are:

- As the model had to train _60 million_ parameters (which is quite a lot), it was prone to overfitting. According to the paper, the usage of Dropout and Data Augmentation significantly helped in reducing overfitting. The first and second fully connected layers in the architecture thus used a dropout of _0.5_ for the purpose. Artificially increasing the number of images through data augmentation helped in the expansion of the dataset dynamically during runtime, which helped the model generalize better.
- Another distinct factor was using the ReLU activation function instead of tanh or sigmoid, which resulted in faster training times (a decrease in training time by _6 times_). Deep Learning Networks usually employ ReLU non-linearity to achieve faster training times as the others start saturating when they hit higher activation values.

### The Architecture

The architecture consists of 5 Convolutional layers, with the 1st, 2nd and 5th having Max-Pooling layers for proper feature extraction. The Max-Pooling layers are _overlapped_ having strides of _2_ with filter size _3×3_. This resulted in decreasing the _top-1 and top-5_ error rates by **_0.4%_** and **_0.3%_** respectively in comparison to non-overlapped Max-Pooling layers. They are followed by _2_ fully-connected layers (each with dropout) and a softmax layer at the end for predictions.

The figure below shows the architecture of AlexNet with all the layers defined.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200315210219/AlexNet.png)

**Code: Python code to implement AlexNet for object classification**

|     |
| --- |
| `model ` `=` `Sequential() `<br>` `<br>`# 1st Convolutional Layer `<br>`model.add(Conv2D(filters ` `=` `96` `, input_shape ` `=` `(` `224` `, ` `224` `, ` `3` `),  `<br>`            ``kernel_size ` `=` `(` `11` `, ` `11` `), strides ` `=` `(` `4` `, ` `4` `),  `<br>`            ``padding ` `=` `'valid'` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Max-Pooling  `<br>`model.add(MaxPooling2D(pool_size ` `=` `(` `2` `, ` `2` `), `<br>`            ``strides ` `=` `(` `2` `, ` `2` `), padding ` `=` `'valid'` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# 2nd Convolutional Layer `<br>`model.add(Conv2D(filters ` `=` `256` `, kernel_size ` `=` `(` `11` `, ` `11` `),  `<br>`            ``strides ` `=` `(` `1` `, ` `1` `), padding ` `=` `'valid'` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Max-Pooling `<br>`model.add(MaxPooling2D(pool_size ` `=` `(` `2` `, ` `2` `), strides ` `=` `(` `2` `, ` `2` `),  `<br>`            ``padding ` `=` `'valid'` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# 3rd Convolutional Layer `<br>`model.add(Conv2D(filters ` `=` `384` `, kernel_size ` `=` `(` `3` `, ` `3` `),  `<br>`            ``strides ` `=` `(` `1` `, ` `1` `), padding ` `=` `'valid'` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# 4th Convolutional Layer `<br>`model.add(Conv2D(filters ` `=` `384` `, kernel_size ` `=` `(` `3` `, ` `3` `),  `<br>`            ``strides ` `=` `(` `1` `, ` `1` `), padding ` `=` `'valid'` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# 5th Convolutional Layer `<br>`model.add(Conv2D(filters ` `=` `256` `, kernel_size ` `=` `(` `3` `, ` `3` `),  `<br>`            ``strides ` `=` `(` `1` `, ` `1` `), padding ` `=` `'valid'` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Max-Pooling `<br>`model.add(MaxPooling2D(pool_size ` `=` `(` `2` `, ` `2` `), strides ` `=` `(` `2` `, ` `2` `),  `<br>`            ``padding ` `=` `'valid'` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# Flattening `<br>`model.add(Flatten()) `<br>` `<br>`# 1st Dense Layer `<br>`model.add(Dense(` `4096` `, input_shape ` `=` `(` `224` `*` `224` `*` `3` `, ))) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Add Dropout to prevent overfitting `<br>`model.add(Dropout(` `0.4` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# 2nd Dense Layer `<br>`model.add(Dense(` `4096` `)) `<br>`model.add(Activation(` `'relu'` `)) `<br>`# Add Dropout `<br>`model.add(Dropout(` `0.4` `)) `<br>`# Batch Normalisation `<br>`model.add(BatchNormalization()) `<br>` `<br>`# Output Softmax Layer `<br>`model.add(Dense(num_classes)) `<br>`model.add(Activation(` `'softmax'` `)) ` |

```

```

```

```

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/)

[Python \| Customer Churn Analysis Prediction](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/)

[J](https://www.geeksforgeeks.org/user/JaideepSinghSandhu/)

[JaideepSinghSandhu](https://www.geeksforgeeks.org/user/JaideepSinghSandhu/)

Follow

2

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Getting started with Kaggle : A quick guide for beginners\\
\\
\\
Kaggle is an online community of Data Scientists and Machine Learning Engineers which is owned by Google. A general feeling of beginners in the field of Machine Learning and Data Science towards the website is of hesitance. This feeling mainly arises because of the misconceptions that the outside pe\\
\\
3 min read](https://www.geeksforgeeks.org/getting-started-with-kaggle-a-quick-guide-for-beginners/?ref=ml_lbp)
[Train a Deep Learning Model With Pytorch\\
\\
\\
Neural Network is a type of machine learning model inspired by the structure and function of human brain. It consists of layers of interconnected nodes called neurons which process and transmit information. Neural networks are particularly well-suited for tasks such as image and speech recognition,\\
\\
6 min read](https://www.geeksforgeeks.org/train-a-deep-learning-model-with-pytorch/?ref=ml_lbp)
[Integrating a ML Model with React and Flask\\
\\
\\
Here, we have a task of how to integrate a machine learning model with React, and Flask. In this article, we will see how to integrate a machine-learning model with React and Flask. What is the Car Prediction Model?A car price prediction model is designed to predict selling prices based on specific\\
\\
6 min read](https://www.geeksforgeeks.org/integrating-a-ml-model-with-react-and-flask/?ref=ml_lbp)
[Image Recognition with Mobilenet\\
\\
\\
Introduction: Image Recognition plays an important role in many fields like medical disease analysis, and many more. In this article, we will mainly focus on how to Recognize the given image, what is being displayed. We are assuming to have a pre-knowledge of Tensorflow, Keras, Python, MachineLearni\\
\\
5 min read](https://www.geeksforgeeks.org/image-recognition-with-mobilenet/?ref=ml_lbp)
[Machine Learning Model with Teachable Machine\\
\\
\\
Teachable Machine is a web-based tool developed by Google that allows users to train their own machine learning models without any coding experience. It uses a web camera to gather images or videos, and then uses those images to train a machine learning model. The user can then use the model to clas\\
\\
7 min read](https://www.geeksforgeeks.org/machine-learning-model-with-teachable-machine/?ref=ml_lbp)
[A beginner's guide to supervised learning with Python\\
\\
\\
Supervised learning is a foundational concept, and Python provides a robust ecosystem to explore and implement these powerful algorithms. Explore the fundamentals of supervised learning with Python in this beginner's guide. Learn the basics, build your first model, and dive into the world of predict\\
\\
10 min read](https://www.geeksforgeeks.org/a-beginners-guide-to-supervised-learning-with-python/?ref=ml_lbp)
[Reinforcement Learning with TensorFlow Agents\\
\\
\\
Reinforcement learning (RL) represents a dynamic and powerful approach within machine learning, focusing on how agents should take actions in an environment to maximize cumulative rewards. TensorFlow Agents (TF-Agents) is a versatile and user-friendly library designed to streamline the process of de\\
\\
6 min read](https://www.geeksforgeeks.org/reinforcement-learning-with-tensorflow-agents/?ref=ml_lbp)
[Getting started with Machine Learning \|\| Machine Learning Roadmap\\
\\
\\
Machine Learning (ML) represents a branch of artificial intelligence (AI) focused on enabling systems to learn from data, uncover patterns, and autonomously make decisions. In today's era dominated by data, ML is transforming industries ranging from healthcare to finance, offering robust tools for p\\
\\
11 min read](https://www.geeksforgeeks.org/getting-started-machine-learning/?ref=ml_lbp)
[Deep Learning with Python OpenCV\\
\\
\\
Opencv 3.3 brought with a very improved and efficient (dnn) module which makes it very for you to use deep learning with OpenCV. You still cannot train models in OpenCV, and they probably don't have any intention of doing anything like that, but now you can very easily use image processing and use t\\
\\
5 min read](https://www.geeksforgeeks.org/deep-learning-with-python-opencv/?ref=ml_lbp)
[Keeping the eye on Keras models with CodeMonitor\\
\\
\\
If you work with deep learning, you probably have faced a situation where your model takes too long to learn, and you have to keep watching its progress, however staying in front of the desk watching it learn is not exactly the most exciting thing ever, for this reason, this article's purpose is to\\
\\
4 min read](https://www.geeksforgeeks.org/keeping-the-eye-on-keras-models-with-codemonitor/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-getting-started-with-alexnet/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=932786303.1745057245&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=230910826)