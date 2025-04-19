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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/k-means-clustering-introduction/?type%3Darticle%26id%3D142154&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Image Segmentation using K Means Clustering\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)

# K means Clustering – Introduction

Last Updated : 15 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

[K-Means Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/) is an [Unsupervised Machine Learning](https://www.geeksforgeeks.org/ml-types-learning-part-2/) algorithm which groups the unlabeled dataset into different clusters. The article aims to explore the fundamentals and working of k means clustering along with its implementation.

## Understanding K-means Clustering

K-means clustering is a technique used to organize data into **groups based on their similarity**. For example **online store uses K-Means to group customers based on purchase frequency and spending creating segments like Budget Shoppers, Frequent Buyers and Big Spenders for personalised marketing.**

The algorithm works by first randomly picking some central points called **centroids** and each data point is then assigned to the closest centroid forming a cluster. After all the points are assigned to a cluster the centroids are updated by finding the average position of the points in each cluster. This process repeats until the centroids stop changing forming clusters. The goal of clustering is to divide the data points into clusters so that similar data points belong to same group.

## How k-means clustering works?

We are given a data set of items with certain features and values for these features (like a vector). The task is to categorize those items into groups. To achieve this, we will use the K-means algorithm. ‘K’ in the name of the algorithm represents the number of groups/clusters we want to classify our items into.

![k_means_clustering](https://media.geeksforgeeks.org/wp-content/uploads/20250114084120247428/k_means_clustering.webp)

K means Clustering

The algorithm will categorize the items into k groups or clusters of similarity. To calculate that similarity, we will use the [Euclidean distance](https://www.geeksforgeeks.org/euclidean-distance/) as a measurement. The algorithm works as follows:

1. First, we randomly initialize k points, called means or cluster centroids.
2. We categorize each item to its closest mean, and we update the mean’s coordinates, which are the averages of the items categorized in that cluster so far.
3. We repeat the process for a given number of iterations and at the end, we have our clusters.

The “points” mentioned above are called means because they are the mean values of the items categorized in them. To initialize these means, we have a lot of options. An intuitive method is to initialize the means at random items in the data set. Another method is to initialize the means at random values between the boundaries of the data set. For example for a feature _x_ the items have values in \[0,3\] we will initialize the means with values for _x_ at \[0,3\].

## Implementation of K-Means Clustering in Python

We will use blobs datasets and show how clusters are made.

**Step 1: Importing the necessary libraries**

We are importing [Numpy](https://www.geeksforgeeks.org/numpy-in-python-set-1-introduction/) for statistical computations, [Matplotlib](https://www.geeksforgeeks.org/matplotlib-tutorial/) to plot the [graph,](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/) and make\_blobs from sklearn.datasets.

Python`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
`

#### **Step 2**: Create the custom dataset with make\_blobs and plot it

Python`
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)
fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()
`

**Output**:

![Clustering dataset - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230320171738/download-(25).png)

Clustering dataset

#### **Step 3:** Initialize the random centroids

The code initializes three clusters for K-means clustering. It sets a random seed and generates random cluster centers within a specified range, and creates an empty [list](https://www.geeksforgeeks.org/list-cpp-stl/) of points for each cluster.

Python`
k = 3
clusters = {}
np.random.seed(23)
for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }

    clusters[idx] = cluster

clusters
`

**Output:**

```
{0: {'center': array([0.06919154, 1.78785042]), 'points': []},
 1: {'center': array([ 1.06183904, -0.87041662]), 'points': []},
 2: {'center': array([-1.11581855,  0.74488834]), 'points': []}}

```

#### **Step 4:** Plot the random initialize center with data points

Python`
plt.scatter(X[:,0],X[:,1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '*',c = 'red')
plt.show()
`

**Output**:

![Data points with random center - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230320172346/download-(27).png)

Data points with random center

The plot displays a scatter plot of data points (X\[:,0\], X\[:,1\]) with grid lines. It also marks the initial cluster centers (red stars) generated for K-means clustering.

#### **Step 5:** Define Euclidean distance

Python`
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))
`

#### **Step 6:** Create the function to Assign and Update the cluster center

This step assigns data points to the nearest cluster center, and the M-step updates cluster centers based on the mean of assigned points in K-means clustering.

Python`
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []

        curr_x = X[idx]

        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center

            clusters[i]['points'] = []
    return clusters
`

#### Step 7: Create the function to Predict the cluster for the datapoints

Python`
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred
`

#### **Step 8:** Assign, Update, and predict the cluster center

Python`
clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)
`

#### **Step 9:** Plot the data points with their predicted cluster center

Python`
plt.scatter(X[:,0],X[:,1],c = pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.show()
`

**Output**:

![K-means Clustering - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230320173915/download-(28).png)

K-means Clustering

The plot shows data points colored by their predicted clusters. The red markers represent the updated cluster centers after the E-M steps in the K-means clustering algorithm.

> **Now, what is Elbow Method? : It** is a graphical tool used to determine the optimal number of clusters (k) in K-means. Selecting the right number of clusters is crucial for meaningful segmentation. Please refer to [Elbow Method for optimal value of k in KMeans](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/).

In conclusion, K-means clustering is a powerful unsupervised machine learning algorithm for grouping unlabeled datasets. Its objective is to divide data into clusters, making similar data points part of the same group. The algorithm initializes cluster centroids and iteratively assigns data points to the nearest centroid, updating centroids based on the mean of points in each cluster.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)

[Image Segmentation using K Means Clustering](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[100+ Machine Learning Projects with Source Code \[2025\]\\
\\
\\
This article provides over 100 Machine Learning projects and ideas to provide hands-on experience for both beginners and professionals. Whether you're a student enhancing your resume or a professional advancing your career these projects offer practical insights into the world of Machine Learning an\\
\\
6 min read](https://www.geeksforgeeks.org/machine-learning-projects/)

## Classification Projects

- [Wine Quality Prediction - Machine Learning\\
\\
\\
Here we will predict the quality of wine on the basis of given features. We use the wine quality dataset available on Internet for free. This dataset has the fundamental features which are responsible for affecting the quality of the wine. By the use of several Machine learning models, we will predi\\
\\
5 min read](https://www.geeksforgeeks.org/wine-quality-prediction-machine-learning/)

* * *

- [ML \| Credit Card Fraud Detection\\
\\
\\
Fraudulent credit card transactions are a significant challenge for financial institutions and consumers alike. Detecting these fraudulent activities in real-time is important to prevent financial losses and protect customers from unauthorized charges. In this article we will explore how to build a\\
\\
5 min read](https://www.geeksforgeeks.org/ml-credit-card-fraud-detection/)

* * *

- [Disease Prediction Using Machine Learning\\
\\
\\
Disease prediction using machine learning is used in healthcare to provide accurate and early diagnosis based on patient symptoms. We can build predictive models that identify diseases efficiently. In this article, we will explore the end-to-end implementation of such a system. Step 1: Import Librar\\
\\
5 min read](https://www.geeksforgeeks.org/disease-prediction-using-machine-learning/)

* * *

- [Recommendation System in Python\\
\\
\\
Industry leaders like Netflix, Amazon and Uber Eats have transformed how individuals access products and services. They do this by using recommendation algorithms that improve the user experience. These systems offer personalized recommendations based on users interests and preferences. In this arti\\
\\
7 min read](https://www.geeksforgeeks.org/recommendation-system-in-python/)

* * *

- [Detecting Spam Emails Using Tensorflow in Python\\
\\
\\
Spam messages are unsolicited or unwanted emails/messages sent in bulk to users. Detecting spam emails automatically helps prevent unnecessary clutter in users' inboxes. In this article, we will build a spam email detection model that classifies emails as Spam or Ham (Not Spam) using TensorFlow, one\\
\\
5 min read](https://www.geeksforgeeks.org/detecting-spam-emails-using-tensorflow-in-python/)

* * *

- [SMS Spam Detection using TensorFlow in Python\\
\\
\\
In today's society, practically everyone has a mobile phone, and they all get communications (SMS/ email) on their phone regularly. But the essential point is that majority of the messages received will be spam, with only a few being ham or necessary communications. Scammers create fraudulent text m\\
\\
8 min read](https://www.geeksforgeeks.org/sms-spam-detection-using-tensorflow-in-python/)

* * *

- [Python \| Classify Handwritten Digits with Tensorflow\\
\\
\\
Classifying handwritten digits is the basic problem of the machine learning and can be solved in many ways here we will implement them by using TensorFlowUsing a Linear Classifier Algorithm with tf.contrib.learn linear classifier achieves the classification of handwritten digits by making a choice b\\
\\
4 min read](https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/)

* * *

- [Recognizing HandWritten Digits in Scikit Learn\\
\\
\\
Scikit learn is one of the most widely used machine learning libraries in the machine learning community the reason behind that is the ease of code and availability of approximately all functionalities which a machine learning developer will need to build a machine learning model. In this article, w\\
\\
10 min read](https://www.geeksforgeeks.org/recognizing-handwritten-digits-in-scikit-learn/)

* * *

- [Identifying handwritten digits using Logistic Regression in PyTorch\\
\\
\\
Logistic Regression is a very commonly used statistical method that allows us to predict a binary output from a set of independent variables. The various properties of logistic regression and its Python implementation have been covered in this article previously. Now, we shall find out how to implem\\
\\
7 min read](https://www.geeksforgeeks.org/identifying-handwritten-digits-using-logistic-regression-pytorch/)

* * *

- [Python \| Customer Churn Analysis Prediction\\
\\
\\
Customer Churn It is when an existing customer, user, subscriber, or any kind of return client stops doing business or ends the relationship with a company. Types of Customer Churn - Contractual Churn : When a customer is under a contract for a service and decides to cancel the service e.g. Cable TV\\
\\
5 min read](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/)

* * *

- [Online Payment Fraud Detection using Machine Learning in Python\\
\\
\\
As we are approaching modernity, the trend of paying online is increasing tremendously. It is very beneficial for the buyer to pay online as it saves time, and solves the problem of free money. Also, we do not need to carry cash with us. But we all know that Good thing are accompanied by bad things.\\
\\
5 min read](https://www.geeksforgeeks.org/online-payment-fraud-detection-using-machine-learning-in-python/)

* * *

- [Flipkart Reviews Sentiment Analysis using Python\\
\\
\\
Sentiment analysis is a NLP task used to determine the sentiment behind textual data. In context of product reviews it helps in understanding whether the feedback given by customers is positive, negative or neutral. It helps businesses gain valuable insights about customer experiences, product quali\\
\\
4 min read](https://www.geeksforgeeks.org/flipkart-reviews-sentiment-analysis-using-python/)

* * *

- [Loan Approval Prediction using Machine Learning\\
\\
\\
LOANS are the major requirement of the modern world. By this only, Banks get a major part of the total profit. It is beneficial for students to manage their education and living expenses, and for people to buy any kind of luxury like houses, cars, etc. But when it comes to deciding whether the appli\\
\\
5 min read](https://www.geeksforgeeks.org/loan-approval-prediction-using-machine-learning/)

* * *

- [Loan Eligibility Prediction using Machine Learning Models in Python\\
\\
\\
Have you ever thought about the apps that can predict whether you will get your loan approved or not? In this article, we are going to develop one such model that can predict whether a person will get his/her loan approved or not by using some of the background information of the applicant like the\\
\\
5 min read](https://www.geeksforgeeks.org/loan-eligibility-prediction-using-machine-learning-models-in-python/)

* * *

- [Stock Price Prediction using Machine Learning in Python\\
\\
\\
Machine learning proves immensely helpful in many industries in automating tasks that earlier required human labor one such application of ML is predicting whether a particular trade will be profitable or not. In this article, we will learn how to predict a signal that indicates whether buying a par\\
\\
8 min read](https://www.geeksforgeeks.org/stock-price-prediction-using-machine-learning-in-python/)

* * *

- [Bitcoin Price Prediction using Machine Learning in Python\\
\\
\\
Machine learning proves immensely helpful in many industries in automating tasks that earlier required human labor one such application of ML is predicting whether a particular trade will be profitable or not. In this article, we will learn how to predict a signal that indicates whether buying a par\\
\\
7 min read](https://www.geeksforgeeks.org/bitcoin-price-prediction-using-machine-learning-in-python/)

* * *

- [Handwritten Digit Recognition using Neural Network\\
\\
\\
Handwritten digit recognition is a classic problem in machine learning and computer vision. It involves recognizing handwritten digits (0-9) from images or scanned documents. This task is widely used as a benchmark for evaluating machine learning models especially neural networks due to its simplici\\
\\
5 min read](https://www.geeksforgeeks.org/handwritten-digit-recognition-using-neural-network/)

* * *

- [Parkinson Disease Prediction using Machine Learning - Python\\
\\
\\
Parkinson's disease is a progressive neurological disorder that affects movement. Stiffening, tremors and slowing down of movements may be signs of Parkinson's disease. While there is no certain diagnostic test, but we can use machine learning in predicting whether a person has Parkinson's disease b\\
\\
8 min read](https://www.geeksforgeeks.org/parkinson-disease-prediction-using-machine-learning-python/)

* * *

- [Spaceship Titanic Project using Machine Learning - Python\\
\\
\\
If you are a machine learning enthusiast you must have done the Titanic project in which you would have predicted whether a person will survive or not.Â  Spaceship Titanic Project using Machine Learning in PythonIn this article, we will try to solve one such problem which is a slightly modified versi\\
\\
9 min read](https://www.geeksforgeeks.org/spaceship-titanic-project-using-machine-learning-python/)

* * *

- [Rainfall Prediction using Machine Learning - Python\\
\\
\\
Today there are no certain methods by using which we can predict whether there will be rainfall today or not. Even the meteorological department's prediction fails sometimes. In this article, we will learn how to build a machine-learning model which can predict whether there will be rainfall today o\\
\\
7 min read](https://www.geeksforgeeks.org/rainfall-prediction-using-machine-learning-python/)

* * *

- [Autism Prediction using Machine Learning\\
\\
\\
Autism is a neurological disorder that affects a person's ability to interact with others, make eye contact with others, learn and have other behavioral issue. However there is no certain way to tell whether a person has Autism or not because there are no such diagnostics methods available to diagno\\
\\
8 min read](https://www.geeksforgeeks.org/autism-prediction-using-machine-learning/)

* * *

- [Predicting Stock Price Direction using Support Vector Machines\\
\\
\\
We are going to implement an End-to-End project using Support Vector Machines to live Trade For us. You Probably must have Heard of the term stock market which is known to have made the lives of thousands and to have destroyed the lives of millions. If you are not familiar with the stock market you\\
\\
5 min read](https://www.geeksforgeeks.org/predicting-stock-price-direction-using-support-vector-machines/)

* * *

- [Fake News Detection Model using TensorFlow in Python\\
\\
\\
Fake news is a type of misinformation that can mislead readers, influence public opinion, and even damage reputations. Detecting fake news prevents its spread and protects individuals and organizations. Media outlets often use these models to help filter and verify content, ensuring that the news sh\\
\\
5 min read](https://www.geeksforgeeks.org/fake-news-detection-model-using-tensorflow-in-python/)

* * *

- [CIFAR-10 Image Classification in TensorFlow\\
\\
\\
Prerequisites:Image ClassificationConvolution Neural Networks including basic pooling, convolution layers with normalization in neural networks, and dropout.Data Augmentation.Neural Networks.Numpy arrays.In this article, we are going to discuss how to classify images using TensorFlow. Image Classifi\\
\\
8 min read](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/)

* * *

- [Black and white image colorization with OpenCV and Deep Learning\\
\\
\\
In this article, we'll create a program to convert a black & white image i.e grayscale image to a colour image. We're going to use the Caffe colourization model for this program. And you should be familiar with basic OpenCV functions and uses like reading an image or how to load a pre-trained mo\\
\\
3 min read](https://www.geeksforgeeks.org/black-and-white-image-colorization-with-opencv-and-deep-learning/)

* * *

- [ML \| Breast Cancer Wisconsin Diagnosis using Logistic Regression\\
\\
\\
Breast Cancer Wisconsin Diagnosis dataset is commonly used in machine learning to classify breast tumors as malignant (cancerous) or benign (non-cancerous) based on features extracted from breast mass images. In this article we will apply Logistic Regression algorithm for binary classification to pr\\
\\
5 min read](https://www.geeksforgeeks.org/ml-kaggle-breast-cancer-wisconsin-diagnosis-using-logistic-regression/)

* * *

- [ML \| Cancer cell classification using Scikit-learn\\
\\
\\
Machine learning is used in solving real-world problems including medical diagnostics. One such application is classifying cancer cells based on their features and determining whether they are 'malignant' or 'benign'. In this article, we will use Scikit-learn to build a classifier for cancer cell de\\
\\
4 min read](https://www.geeksforgeeks.org/ml-cancer-cell-classification-using-scikit-learn/)

* * *

- [ML \| Kaggle Breast Cancer Wisconsin Diagnosis using KNN and Cross Validation\\
\\
\\
Dataset : It is given by Kaggle from UCI Machine Learning Repository, in one of its challenges. It is a dataset of Breast Cancer patients with Malignant and Benign tumor. K-nearest neighbour algorithm is used to predict whether is patient is having cancer (Malignant tumour) or not (Benign tumour). I\\
\\
3 min read](https://www.geeksforgeeks.org/ml-kaggle-breast-cancer-wisconsin-diagnosis-using-knn/)

* * *

- [Human Scream Detection and Analysis for Controlling Crime Rate - Project Idea\\
\\
\\
Project Title: Human Scream Detection and Analysis for Controlling Crime Rate using Machine Learning and Deep Learning Crime is the biggest social problem of our society which is spreading day by day. Thousands of crimes are committed every day, and still many are occurring right now also all over t\\
\\
6 min read](https://www.geeksforgeeks.org/human-scream-detection-and-analysis-for-controlling-crime-rate-project-idea/)

* * *

- [Multiclass image classification using Transfer learning\\
\\
\\
Image classification is one of the supervised machine learning problems which aims to categorize the images of a dataset into their respective categories or labels. Classification of images of various dog breeds is a classic image classification problem. So, we have to classify more than one class t\\
\\
9 min read](https://www.geeksforgeeks.org/multiclass-image-classification-using-transfer-learning/)

* * *

- [Intrusion Detection System Using Machine Learning Algorithms\\
\\
\\
Problem Statement: The task is to build a network intrusion detector, a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and good normal connections. Introduction: Intrusion Detection System is a software application to detect network intrusion using\\
\\
11 min read](https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/)

* * *

- [Heart Disease Prediction using ANN\\
\\
\\
Deep Learning is a technology of which mimics a human brain in the sense that it consists of multiple neurons with multiple layers like a human brain. The network so formed consists of an input layer, an output layer, and one or more hidden layers. The network tries to learn from the data that is fe\\
\\
3 min read](https://www.geeksforgeeks.org/heart-disease-prediction-using-ann/)

* * *


## Regression Projects

- [IPL Score Prediction using Deep Learning\\
\\
\\
In the modern era of cricket analytics, where each run and decision can change the outcome, the application of Deep Learning for IPL score prediction stands at the forefront of innovation. This article explores the cutting-edge use of advanced algorithms to forecast IPL score in live matches with hi\\
\\
7 min read](https://www.geeksforgeeks.org/ipl-score-prediction-using-deep-learning/)

* * *

- [Dogecoin Price Prediction with Machine Learning\\
\\
\\
Dogecoin is a cryptocurrency, like Ethereum or Bitcoin â€” despite the fact that it's totally different than both of these famous coins. Dogecoin was initially made to some extent as a joke for crypto devotees and took its name from a previously well-known meme. In this article, we will be implementin\\
\\
4 min read](https://www.geeksforgeeks.org/dogecoin-price-prediction-with-machine-learning/)

* * *

- [Zillow Home Value (Zestimate) Prediction in ML\\
\\
\\
In this article, we will try to implement a house price index calculator which revolutionized the whole real estate industry in the US. This will be a regression task in which we have been provided with logarithm differences between the actual and the predicted prices of those homes by using a bench\\
\\
6 min read](https://www.geeksforgeeks.org/zillow-home-value-zestimate-prediction-in-ml/)

* * *

- [Calories Burnt Prediction using Machine Learning\\
\\
\\
In this article, we will learn how to develop a machine learning model using Python which can predict the number of calories a person has burnt during a workout based on some biological measures. Importing Libraries and DatasetPython libraries make it easy for us to handle the data and perform typic\\
\\
5 min read](https://www.geeksforgeeks.org/calories-burnt-prediction-using-machine-learning/)

* * *

- [Vehicle Count Prediction From Sensor Data\\
\\
\\
Prerequisite: Regression and Classification \| Supervised Machine Learning Sensors which are placed in road junctions collect the data of no of vehicles at different junctions and gives data to the transport manager. Now our task is to predict the total no of vehicles based on sensor data. This artic\\
\\
3 min read](https://www.geeksforgeeks.org/vehicle-count-prediction-from-sensor-data/)

* * *

- [Analyzing Selling Price of used Cars using Python\\
\\
\\
Analyzing the selling price of used cars is essential for making informed decisions in the automotive market. Using Python, we can efficiently process and visualize data to uncover key factors influencing car prices. This analysis not only aids buyers and sellers but also enables predictive modeling\\
\\
4 min read](https://www.geeksforgeeks.org/analyzing-selling-price-of-used-cars-using-python/)

* * *

- [Box Office Revenue Prediction Using Linear Regression in ML\\
\\
\\
When a movie is produced then the director would certainly like to maximize his/her movie's revenue. But can we predict what will be the revenue of a movie by using its genre or budget information? This is exactly what we'll learn in this article, we will learn how to implement a machine learning al\\
\\
6 min read](https://www.geeksforgeeks.org/box-office-revenue-prediction-using-linear-regression-in-ml/)

* * *

- [House Price Prediction using Machine Learning in Python\\
\\
\\
House price prediction is a problem in the real estate industry to make informed decisions. By using machine learning algorithms we can predict the price of a house based on various features such as location, size, number of bedrooms and other relevant factors. In this article we will explore how to\\
\\
6 min read](https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/)

* * *

- [ML \| Boston Housing Kaggle Challenge with Linear Regression\\
\\
\\
Boston Housing Data: This dataset was taken from the StatLib library and is maintained by Carnegie Mellon University. This dataset concerns the housing prices in the housing city of Boston. The dataset provided has 506 instances with 13 features.The Description of the dataset is taken fromÂ the below\\
\\
3 min read](https://www.geeksforgeeks.org/ml-boston-housing-kaggle-challenge-with-linear-regression/)

* * *

- [Stock Price Prediction Project using TensorFlow\\
\\
\\
Stock price prediction is a challenging task in the field of finance with applications ranging from personal investment strategies to algorithmic trading. In this article we will explore how to build a stock price prediction model using TensorFlow and Long Short-Term Memory (LSTM) networks a type of\\
\\
5 min read](https://www.geeksforgeeks.org/stock-price-prediction-project-using-tensorflow/)

* * *

- [Medical Insurance Price Prediction using Machine Learning - Python\\
\\
\\
You must have heard some advertisements regarding medical insurance that promises to help financially in case of any medical emergency. One who purchases this type of insurance has to pay premiums monthly and this premium amount varies vastly depending upon various factors.Â  Medical Insurance Price\\
\\
7 min read](https://www.geeksforgeeks.org/medical-insurance-price-prediction-using-machine-learning-python/)

* * *

- [Inventory Demand Forecasting using Machine Learning - Python\\
\\
\\
Vendors selling everyday items need to keep their stock updated so that customers donâ€™t leave empty-handed. Maintaining the right stock levels helps avoid shortages that disappoint customers and prevents overstocking which can increase costs. In this article weâ€™ll learn how to use Machine Learning (\\
\\
6 min read](https://www.geeksforgeeks.org/inventory-demand-forecasting-using-machine-learning-python/)

* * *

- [Ola Bike Ride Request Forecast using ML\\
\\
\\
From telling rickshaw-wala where to go, to tell him where to come we have grown up. Yes, we are talking about online cab and bike facility providers like OLA and Uber. If you had used this app some times then you must have paid some day less and someday more for the same journey. But have you ever t\\
\\
8 min read](https://www.geeksforgeeks.org/ola-bike-ride-request-forecast-using-ml/)

* * *

- [Waiter's Tip Prediction using Machine Learning\\
\\
\\
If you have recently visited a restaurant for a family dinner or lunch and you have tipped the waiter for his generous behavior then this project might excite you. As in this article, we will try to predict what amount of tip a person will give based on his/her visit to the restaurant using some fea\\
\\
7 min read](https://www.geeksforgeeks.org/waiters-tip-prediction-using-machine-learning/)

* * *

- [Predict Fuel Efficiency Using Tensorflow in Python\\
\\
\\
Predicting fuel efficiency is a important task in automotive design and environmental sustainability. In this article we will build a fuel efficiency prediction model using TensorFlow one of the most popular deep learning libraries. We will use the Auto MPG dataset which contains features like engin\\
\\
5 min read](https://www.geeksforgeeks.org/predict-fuel-efficiency-using-tensorflow-in-python/)

* * *

- [Microsoft Stock Price Prediction with Machine Learning\\
\\
\\
In this article, we will implement Microsoft Stock Price Prediction with a Machine Learning technique. We will use TensorFlow, an Open-Source Python Machine Learning Framework developed by Google. TensorFlow makes it easy to implement Time Series forecasting data. Since Stock Price Prediction is one\\
\\
5 min read](https://www.geeksforgeeks.org/microsoft-stock-price-prediction-with-machine-learning/)

* * *

- [Share Price Forecasting Using Facebook Prophet\\
\\
\\
Time series forecast can be used in a wide variety of applications such as Budget Forecasting, Stock Market Analysis, etc. But as useful it is also challenging to forecast the correct projections, Thus can't be easily automated because of the underlying assumptions and factors. The analysts who prod\\
\\
6 min read](https://www.geeksforgeeks.org/share-price-forecasting-using-facebook-prophet/)

* * *

- [Python \| Implementation of Movie Recommender System\\
\\
\\
Recommender System is a system that seeks to predict or filter preferences according to the user's choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general.Â Recommender systems produc\\
\\
3 min read](https://www.geeksforgeeks.org/python-implementation-of-movie-recommender-system/)

* * *

- [How can Tensorflow be used with abalone dataset to build a sequential model?\\
\\
\\
In this article, we will learn how to build a sequential model using TensorFlow in Python to predict the age of an abalone. We may wonder what is an abalone. Answer to this question is that it is a kind of snail. Generally, the age of an Abalone is determined by the physical examination of the abalo\\
\\
8 min read](https://www.geeksforgeeks.org/how-can-tensorflow-be-used-with-abalone-dataset-to-build-a-sequential-model/)

* * *


## Computer Vision Projects

- [OCR of Handwritten digits \| OpenCV\\
\\
\\
OCR which stands for Optical Character Recognition is a computer vision technique used to identify the different types of handwritten digits that are used in common mathematics. To perform OCR in OpenCV we will use the KNN algorithm which detects the nearest k neighbors of a particular data point an\\
\\
2 min read](https://www.geeksforgeeks.org/ocr-of-handwritten-digits-opencv/)

* * *

- [Cartooning an Image using OpenCV - Python\\
\\
\\
Computer Vision as you know (or even if you donâ€™t) is a very powerful tool with immense possibilities. So, when I set up to prepare a comic of one of my friendâ€™s college life, I soon realized that I needed something that would reduce my efforts of actually painting it but would retain the quality an\\
\\
4 min read](https://www.geeksforgeeks.org/cartooning-an-image-using-opencv-python/)

* * *

- [Count number of Object using Python-OpenCV\\
\\
\\
In this article, we will use image processing to count the number of Objects using OpenCV in Python. Google Colab link: https://colab.research.google.com/drive/10lVjcFhdy5LVJxtSoz18WywM92FQAOSV?usp=sharing Module neededOpenCv: OpenCv is an open-source library that is useful for computer vision appli\\
\\
3 min read](https://www.geeksforgeeks.org/count-number-of-object-using-python-opencv/)

* * *

- [Count number of Faces using Python - OpenCV\\
\\
\\
Prerequisites: Face detection using dlib and openCV In this article, we will use image processing to detect and count the number of faces. We are not supposed to get all the features of the face. Instead, the objective is to obtain the bounding box through some methods i.e. coordinates of the face i\\
\\
3 min read](https://www.geeksforgeeks.org/count-number-of-faces-using-python-opencv/)

* * *

- [Text Detection and Extraction using OpenCV and OCR\\
\\
\\
OpenCV (Open source computer vision) is a library of programming functions mainly aimed at real-time computer vision. OpenCV in python helps to process an image and apply various functions like resizing image, pixel manipulations, object detection, etc. In this article, we will learn how to use cont\\
\\
5 min read](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/)

* * *

- [FaceMask Detection using TensorFlow in Python\\
\\
\\
In this article, weâ€™ll discuss our two-phase COVID-19 face mask detector, detailing how our computer vision/deep learning pipeline will be implemented. Weâ€™ll use this Python script to train a face mask detector and review the results. Given the trained COVID-19 face mask detector, weâ€™ll proceed to i\\
\\
9 min read](https://www.geeksforgeeks.org/facemask-detection-using-tensorflow-in-python/)

* * *

- [Dog Breed Classification using Transfer Learning\\
\\
\\
In this tutorial, we will demonstrate how to build a dog breed classifier using transfer learning. This method allows us to use a pre-trained deep learning model and fine-tune it to classify images of different dog breeds. Why to use Transfer Learning for Dog Breed ClassificationTransfer learning is\\
\\
9 min read](https://www.geeksforgeeks.org/dog-breed-classification-using-transfer-learning/)

* * *

- [Flower Recognition Using Convolutional Neural Network\\
\\
\\
Convolutional Neural Network (CNN) are a type of deep learning model specifically designed for processing structured grid data such as images. In this article we will build a CNN model to classify different types of flowers from a dataset containing images of various flowers like roses, daisies, dan\\
\\
6 min read](https://www.geeksforgeeks.org/flower-recognition-using-convolutional-neural-network/)

* * *

- [Emojify using Face Recognition with Machine Learning\\
\\
\\
In this article, we will learn how to implement a modification app that will show an emoji of expression which resembles the expression on your face. This is a fun project based on computer vision in which we use an image classification model in reality to classify different expressions of a person.\\
\\
7 min read](https://www.geeksforgeeks.org/emojify-using-face-recognition-with-machine-learning/)

* * *

- [Cat & Dog Classification using Convolutional Neural Network in Python\\
\\
\\
Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed for processing images. Unlike traditional neural networks CNNs uses convolutional layers to automatically and efficiently extract features such as edges, textures and patterns from images. This makes them hi\\
\\
5 min read](https://www.geeksforgeeks.org/cat-dog-classification-using-convolutional-neural-network-in-python/)

* * *

- [Traffic Signs Recognition using CNN and Keras in Python\\
\\
\\
We always come across incidents of accidents where drivers' Overspeed or lack of vision leads to major accidents. In winter, the risk of road accidents has a 40-50% increase because of the traffic signs' lack of visibility. So here in this article, we will be implementing Traffic Sign recognition us\\
\\
6 min read](https://www.geeksforgeeks.org/traffic-signs-recognition-using-cnn-and-keras-in-python/)

* * *

- [Lung Cancer Detection using Convolutional Neural Network (CNN)\\
\\
\\
Computer Vision is one of the applications of deep neural networks that helps us to automate tasks that earlier required years of expertise and one such use in predicting the presence of cancerous cells. In this article, we will learn how to build a classifier using a simple Convolution Neural Netwo\\
\\
7 min read](https://www.geeksforgeeks.org/lung-cancer-detection-using-convolutional-neural-network-cnn/)

* * *

- [Lung Cancer Detection Using Transfer Learning\\
\\
\\
Computer Vision is one of the applications of deep neural networks that enables us to automate tasks that earlier required years of expertise and one such use in predicting the presence of cancerous cells. In this article, we will learn how to build a classifier using the Transfer Learning technique\\
\\
8 min read](https://www.geeksforgeeks.org/lung-cancer-detection-using-transfer-learning/)

* * *

- [Pneumonia Detection using Deep Learning\\
\\
\\
In this article, we will discuss solving a medical problem i.e. Pneumonia which is a dangerous disease that may occur in one or both lungs usually caused by viruses, fungi or bacteria. We will detect this lung disease based on the x-rays we have. Chest X-rays dataset is taken from Kaggle which conta\\
\\
7 min read](https://www.geeksforgeeks.org/pneumonia-detection-using-deep-learning/)

* * *

- [Detecting Covid-19 with Chest X-ray\\
\\
\\
COVID-19 pandemic is one of the biggest challenges for the healthcare system right now. It is a respiratory disease that affects our lungs and can cause lasting damage to the lungs that led to symptoms such as difficulty in breathing and in some cases pneumonia and respiratory failure. In this artic\\
\\
9 min read](https://www.geeksforgeeks.org/detecting-covid-19-with-chest-x-ray/)

* * *

- [Skin Cancer Detection using TensorFlow\\
\\
\\
In this article, we will learn how to implement a Skin Cancer Detection model using Tensorflow. We will use a dataset that contains images for the two categories that are malignant or benign. We will use the transfer learning technique to achieve better results in less amount of training. We will us\\
\\
5 min read](https://www.geeksforgeeks.org/skin-cancer-detection-using-tensorflow/)

* * *

- [Age Detection using Deep Learning in OpenCV\\
\\
\\
The task of age prediction might sound simple at first but it's quite challenging in real-world applications. While predicting age is typically seen as a regression problem this approach faces many uncertainties like camera quality, brightness, climate condition, background, etc. In this article we'\\
\\
5 min read](https://www.geeksforgeeks.org/age-detection-using-deep-learning-in-opencv/)

* * *

- [Face and Hand Landmarks Detection using Python - Mediapipe, OpenCV\\
\\
\\
In this article, we will use mediapipe python library to detect face and hand landmarks. We will be using a Holistic model from mediapipe solutions to detect all the face and hand landmarks. We will be also seeing how we can access different landmarks of the face and hands which can be used for diff\\
\\
4 min read](https://www.geeksforgeeks.org/face-and-hand-landmarks-detection-using-python-mediapipe-opencv/)

* * *

- [Detecting COVID-19 From Chest X-Ray Images using CNN\\
\\
\\
A Django Based Web Application built for the purpose of detecting the presence of COVID-19 from Chest X-Ray images with multiple machine learning models trained on pre-built architectures. Three different machine learning models were used to build this project namely Xception, ResNet50, and VGG16. T\\
\\
5 min read](https://www.geeksforgeeks.org/detecting-covid-19-from-chest-x-ray-images-using-cnn/)

* * *

- [Image Segmentation Using TensorFlow\\
\\
\\
Image segmentation refers to the task of annotating a single class to different groups of pixels. While the input is an image, the output is a mask that draws the region of the shape in that image. Image segmentation has wide applications in domains such as medical image analysis, self-driving cars,\\
\\
7 min read](https://www.geeksforgeeks.org/image-segmentation-using-tensorflow/)

* * *

- [License Plate Recognition with OpenCV and Tesseract OCR\\
\\
\\
License Plate Recognition is widely used for automated identification of vehicle registration plates for security purpose and law enforcement. By combining computer vision techniques with Optical Character Recognition (OCR) we can extract license plate numbers from images enabling applications in ar\\
\\
5 min read](https://www.geeksforgeeks.org/license-plate-recognition-with-opencv-and-tesseract-ocr/)

* * *

- [Detect and Recognize Car License Plate from a video in real time\\
\\
\\
Recognizing a Car License Plate is a very important task for a camera surveillance-based security system. We can extract the license plate from an image using some computer vision techniques and then we can use Optical Character Recognition to recognize the license number. Here I will guide you thro\\
\\
11 min read](https://www.geeksforgeeks.org/detect-and-recognize-car-license-plate-from-a-video-in-real-time/)

* * *

- [Residual Networks (ResNet) - Deep Learning\\
\\
\\
After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, Every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate. This works for less number of layers, but when we increase the number of layers, there is a common proble\\
\\
9 min read](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

* * *


## Natural Language Processing Projects

- [Twitter Sentiment Analysis using Python\\
\\
\\
This article covers the sentiment analysis of any topic by parsing the tweets fetched from Twitter using Python. What is sentiment analysis? Sentiment Analysis is the process of 'computationally' determining whether a piece of writing is positive, negative or neutral. Itâ€™s also known as opinion mini\\
\\
10 min read](https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/)

* * *

- [Facebook Sentiment Analysis using python\\
\\
\\
This article is a Facebook sentiment analysis using Vader, nowadays many government institutions and companies need to know their customers' feedback and comment on social media such as Facebook. What is sentiment analysis? Sentiment analysis is one of the best modern branches of machine learning, w\\
\\
6 min read](https://www.geeksforgeeks.org/facebook-sentiment-analysis-using-python/)

* * *

- [Next Sentence Prediction using BERT\\
\\
\\
Pre-requisite: BERT-GFG BERT stands for Bidirectional Representation for Transformers. It was proposed by researchers at Google Research in 2018. Although, the main aim of that was to improve the understanding of the meaning of queries related to Google Search. A study shows that Google encountered\\
\\
7 min read](https://www.geeksforgeeks.org/next-sentence-prediction-using-bert/)

* * *

- [Hate Speech Detection using Deep Learning\\
\\
\\
There must be times when you have come across some social media post whose main aim is to spread hate and controversies or use abusive language on social media platforms. As the post consists of textual information to filter out such Hate Speeches NLP comes in handy. This is one of the main applicat\\
\\
7 min read](https://www.geeksforgeeks.org/hate-speech-detection-using-deep-learning/)

* * *

- [Image Caption Generator using Deep Learning on Flickr8K dataset\\
\\
\\
Generating a caption for a given image is a challenging problem in the deep learning domain. In this article we will use different computer vision and NLP techniques to recognize the context of an image and describe them in a natural language like English. We will build a working model of the image\\
\\
12 min read](https://www.geeksforgeeks.org/image-caption-generator-using-deep-learning-on-flickr8k-dataset/)

* * *

- [Movie recommendation based on emotion in Python\\
\\
\\
Movies that effectively portray and explore emotions resonate deeply with audiences because they tap into our own emotional experiences and vulnerabilities. A well-crafted emotional movie can evoke empathy, understanding, and self-reflection, allowing viewers to connect with the characters and their\\
\\
4 min read](https://www.geeksforgeeks.org/movie-recommendation-based-emotion-python/)

* * *

- [Speech Recognition in Python using Google Speech API\\
\\
\\
Speech Recognition is an important feature in several applications used such as home automation, artificial intelligence, etc. This article aims to provide an introduction to how to make use of the SpeechRecognition library of Python. This is useful as it can be used on microcontrollers such as Rasp\\
\\
4 min read](https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/)

* * *

- [Voice Assistant using python\\
\\
\\
As we know Python is a suitable language for scriptwriters and developers. Letâ€™s write a script for Voice Assistant using Python. The query for the assistant can be manipulated as per the userâ€™s need. Speech recognition is the process of converting audio into text. This is commonly used in voice ass\\
\\
11 min read](https://www.geeksforgeeks.org/voice-assistant-using-python/)

* * *

- [Human Activity Recognition - Using Deep Learning Model\\
\\
\\
Human activity recognition using smartphone sensors like accelerometer is one of the hectic topics of research. HAR is one of the time series classification problem. In this project various machine learning and deep learning models have been worked out to get the best final result. In the same seque\\
\\
6 min read](https://www.geeksforgeeks.org/human-activity-recognition-using-deep-learning-model/)

* * *

- [Fine-tuning BERT model for Sentiment Analysis\\
\\
\\
Google created a transformer-based machine learning approach for natural language processing pre-training called Bidirectional Encoder Representations from Transformers. It has a huge number of parameters, hence training it on a small dataset would lead to overfitting. This is why we use a pre-train\\
\\
7 min read](https://www.geeksforgeeks.org/fine-tuning-bert-model-for-sentiment-analysis/)

* * *

- [Sentiment Classification Using BERT\\
\\
\\
BERT stands for Bidirectional Representation for Transformers and was proposed by researchers at Google AI language in 2018. Although the main aim of that was to improve the understanding of the meaning of queries related to Google Search, BERT becomes one of the most important and complete architec\\
\\
13 min read](https://www.geeksforgeeks.org/sentiment-classification-using-bert/)

* * *

- [Sentiment Analysis with an Recurrent Neural Networks (RNN)\\
\\
\\
Recurrent Neural Networks (RNNs) excel in sequence tasks such as sentiment analysis due to their ability to capture context from sequential data. In this article we will be apply RNNs to analyze the sentiment of customer reviews from Swiggy food delivery platform. The goal is to classify reviews as\\
\\
3 min read](https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/)

* * *

- [Building an Autocorrector Using NLP in Python\\
\\
\\
Autocorrect feature predicts and correct misspelled words, it helps to save time invested in the editing of articles, emails and reports. This feature is added many websites and social media platforms to ensure easy typing. In this tutorial we will build a Python-based autocorrection feature using N\\
\\
4 min read](https://www.geeksforgeeks.org/autocorrector-feature-using-nlp-in-python/)

* * *

- [Python \| NLP analysis of Restaurant reviews\\
\\
\\
Natural language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. It is the branch of mach\\
\\
7 min read](https://www.geeksforgeeks.org/python-nlp-analysis-of-restaurant-reviews/)

* * *

- [Restaurant Review Analysis Using NLP and SQLite\\
\\
\\
Normally, a lot of businesses are remained as failures due to lack of profit, lack of proper improvement measures. Mostly, restaurant owners face a lot of difficulties to improve their productivity. This project really helps those who want to increase their productivity, which in turn increases thei\\
\\
9 min read](https://www.geeksforgeeks.org/restaurant-review-analysis-using-nlp-and-sqlite/)

* * *

- [Twitter Sentiment Analysis using Python\\
\\
\\
This article covers the sentiment analysis of any topic by parsing the tweets fetched from Twitter using Python. What is sentiment analysis? Sentiment Analysis is the process of 'computationally' determining whether a piece of writing is positive, negative or neutral. Itâ€™s also known as opinion mini\\
\\
10 min read](https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/)

* * *


## Clustering Projects

- [Customer Segmentation using Unsupervised Machine Learning in Python\\
\\
\\
Customer Segmentation involves grouping customers based on shared characteristics, behaviors and preferences. By segmenting customers, businesses can tailor their strategies and target specific groups more effectively and enhance overall market value. Today we will use Unsupervised Machine Learning\\
\\
5 min read](https://www.geeksforgeeks.org/customer-segmentation-using-unsupervised-machine-learning-in-python/)

* * *

- [Music Recommendation System Using Machine Learning\\
\\
\\
When did we see a video on youtube let's say it was funny then the next time you open your youtube app you get recommendations of some funny videos in your feed ever thought about how? This is nothing but an application of Machine Learning using which recommender systems are built to provide persona\\
\\
4 min read](https://www.geeksforgeeks.org/music-recommendation-system-using-machine-learning/)

* * *

- [K means Clustering - Introduction\\
\\
\\
K-Means Clustering is an Unsupervised Machine Learning algorithm which groups the unlabeled dataset into different clusters. The article aims to explore the fundamentals and working of k means clustering along with its implementation. Understanding K-means ClusteringK-means clustering is a technique\\
\\
6 min read](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

* * *

- [Image Segmentation using K Means Clustering\\
\\
\\
Image Segmentation: In computer vision, image segmentation is the process of partitioning an image into multiple segments. The goal of segmenting an image is to change the representation of an image into something that is more meaningful and easier to analyze. It is usually used for locating objects\\
\\
4 min read](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)

* * *


## Recommender System Project

- [AI Driven Snake Game using Deep Q Learning\\
\\
\\
Content has been removed from this Article\\
\\
1 min read](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/)

* * *


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1910900812.1745056593&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=1106310555)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)