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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/clustering-in-machine-learning/?type%3Darticle%26id%3D172234&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
K means Clustering - Introduction\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

# Clustering in Machine Learning

Last Updated : 27 Jan, 2025

Comments

Improve

Suggest changes

67 Likes

Like

Report

In real world, not every **data we work upon has a target variable**. Have you ever wondered how Netflix groups similar movies together or how Amazon organizes its vast product catalog? These are **real-world applications of clustering**. This kind of data cannot be analyzed using supervised learning algorithms.

When the goal is to group similar data points in a dataset, then we use cluster analysis. In this guide, we’ll learn understand concept of clustering, its applications, and some popular clustering algorithms.

## What is Clustering?

The task of **grouping data points based on their similarity with each other is called Clustering or Cluster Analysis**. This method is defined under the branch of [unsupervised learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/), which aims at gaining insights from unlabelled data points.

Think of it as you have a dataset of customers shopping habits. **Clustering can help you group customers with similar purchasing behaviors, which can then be used for targeted marketing, product recommendations, or customer segmentation**

For Example, In the graph given below, we can clearly see that there are 3 circular clusters forming on the basis of distance.

![Clustering in Machine Learning](https://media.geeksforgeeks.org/wp-content/uploads/merge3cluster.jpg)

Now it is not necessary that the clusters formed **must be circular in shape**. The shape of clusters can be arbitrary. There are many algorithms that work well with detecting arbitrary shaped clusters.

For example, In the below given graph we can see that the clusters formed are not circular in shape.

![Arbitrary shaped clusters identified by Clustering analysis](https://media.geeksforgeeks.org/wp-content/uploads/clusteringg.jpg)

## Types of Clustering

Broadly speaking, there are 2 types of clustering that can be performed to group similar data points:

- **Hard Clustering:** In this type of clustering, each data point belongs to a cluster completely or not. For example, Let’s say there are 4 data point and we have to cluster them into 2 clusters. So each data point will either belong to cluster 1 or cluster 2.

| Data Points | Clusters |
| --- | --- |
| A | C1 |
| B | C2 |
| C | C2 |
| D | C1 |

- **Soft Clustering:** In this type of clustering, instead of assigning each data point into a separate cluster, a probability or likelihood of that point being that cluster is evaluated. For example, Let’s say there are 4 data point and we have to cluster them into 2 clusters. So we will be evaluating a probability of a data point belonging to both clusters. This probability is calculated for all data points.

| Data Points | Probability of C1 | Probability of C2 |
| A | 0.91 | 0.09 |
| B | 0.3 | 0.7 |
| C | 0.17 | 0.83 |
| D | 1 | 0 |

## Uses of Clustering

Now before we begin with types of clustering algorithms, we will go through the use cases of Clustering algorithms. Clustering algorithms are majorly used for:

- **Market Segmentation:** Businesses use clustering to group their customers and use targeted advertisements to attract more audience.
- **Market Basket Analysis:** Shop owners analyze their sales and figure out which items are majorly bought together by the customers. For example, In USA, according to a study diapers and beers were usually bought together by fathers.
- **Social Network Analysis:** Social media sites use your data to understand your browsing behavior and provide you with targeted friend recommendations or content recommendations.
- **Medical Imaging:** Doctors use Clustering to find out diseased areas in diagnostic images like X-rays.
- **Anomaly Detection:** To find outliers in a stream of real-time dataset or forecasting fraudulent transactions we can use clustering to identify them.
- **Simplify working with large datasets:** Each cluster is given a cluster ID after clustering is complete. Now, you may reduce a feature set’s whole feature set into its cluster ID. Clustering is effective when it can represent a complicated case with a straightforward cluster ID. Using the same principle, clustering data can make complex datasets simpler.

There are many more use cases for clustering but there are some of the major and common use cases of clustering. Moving forward we will be discussing Clustering Algorithms that will help you perform the above tasks.

## Types of Clustering Methods

At the surface level, **clustering helps in the analysis of unstructured data.** **Graphing, the shortest distance, and the density of the data points are a few of the elements that influence cluster formation**. Clustering is the process of determining how related the objects are **based on a metric called the similarity measure**.

Similarity metrics **are easier to locate in smaller sets of features and harder as the number of features increases**. Depending on the type of clustering algorithm being utilized, several techniques are employed to group the data from the datasets. In this part, the clustering techniques are described. Various types of clustering algorithms are:

1. Centroid-based Clustering (Partitioning methods)
2. Density-based Clustering (Model-based methods)
3. Connectivity-based Clustering (Hierarchical clustering)

We will be going through each of these types in brief.

### 1\. Centroid-based Clustering (Partitioning methods)

Centroid-based clustering organizes data points around central vectors (centroids) that represent clusters. Each data point belongs to the cluster with the nearest centroid. Generally, the similarity measure chosen for these algorithms are Euclidian distance, Manhattan Distance or Minkowski Distance.

The datasets are separated into a **predetermined number of clusters, and each cluster is referenced by a vector of values. When compared to the vector value, the input data variable shows no difference and joins the cluster.**

The major drawback for centroid-based algorithms is the requirement that we establish the number of clusters, “k,” either intuitively or scientifically (using the Elbow Method) before any clustering machine learning system starts allocating the data points. Despite this limitation, it remains the most popular type of clustering due to its simplicity and efficiency. Popular algorithms of [Centroid-based clustering](https://www.geeksforgeeks.org/partitioning-method-k-mean-in-data-mining/) are:

- [K-means](https://www.geeksforgeeks.org/k-means-clustering-introduction/) and
- [K-medoids](https://www.geeksforgeeks.org/ml-k-medoids-clustering-with-example/) clustering

are some examples of this type clustering.

### 2\. Density-based Clustering (Model-based methods)

Density-based clustering identifies clusters as areas of high density separated by regions of low density in the data space. Unlike centroid-based methods, density-based clustering **automatically determines the number of clusters and is less susceptible to initialization positions**. **Key Characteristics:**

- Can find arbitrarily shaped clusters
- Handles noise and outliers well
- Excels with clusters of different sizes and shapes
- Ideal for datasets with irregularly shaped or overlapping clusters
- Effectively manages both dense and sparse data regions
- Focus on local density allows detection of various cluster morphologies

> The most popular [density-based clustering](https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/) algorithm is [DBSCAN](https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/) and [OPTICS (Ordering Points To Identify Clustering Structure)](https://www.geeksforgeeks.org/ml-optics-clustering-explanation/).

### 3\. Connectivity-based Clustering (Hierarchical clustering)

Connectivity-based clustering builds a **hierarchy of clusters using a measure of connectivity based on distance** when organizing a collection of items based on their similarities.  This method builds a **dendrogram**, a tree-like structure that visually represents the relationships between objects.

> At the base of the tree, each object starts as its own individual cluster. The algorithm then evaluates how similar the objects are to one another and begins merging the closest pairs of clusters into larger groups. This process continues iteratively, with clusters being combined step by step, until all objects are united into a single cluster at the top of the tree.

There are 2 approaches for [Hierarchical clustering](https://www.geeksforgeeks.org/hierarchical-clustering/):

- **Divisive Clustering:** It follows a top-down approach, here we consider all data points to be part one big cluster and then this cluster is divide into smaller groups.
- **Agglomerative Clustering:** It follows a bottom-up approach, here we consider all data points to be part of individual clusters and then these clusters are clubbed together to make one big cluster with all data points.

For implementing and understand difference between both techniques , please refer to : [Agglomerative clustering and Divisive clustering](https://www.geeksforgeeks.org/difference-between-agglomerative-clustering-and-divisive-clustering/)

> Till now, we have understood **traditional “hard” clustering methods**, where each data point is assigned to exactly one cluster. These methods, like K-Means and hierarchical clustering, are powerful and widely used, but they have limitations when dealing with ambiguous or overlapping data. After learning all about hard clustering methods we can addresses these limitations with **soft clustering that** allows data points to belong to **multiple clusters simultaneously**, with varying degrees of membership. This approach is particularly useful when the boundaries between clusters are not clear-cut or when data points exhibit characteristics of more than one group.

Two of the most popular soft clustering techniques are:

### 4\. Distribution-based Clustering

Distribution-based clustering is a technique that assumes **data points are generated from a mixture of probability distributions (e.g., Gaussian, Poisson, etc.)**. The goal is to identify clusters by estimating the parameters of these distributions. In distribution-based clustering:

- Each cluster is represented by a probability distribution.
- Data points are assigned to clusters based on how likely they are to belong to each distribution.
- Unlike distance-based methods (e.g., K-Means), this approach can capture clusters of varying shapes, sizes, and densities.

Many real-world datasets, such as sensor data, financial data, or biological measurements, naturally follow statistical distributions. The most popular distribution-based clustering algorithm is [Gaussian Mixture Model](https://www.geeksforgeeks.org/gaussian-mixture-model/).

### 5\. **Fuzzy Clustering**

Fuzzy clustering allows data points to belong to multiple clusters with varying degrees of membership.

-  Each data point is assigned a membership value between 0 and 1 for every cluster.
- These membership values indicate the degree to which a data point belongs to a particular cluster.

Please refer to [fuzzy clustering methods](https://www.geeksforgeeks.org/ml-fuzzy-clustering/) for in-depth understanding. Although this method and it’s algorithms are used for higher-level problem statements involving complex datasets

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

[K means Clustering - Introduction](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

[S](https://www.geeksforgeeks.org/user/Surya%20Priy/)

[Surya Priy](https://www.geeksforgeeks.org/user/Surya%20Priy/)

Follow

67

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [ML-Clustering](https://www.geeksforgeeks.org/tag/ml-clustering/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Machine Learning Algorithms\\
\\
\\
Machine learning algorithms are essentially sets of instructions that allow computers to learn from data, make predictions, and improve their performance over time without being explicitly programmed. Machine learning algorithms are broadly categorized into three types: Supervised Learning: Algorith\\
\\
9 min read](https://www.geeksforgeeks.org/machine-learning-algorithms/)
[Top 15 Machine Learning Algorithms Every Data Scientist Should Know in 2025\\
\\
\\
Machine Learning (ML) Algorithms are the backbone of everything from Netflix recommendations to fraud detection in financial institutions. These algorithms form the core of intelligent systems, empowering organizations to analyze patterns, predict outcomes, and automate decision-making processes. Wi\\
\\
15 min read](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/)

## Linear Model Regression

- [Ordinary Least Squares (OLS) using statsmodels\\
\\
\\
Ordinary Least Squares (OLS) is a widely used statistical method for estimating the parameters of a linear regression model. It minimizes the sum of squared residuals between observed and predicted values. In this article we will learn how to implement Ordinary Least Squares (OLS) regression using P\\
\\
3 min read](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)

* * *

- [Linear Regression (Python Implementation)\\
\\
\\
Linear regression is a statistical method that is used to predict a continuous dependent variable i.e target variable based on one or more independent variables. This technique assumes a linear relationship between the dependent and independent variables which means the dependent variable changes pr\\
\\
14 min read](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

* * *

- [ML \| Multiple Linear Regression using Python\\
\\
\\
Linear regression is a fundamental statistical method widely used for predictive analysis. It models the relationship between a dependent variable and a single independent variable by fitting a linear equation to the data. Multiple Linear Regression is an extension of this concept that allows us to\\
\\
4 min read](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/)

* * *

- [Polynomial Regression ( From Scratch using Python )\\
\\
\\
Prerequisites Linear RegressionGradient DescentIntroductionLinear Regression finds the correlation between the dependent variable ( or target variable ) and independent variables ( or features ). In short, it is a linear model to fit the data linearly. But it fails to fit and catch the pattern in no\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/)

* * *

- [Bayesian Linear Regression\\
\\
\\
Linear regression is based on the assumption that the underlying data is normally distributed and that all relevant predictor variables have a linear relationship with the outcome. But In the real world, this is not always possible, it will follows these assumptions, Bayesian regression could be the\\
\\
11 min read](https://www.geeksforgeeks.org/implementation-of-bayesian-regression/)

* * *

- [How to Perform Quantile Regression in Python\\
\\
\\
In this article, we are going to see how to perform quantile regression in Python. Linear regression is defined as the statistical method that constructs a relationship between a dependent variable and an independent variable as per the given set of variables. While performing linear regression we a\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-perform-quantile-regression-in-python/)

* * *

- [Isotonic Regression in Scikit Learn\\
\\
\\
Isotonic regression is a regression technique in which the predictor variable is monotonically related to the target variable. This means that as the value of the predictor variable increases, the value of the target variable either increases or decreases in a consistent, non-oscillating manner. Mat\\
\\
6 min read](https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/)

* * *

- [Stepwise Regression in Python\\
\\
\\
Stepwise regression is a method of fitting a regression model by iteratively adding or removing variables. It is used to build a model that is accurate and parsimonious, meaning that it has the smallest number of variables that can explain the data. There are two main types of stepwise regression: F\\
\\
6 min read](https://www.geeksforgeeks.org/stepwise-regression-in-python/)

* * *

- [Least Angle Regression (LARS)\\
\\
\\
Regression is a supervised machine learning task that can predict continuous values (real numbers), as compared to classification, that can predict categorical or discrete values. Before we begin, if you are a beginner, I highly recommend this article. Least Angle Regression (LARS) is an algorithm u\\
\\
3 min read](https://www.geeksforgeeks.org/least-angle-regression-lars/)

* * *


## Linear Model Classification

- [Logistic Regression in Machine Learning\\
\\
\\
In our previous discussion, we explored the fundamentals of machine learning and walked through a hands-on implementation of Linear Regression. Now, let's take a step forward and dive into one of the first and most widely used classification algorithms â€” Logistic Regression What is Logistic Regressi\\
\\
13 min read](https://www.geeksforgeeks.org/understanding-logistic-regression/)

* * *

- [Understanding Activation Functions in Depth\\
\\
\\
In artificial neural networks, the activation function of a neuron determines its output for a given input. This output serves as the input for subsequent neurons in the network, continuing the process until the network solves the original problem. Consider a binary classification problem, where the\\
\\
6 min read](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

* * *


## Regularization

- [Implementation of Lasso Regression From Scratch using Python\\
\\
\\
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that combines prediction with feature selection. It does this by adding a penalty term to the cost function shrinking less relevant feature's coefficients to zero. This makes it effective for high-dim\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/)

* * *

- [Implementation of Ridge Regression from Scratch using Python\\
\\
\\
Prerequisites: Linear Regression Gradient Descent Introduction: Ridge Regression ( or L2 Regularization ) is a variation of Linear Regression. In Linear Regression, it minimizes the Residual Sum of Squares ( or RSS or cost function ) to fit the training examples perfectly as possible. The cost funct\\
\\
4 min read](https://www.geeksforgeeks.org/implementation-of-ridge-regression-from-scratch-using-python/)

* * *

- [Implementation of Elastic Net Regression From Scratch\\
\\
\\
Prerequisites: Linear RegressionGradient DescentLasso & Ridge RegressionIntroduction: Elastic-Net Regression is a modification of Linear Regression which shares the same hypothetical function for prediction. The cost function of Linear Regression is represented by J. \[Tex\]\\frac{1}{m} \\sum\_{i=1}^\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)

* * *


## K-Nearest Neighbors (KNN)

- [Implementation of Elastic Net Regression From Scratch\\
\\
\\
Prerequisites: Linear RegressionGradient DescentLasso & Ridge RegressionIntroduction: Elastic-Net Regression is a modification of Linear Regression which shares the same hypothetical function for prediction. The cost function of Linear Regression is represented by J. \[Tex\]\\frac{1}{m} \\sum\_{i=1}^\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)

* * *

- [Brute Force Approach and its pros and cons\\
\\
\\
In this article, we will discuss the Brute Force Algorithm and what are its pros and cons. What is the Brute Force Algorithm?A brute force algorithm is a simple, comprehensive search strategy that systematically explores every option until a problem's answer is discovered. It's a generic approach to\\
\\
3 min read](https://www.geeksforgeeks.org/brute-force-approach-and-its-pros-and-cons/)

* * *

- [Implementation of KNN classifier using Scikit - learn - Python\\
\\
\\
K-Nearest Neighbors isÂ aÂ mostÂ simpleÂ butÂ fundamentalÂ classifierÂ algorithmÂ in Machine Learning. ItÂ isÂ underÂ the supervised learningÂ categoryÂ andÂ usedÂ withÂ greatÂ intensityÂ forÂ pattern recognition, data mining andÂ analysis ofÂ intrusion.Â It is widely disposable in real-life scenarios since it is non-par\\
\\
3 min read](https://www.geeksforgeeks.org/ml-implementation-of-knn-classifier-using-sklearn/)

* * *

- [Regression using k-Nearest Neighbors in R Programming\\
\\
\\
Machine learning is a subset of Artificial Intelligence that provides a machine with the ability to learn automatically without being explicitly programmed. The machine in such cases improves from the experience without human intervention and adjusts actions accordingly. It is primarily of 3 types:\\
\\
5 min read](https://www.geeksforgeeks.org/regression-using-k-nearest-neighbors-in-r-programming/)

* * *


## Support Vector Machines

- [Support Vector Machine (SVM) Algorithm\\
\\
\\
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. While it can handle regression problems, SVM is particularly well-suited for classification tasks. SVM aims to find the optimal hyperplane in an N-dimensional space to separate data\\
\\
10 min read](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)

* * *

- [Classifying data using Support Vector Machines(SVMs) in Python\\
\\
\\
Introduction to SVMs: In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. A Support Vector Machine (SVM) is a discriminative classifier\\
\\
4 min read](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

* * *

- [Support Vector Regression (SVR) using Linear and Non-Linear Kernels in Scikit Learn\\
\\
\\
Support vector regression (SVR) is a type of support vector machine (SVM) that is used for regression tasks. It tries to find a function that best predicts the continuous output value for a given input value. SVR can use both linear and non-linear kernels. A linear kernel is a simple dot product bet\\
\\
5 min read](https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/)

* * *

- [Major Kernel Functions in Support Vector Machine (SVM)\\
\\
\\
In previous article we have discussed about SVM(Support Vector Machine) in Machine Learning. Now we are going to learnÂ  in detail about SVM Kernel and Different Kernel Functions and its examples. Types of SVM Kernel FunctionsSVM algorithm use the mathematical function defined by the kernel. Kernel F\\
\\
4 min read](https://www.geeksforgeeks.org/major-kernel-functions-in-support-vector-machine-svm/)

* * *


[ML \| Stochastic Gradient Descent (SGD)\\
\\
\\
Stochastic Gradient Descent (SGD) is an optimization algorithm in machine learning, particularly when dealing with large datasets. It is a variant of the traditional gradient descent algorithm but offers several advantages in terms of efficiency and scalability, making it the go-to method for many d\\
\\
8 min read](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)

## Decision Tree

- [Major Kernel Functions in Support Vector Machine (SVM)\\
\\
\\
In previous article we have discussed about SVM(Support Vector Machine) in Machine Learning. Now we are going to learnÂ  in detail about SVM Kernel and Different Kernel Functions and its examples. Types of SVM Kernel FunctionsSVM algorithm use the mathematical function defined by the kernel. Kernel F\\
\\
4 min read](https://www.geeksforgeeks.org/major-kernel-functions-in-support-vector-machine-svm/)

* * *

- [CART (Classification And Regression Tree) in Machine Learning\\
\\
\\
CART( Classification And Regression Trees) is a variation of the decision tree algorithm. It can handle both classification and regression tasks. Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train Decision Trees (also called â€œgrowingâ€ trees). CART was first produced b\\
\\
11 min read](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/)

* * *

- [Decision Tree Classifiers in R Programming\\
\\
\\
Classification is the task in which objects of several categories are categorized into their respective classes using the properties of classes. A classification model is typically used to, Predict the class label for a new unlabeled data objectProvide a descriptive model explaining what features ch\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-classifiers-in-r-programming/)

* * *

- [Python \| Decision Tree Regression using sklearn\\
\\
\\
When it comes to predicting continuous values, Decision Tree Regression is a powerful and intuitive machine learning technique. Unlike traditional linear regression, which assumes a straight-line relationship between input features and the target variable, Decision Tree Regression is a non-linear re\\
\\
4 min read](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/)

* * *


## Ensemble Learning

- [Ensemble Methods in Python\\
\\
\\
Ensemble means a group of elements viewed as a whole rather than individually. An Ensemble method creates multiple models and combines them to solve it. Ensemble methods help to improve the robustness/generalizability of the model. In this article, we will discuss some methods with their implementat\\
\\
11 min read](https://www.geeksforgeeks.org/ensemble-methods-in-python/)

* * *

- [Random Forest Regression in Python\\
\\
\\
A random forest is an ensemble learning method that combines the predictions from multiple decision trees to produce a more accurate and stable prediction. It is a type of supervised learning algorithm that can be used for both classification and regression tasks. In regression task we can use Rando\\
\\
9 min read](https://www.geeksforgeeks.org/random-forest-regression-in-python/)

* * *

- [ML \| Extra Tree Classifier for Feature Selection\\
\\
\\
Prerequisites: Decision Tree Classifier Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a "forest" to output it's classification result. In concept, it is very si\\
\\
6 min read](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/)

* * *

- [Implementing the AdaBoost Algorithm From Scratch\\
\\
\\
AdaBoost means Adaptive Boosting and it is a is a powerful ensemble learning technique that combines multiple weak classifiers to create a strong classifier. It works by sequentially adding classifiers to correct the errors made by previous models giving more weight to the misclassified data points.\\
\\
3 min read](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/)

* * *

- [XGBoost\\
\\
\\
Traditional machine learning models like decision trees and random forests are easy to interpret but often struggle with accuracy on complex datasets. XGBoost, short for eXtreme Gradient Boosting, is an advanced machine learning algorithm designed for efficiency, speed, and high performance. What is\\
\\
9 min read](https://www.geeksforgeeks.org/xgboost/)

* * *

- [CatBoost in Machine Learning\\
\\
\\
When working with machine learning, we often deal with datasets that include categorical data. We use techniques like One-Hot Encoding or Label Encoding to convert these categorical features into numerical values. However One-Hot Encoding can lead to sparse matrix and cause overfitting. This is wher\\
\\
7 min read](https://www.geeksforgeeks.org/catboost-ml/)

* * *

- [LightGBM (Light Gradient Boosting Machine)\\
\\
\\
LightGBM is an open-source high-performance framework developed by Microsoft. It is an ensemble learning framework that uses gradient boosting method which constructs a strong learner by sequentially adding weak learners in a gradient descent manner. It's designed for efficiency, scalability and hig\\
\\
7 min read](https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/)

* * *

- [Stacking in Machine Learning\\
\\
\\
Stacking is a way to ensemble multiple classifications or regression model. There are many ways to ensemble models, the widely known models are Bagging or Boosting. Bagging allows multiple similar models with high variance are averaged to decrease variance. Boosting builds multiple incremental model\\
\\
2 min read](https://www.geeksforgeeks.org/stacking-in-machine-learning/)

* * *


Like67

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/clustering-in-machine-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1316801127.1745056455&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=919774549)