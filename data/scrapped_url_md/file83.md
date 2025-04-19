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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/hierarchical-clustering/?type%3Darticle%26id%3D301360&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Implementing Agglomerative Clustering using Sklearn\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/)

# Hierarchical Clustering in Machine Learning

Last Updated : 04 Feb, 2025

Comments

Improve

Suggest changes

25 Likes

Like

Report

In the real world **data often lacks a target variable making supervised learning impractical.** Have you ever wondered how social networks like Facebook recommend friends or how scientists group similar species together? These are some examples of hierarchical clustering that we will learn about in this article.

## Why hierarchical clustering?

**Hierarchical clustering** is a technique used to group similar data points together based on their similarity creating a **hierarchy or tree-like structure**. The key idea is to begin with each data point as its own separate cluster and then progressively merge or split them based on their similarity.

Lets understand this with the help of an example

Imagine you have four fruits with different weights: an **apple (100g)**, **a banana (120g), a cherry (50g), and a grape (30g)**. Hierarchical clustering starts by treating each _**fruit as its own group**_.

- It then merges the closest groups based on their weights.
- First, the cherry and grape are grouped together because they are the lightest.
- Next, the apple and banana are grouped together.

Finally, all the fruits are merged into one large group, showing how hierarchical clustering progressively combines the most similar data points.

## Getting Started with Dendogram

A **dendrogram** is like a family tree for clusters. It shows how individual data points or groups of data merge together. The bottom shows each data point as its own group, and as you move up, similar groups are combined. The lower the merge point, the more similar the groups are. It helps you see how things are grouped step by step.

The working of the dendrogram can be explained using the below diagram:

![UntitledDiagram7](https://media.geeksforgeeks.org/wp-content/uploads/20250204154508934309/UntitledDiagram7.png)

Dendogram

In this image, on the left side, there are five points labeled P, Q, R, S, and T. These represent individual data points that are being clustered. On the right side, there’s a **dendrogram**, which shows how these points are grouped together step by step.

- At the bottom of the dendrogram, the points P, Q, R, S, and T are all separate.
- As you move up, the closest points are merged into a single group.
- The lines connecting the points show how they are progressively merged based on similarity.
- The height at which they are connected shows how similar the points are to each other; the shorter the line, the more similar they are

## Types of Hierarchical Clustering

Now that we understand the basics of hierarchical clustering, let’s explore the two main types of hierarchical clustering.

1. Agglomerative Clustering
2. Divisive clustering

### **Hierarchical Agglomerative Clustering**

It is also known as the **bottom-up approach** or **hierarchical agglomerative clustering (HAC)**. Unlike flat clustering hierarchical clustering provides a structured way to group data. This clustering algorithm does not require us to prespecify the number of clusters. Bottom-up algorithms treat each data as a singleton cluster at the outset and then successively agglomerate pairs of clusters until all clusters have been merged into a single cluster that contains all data.

![Hierarchical Agglomerative Clustering - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20200204181551/Untitled-Diagram71.png)

Hierarchical Agglomerative Clustering

### **Workflow for Hierarchical Agglomerative clustering**

1. **Start with individual points**: Each data point is its own cluster. For example if you have 5 data points you start with 5 clusters each containing just one data point.
2. **Calculate distances between clusters**: Calculate the distance between every pair of clusters. Initially since each cluster has one point this is the distance between the two data points.
3. **Merge the closest clusters**: Identify the two clusters with the smallest distance and merge them into a single cluster.
4. **Update distance matrix**: After merging you now have one less cluster. Recalculate the distances between the new cluster and the remaining clusters.
5. **Repeat steps 3 and 4**: Keep merging the closest clusters and updating the distance matrix until you have only one cluster left.
6. **Create a dendrogram**: As the process continues you can visualize the merging of clusters using a tree-like diagram called a **dendrogram**. It shows the hierarchy of how clusters are merged.

### Python implementation of the above algorithm using the scikit-learn library:

Python`
from sklearn.cluster import AgglomerativeClustering
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],\
              [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering(n_clusters=2).fit(X)
print(clustering.labels_)
`

**Output :**

> \[1, 1, 1, 0, 0, 0\]

### **Hierarchical Divisive clustering**

It is also known as a **top-down approach**. This algorithm also does not require to prespecify the number of clusters. Top-down clustering requires a method for splitting a cluster that contains the whole data and proceeds by splitting clusters recursively until individual data have been split into singleton clusters.

### **Workflow for Hierarchical Divisive clustering :**

1. **Start with all data points in one cluster**: Treat the entire dataset as a single large cluster.
2. **Split the cluster**: Divide the cluster into two smaller clusters. The division is typically done by finding the two most dissimilar points in the cluster and using them to separate the data into two parts.
3. **Repeat the process**: For each of the new clusters, repeat the splitting process:
1. Choose the cluster with the most dissimilar points.
2. Split it again into two smaller clusters.
4. **Stop when each data point is in its own cluster**: Continue this process until every data point is its own cluster, or the stopping condition (such as a predefined number of clusters) is met.

![Hierarchical Divisive clustering - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20200204181916/Untitled-Diagram-153-1.png)

Hierarchical Divisive clustering

### **Computing Distance Matrix**

While merging two clusters we **check the distance between two every pair of clusters and merge the pair with the least distance/most similarity**. But the question is how is that distance determined. There are different ways of defining Inter Cluster distance/similarity. Some of them are:

1. **Min Distance:** Find the minimum distance between any two points of the cluster.
2. **Max Distance:** Find the maximum distance between any two points of the cluster.
3. **Group Average:** Find the average distance between every two points of the clusters.
4. **Ward’s Method:** The similarity of two clusters is based on the increase in squared error when two clusters are merged.

![Distance Matrix Comparision in Hierarchical Clustering - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230427165259/Distance-Matrix-in-Hierarchical--Clustering.webp)

Distance Matrix Comparision in Hierarchical Clustering

## Implementations code for Distance Matrix Comparision

Python`
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
X = np.array([[1, 2], [1, 4], [1, 0],\
              [4, 2], [4, 4], [4, 0]])
Z = linkage(X, 'ward') # Ward Distance
dendrogram(Z) #plotting the dendogram
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
plt.show()
`

**Output**:

![Hierarchical Clustering Dendrogram - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230425190545/download-(10).png)

Hierarchical Clustering Dendrogram

Hierarchical clustering is a powerful unsupervised learning technique that organizes data into a tree-like structure allowing us to visualize relationships between data points using a dendrogram. Unlike flat clustering methods it does not require a predefined number of clusters and provides a structured way to explore data similarity.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/)

[Implementing Agglomerative Clustering using Sklearn](https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/)

[D](https://www.geeksforgeeks.org/user/Debomit%20Dey/)

[Debomit Dey](https://www.geeksforgeeks.org/user/Debomit%20Dey/)

Follow

25

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

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


Like25

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/hierarchical-clustering/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1051254012.1745056603&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=568623959)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056603637&cv=11&fst=1745056603637&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fhierarchical-clustering%2F&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Hierarchical%20Clustering%20in%20Machine%20Learning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1258467562.1745056604&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)