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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/machine-learning-algorithms/?type%3Darticle%26id%3D1048517&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Top 15 Machine Learning Algorithms Every Data Scientist Should Know in 2025\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/)

# Machine Learning Algorithms

Last Updated : 18 Dec, 2024

Comments

Improve

Suggest changes

33 Likes

Like

Report

Machine learning algorithms are essentially sets of instructions that allow computers to learn from data, make predictions, and improve their performance over time without being explicitly programmed. Machine learning algorithms are broadly categorized into three types:

- **Supervised Learning**: Algorithms learn from labeled data, where the input-output relationship is known.
- **Unsupervised Learning**: Algorithms work with unlabeled data to identify patterns or groupings.
- **Reinforcement Learning:** Algorithms learn by interacting with an environment and receiving feedback in the form of rewards or penalties.

![Machine-Learning-Algorithms1-(1)](https://media.geeksforgeeks.org/wp-content/uploads/20230808130011/Machine-Learning-Algorithms1-(1).webp)

## Supervised Learning Algorithms

[Supervised learning](https://www.geeksforgeeks.org/supervised-machine-learning/) algos are trained on datasets where each example is paired with a target or response variable, **known as the label.** The goal is to learn a mapping function from input data to the corresponding output labels, enabling the model to make accurate predictions on unseen data. Supervised learning problems are generally categorized into **two main types:** [**Classification**](https://www.geeksforgeeks.org/getting-started-with-classification/) and [**Regression**](https://www.geeksforgeeks.org/regression-in-machine-learning/) **. M** ost widely used supervised learning algorithms are:

### 1\. Linear Regression

[Linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) is used to predict a continuous value by finding **the best-fit straight line between input (independent variable) and output (dependent variable)**

- Minimizes the difference between actual values and predicted values using a method called “ [least squares](https://www.geeksforgeeks.org/least-square-method/)” to to best fit the data.
- Predicting a person’s weight based on their height or predicting house prices based on size.

### **2\. Logistic Regression**

[Logistic regression](https://www.geeksforgeeks.org/understanding-logistic-regression/) predicts probabilities and assigns data points to binary classes (e.g., spam or not spam).

- It uses a logistic function (S-shaped curve) to model the relationship between input features and class probabilities.
- Used for classification tasks (binary or multi-class).
- Outputs probabilities to classify data into categories.
- **Example :** Predicting whether a customer will buy a product online (yes/no) or diagnosing if a person has a disease (sick/not sick).

> **Note : Despite its name, logistic regression is used for classification tasks, not regression.**

### **3\. Decision Trees**

A [decision tree](https://www.geeksforgeeks.org/decision-tree-introduction-example/) **splits data into branches based on feature values, creating a tree-like structure**.

- Each decision node represents a feature; leaf nodes provide the final prediction.
- The process continues until a final prediction is made at the leaf nodes
- Works for both classification and regression tasks.

For more decision tree algorithms, you can explore:

- [Iterative Dichotomiser 3 (ID3) Algorithms](https://www.geeksforgeeks.org/iterative-dichotomiser-3-id3-algorithm-from-scratch/)
- [C5. Algorithms](https://www.geeksforgeeks.org/c5-0-algorithm-of-decision-tree/)
- [Classification and Regression Trees Algorithms](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/)

### **4\. Support Vector Machines (SVM)**

[SVMs](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) find the best boundary (called a hyperplane) that separates data points into different classes.

- Uses support vectors (critical data points) to define the hyperplane.
- Can handle linear and non-linear problems using [kernel functions](https://www.geeksforgeeks.org/major-kernel-functions-in-support-vector-machine-svm/).
- focuses on maximizing the [margin between classes](https://www.geeksforgeeks.org/using-a-hard-margin-vs-soft-margin-in-svm/), making it robust for high-dimensional data or complex patterns.

### **5\. k-Nearest Neighbors (k-NN)**

[KNN](https://www.geeksforgeeks.org/k-nearest-neighbours/) is a simple algorithm that predicts the output for a new data point based on the similarity (distance) to its nearest neighbors in the training dataset, used for both classification and regression tasks.

- Calculates distance between point with existing data points in training dataset using a [distance metric](https://www.geeksforgeeks.org/how-to-choose-the-right-distance-metric-in-knn/) (e.g., Euclidean, Manhattan, Minkowski)
- identifies k nearest neighbors to new data point based on the calculated distances.
  - For **classification**, algorithm assigns class label that is most common among its k nearest neighbors.
  - For **regression**, the algorithm predicts the value as the average of the values of its k nearest neighbors.

### **6\. Naive Bayes**

Based on [Bayes’ theorem](https://www.geeksforgeeks.org/bayes-theorem/) and assumes all features are independent of each other (hence “naive”)

- Calculates probabilities for each class and assigns the most likely class to a data point.
- Assumption of feature independence might not hold in all cases ( rarely true in real-world data )
- Works well for high-dimensional data.
- Commonly used in text classification tasks like spam filtering : [Naive Bayes](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

### **7\. Random Forest**

[Random forest](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/) is an ensemble method that combines **multiple decision trees.**

- Uses random sampling and feature selection for diversity among trees.
- Final prediction is based on majority voting (classification) or averaging (regression).
- **Advantages**: reduces overfitting compared to individual decision trees.
- Handles large datasets with higher dimensionality.

> **For in-depth understanding :** [**What is Ensemble Learning?**](https://www.geeksforgeeks.org/ensemble-classifier-data-mining/) **–** [**Two types of ensemble methods in ML**](https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/)

### **7\. Gradient Boosting (e.g., XGBoost, LightGBM, CatBoost)**

These algorithms build models sequentially, meaning **each new model corrects errors made by previous ones.** Combines weak learners (like decision trees) to create a strong predictive model.Effective for both regression and classification tasks. : [Gradient Boosting in ML](https://www.geeksforgeeks.org/ml-gradient-boosting/)

- [XGBoost (Extreme Gradient Boosting)](https://www.geeksforgeeks.org/xgboost/) : Advanced version of Gradient Boosting that includes regularization to prevent overfitting. Faster than traditional Gradient Boosting, for large datasets.
- [LightGBM (Light Gradient Boosting Machine)](https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/): Uses a histogram-based approach for faster computation and supports categorical features natively.
- [CatBoost:](https://www.geeksforgeeks.org/catboost-ml/) Designed specifically for categorical data, with built-in encoding techniques. Uses symmetric trees for faster training and better generalization.

For more ensemble learning and gradient boosting approaches, explore:

- [AdaBoost](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/)
- [Stacking](https://www.geeksforgeeks.org/stacking-in-machine-learning/) – ensemble learning

### **8\. Neural Networks ( Including Multilayer Perceptron)**

[Neural Networks](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/), including Multilayer Perceptrons (MLPs), are considered part of supervised machine learning algorithms **as they require labeled data to train and learn the relationship between input and desired output;** network learns to minimize the error **using** [**backpropagation algorithm**](https://www.geeksforgeeks.org/backpropagation-in-neural-network/) **to adjust weights during training.**

- **Multilayer Perceptron (MLP):** Neural network with multiple layers of nodes.
- Used for both classification and regression ( **Examples:** image classification, spam detection, and predicting numerical values like stock prices or house prices)

> For in-depth understanding : [Supervised multi-layer perceptron model](https://www.geeksforgeeks.org/multi-layer-perceptron-a-supervised-neural-network-model-using-sklearn/) – [What is perceptron?](https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/)

## Unsupervised Learning Algorithms

[Unsupervised learning algos](https://www.geeksforgeeks.org/unsupervised-learning/) works with **unlabeled data** to discover hidden patterns or structures without predefined outputs. These are again divided into **three main categories** based on their purpose: [**Clustering**](https://www.geeksforgeeks.org/clustering-in-machine-learning/), [**Association Rule Mining**](https://www.geeksforgeeks.org/association-rule/), and [**Dimensionality Reduction**](https://www.geeksforgeeks.org/dimensionality-reduction/). **First we’ll see algorithms for Clustering, then dimensionality reduction and at last association.**

### **1\. Clustering**

Clustering algorithms group data points into clusters based on their similarities or differences. The goal is to identify natural groupings in the data. Clustering algorithms are divided into **multiple types based on the methods they use to group data**. These types include **Centroid-based methods**, **Distribution-based methods**, **Connectivity-based methods**, and **Density-based methods**. For resources and in-depth understanding, go through the links below.

- **Centroid-based Methods: R** epresent clusters using central points, such as centroids or medoids.
  - [K-Means clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/): Divides data into k clusters by iteratively assigning points to nearest centers, assuming spherical clusters.
  - [K-Means++ clustering](https://www.geeksforgeeks.org/ml-k-means-algorithm/)
  - [K-Mode clustering](https://www.geeksforgeeks.org/k-mode-clustering-in-python/)
  - [Fuzzy C-Means (FCM) Clustering](https://www.geeksforgeeks.org/fuzzy-c-means-clustering-in-matlab/)
- **Distribution-based Methods**
  - [Gaussian mixture models (GMMs)](https://www.geeksforgeeks.org/gaussian-mixture-model/) : Models clusters as overlapping Gaussian distributions, assigning probabilities for data points’ cluster membership.
  - [Expectation-Maximization Algorithms](https://www.geeksforgeeks.org/ml-expectation-maximization-algorithm/)
  - [Dirichlet process mixture models (DPMMs)](https://www.geeksforgeeks.org/dirichlet-process-mixture-models-dpmms/)
- **Connectivity based methods**
  - [Hierarchical clustering](https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/) : Builds a tree-like structure (dendrogram) by merging or splitting clusters, no predefined number.
    - [Agglomerative Clustering](https://www.geeksforgeeks.org/implementing-agglomerative-clustering-using-sklearn/)
    - [Divisive clustering](https://www.geeksforgeeks.org/difference-between-agglomerative-clustering-and-divisive-clustering/)
  - [Affinity propagation](https://www.geeksforgeeks.org/affinity-propagation-in-ml-to-find-the-number-of-clusters/)
- **Density Based methods**
  - [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/) : Forms clusters based on density, allowing arbitrary shapes and detecting outliers, with distance and point parameters.
  - [OPTICS (Ordering Points To Identify the Clustering Structure)](https://www.geeksforgeeks.org/ml-optics-clustering-explanation/)

### 2\. Dimensionality Reduction

[Dimensionality reduction](https://www.geeksforgeeks.org/dimensionality-reduction/) is used to simplify datasets by reducing the number of features while retaining the most important information.

- [Principal Component Analysis (PCA)](https://www.geeksforgeeks.org/principal-component-analysis-pca/): Transforms data into a new set of orthogonal features (principal components) that capture the maximum variance.
- [t-distributed Stochastic Neighbor Embedding (t-SNE)](https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/) **:** Reduces dimensions for visualizing high-dimensional data, preserving local relationships.
- [Non-negative Matrix Factorization (NMF)](https://www.geeksforgeeks.org/non-negative-matrix-factorization/) : Factorizes data into non-negative components, useful for sparse data like text or images.
- [Independent Component Analysis (ICA)](https://www.geeksforgeeks.org/ml-independent-component-analysis/)
- [Isomap](https://www.geeksforgeeks.org/isomap-a-non-linear-dimensionality-reduction-technique/) : Preserves geodesic distances to capture non-linear structures in data.
- [Locally Linear Embedding (LLE)](https://www.geeksforgeeks.org/swiss-roll-reduction-with-lle-in-scikit-learn/) : Preserves local relationships by reconstructing data points from their neighbors.
- [Latent Semantic Analysis (LSA)](https://www.geeksforgeeks.org/latent-semantic-analysis/) : Reduces the dimensionality of text data, revealing hidden patterns.
- [Autoencoders](https://www.geeksforgeeks.org/auto-encoders/) : Neural networks that compress and reconstruct data, useful for feature learning and anomaly detection.

### 3\. Association Rule

Find patterns (called association rules) between items in large datasets, typically in [market basket analysis](https://www.geeksforgeeks.org/market-basket-analysis-in-data-mining/) (e.g., finding that people who buy bread often buy butter). It identifies patterns based solely on the frequency of item occurrences and co-occurrences in the dataset.

- [Apriori algorithm](https://www.geeksforgeeks.org/apriori-algorithm/) : Finds frequent itemsets by iterating through data and pruning non-frequent item combinations.
- [FP-Growth (Frequent Pattern-Growth)](https://www.geeksforgeeks.org/ml-frequent-pattern-growth-algorithm/) : Efficiently mines frequent itemsets using a compressed **FP-tree** structure without candidate generation.
- [ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)](https://www.geeksforgeeks.org/ml-eclat-algorithm/) : Uses vertical data format for faster frequent pattern discovery through efficient intersection of itemsets.

## Reinforcement Learning Algorithms

[Reinforcement learning](https://www.geeksforgeeks.org/what-is-reinforcement-learning/) involves training agents to make a sequence of decisions by rewarding them for good actions and penalizing them for bad ones. Broadly categorized into **Model-Based** and **Model-Free** methods, these approaches differ in how they interact with the environment.

### 1\. **Model-Based Methods**

These methods use a model of the environment to predict outcomes and help the agent plan actions by simulating potential results.

- [Markov decision processes (MDPs)](https://www.geeksforgeeks.org/markov-decision-process/)
- [Bellman equation](https://www.geeksforgeeks.org/bellman-equation/)
- Value iteration algorithm
- [Monte Carlo Tree Search](https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/)

### **2\. Model-Free Methods**

These methods do not build or rely on an explicit model of the environment. Instead, the agent learns directly from experience by interacting with the environment and adjusting its actions based on feedback. Model-Free methods can be further divided into **Value-Based** and **Policy-Based** methods:

- **Value-Based Methods: F** ocus on learning the value of different states or actions, where the agent estimates the expected return from each action and selects the one with the highest value.
  - [Q-Learning](https://www.geeksforgeeks.org/q-learning-in-python/)
  - [SARSA](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)
  - [Monte Carlo Methods](https://www.geeksforgeeks.org/monte-carlo-integration-in-python/)
- **Policy-based Methods:** Directly learn a policy (a mapping from states to actions) without estimating valueswhere the agent continuously adjusts its policy to maximize rewards.
  - [REINFORCE Algorithm](https://www.geeksforgeeks.org/reinforce-algorithm/)
  - [Actor-Critic Algorithm](https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/)
  - [Asynchronous Advantage Actor-Critic (A3C)](https://www.geeksforgeeks.org/asynchronous-advantage-actor-critic-a3c-algorithm/)

> Discover the [Top 15 Machine Learning Algorithms](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/) for Interview Preparation.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Overview of Machine Learning

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/)

[Top 15 Machine Learning Algorithms Every Data Scientist Should Know in 2025](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/)

[![author](https://media.geeksforgeeks.org/auth/profile/sb7ciorr5k5t22woqkes)](https://www.geeksforgeeks.org/user/kartik/)

[kartik](https://www.geeksforgeeks.org/user/kartik/)

Follow

33

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [ML Algorithms](https://www.geeksforgeeks.org/tag/ml-algorithms/)

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


Like33

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/machine-learning-algorithms/?utm_source=geeksforgeeks&utm_medium=gfgcontent_shm&utm_campaign=shm)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1510341990.1745055424&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=374220079)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055423980&cv=11&fst=1745055423980&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3h1v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fmachine-learning-algorithms%2F%3Futm_source%3Dgeeksforgeeks%26utm_medium%3Dgfgcontent_shm%26utm_campaign%3Dshm&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Machine%20Learning%20Algorithms%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=886357251.1745055424&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=ph5yowjciwee)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=4i6tehf08h49)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=invisible&cb=tkkzocmkotxm)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)