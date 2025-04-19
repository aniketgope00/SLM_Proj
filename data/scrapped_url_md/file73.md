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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/xgboost/?type%3Darticle%26id%3D654191&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
CatBoost in Machine Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/catboost-ml/)

# XGBoost

Last Updated : 02 Feb, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Traditional machine learning models like decision trees and random forests are easy to interpret but often struggle with accuracy on complex datasets. XGBoost, short for **eXtreme Gradient Boosting**, is an advanced machine learning algorithm designed for efficiency, speed, and high performance.

## What is XGBoost?

XGBoost is an optimized implementation of [**Gradient Boosting**](https://www.geeksforgeeks.org/ml-gradient-boosting/) and is a type of [**ensemble learning**](https://www.geeksforgeeks.org/a-comprehensive-guide-to-ensemble-learning/) method. Ensemble learning combines multiple weak models to form a stronger model.

- XGBoost uses [**decision trees**](https://www.geeksforgeeks.org/decision-tree/) as its base learners combining them sequentially to improve the model’s performance. Each new tree is trained to correct the errors made by the previous tree and this process is called [boosting](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/).
- It has **built-in parallel processing to train models on large datasets quickly**. XGBoost also supports customizations allowing users to adjust model parameters to optimize performance based on the specific problem.

In this article, we will explore XGBoost step by step covering its core concepts.

## **How XGBoost Works?**

It builds decision trees sequentially with each tree attempting to correct the mistakes made by the previous one. The process can be broken down as follows:

1. **Start with a base learner**: The first model decision tree is trained on the data. In regression tasks this base model simply predict the average of the target variable.
2. **Calculate the errors**: After training the first tree the errors between the predicted and actual values are calculated.
3. **Train the next tree**: The next tree is trained on the errors of the previous tree. This step attempts to correct the errors made by the first tree.
4. **Repeat the process**: This process continues with each new tree trying to correct the errors of the previous trees until a stopping criterion is met.
5. **Combine the predictions**: The final prediction is the sum of the predictions from all the trees.

## Maths Behind XGBoost ALgorithm

It can be viewed as iterative process where we start with an initial prediction often set to zero. After which each tree is added to reduce errors. Mathematically, the model can be represented as:

y^i=∑k=1Kfk(xi)\\hat{y}\_{i} = \\sum\_{k=1}^{K} f\_k(x\_i)y^​i​=∑k=1K​fk​(xi​)

Where y^i\\hat{y}\_{i} y^​i​ is the final predicted value for the ith data point, K is the number of trees in the ensemble and fk(xi) f\_k(x\_i)fk​(xi​) represents the prediction of the K th tree for the ith data point.

The objective function in XGBoost consists of two parts: a **loss function** and a **regularization term**. The loss function measures how well the model fits the data and the regularization term simplify complex trees. The general form of the loss function is:

obj(θ)=∑inl(yi,y^i)+∑k=1KΩ(fk)obj(\\theta) = \\sum\_{i}^{n} l(y\_{i}, \\hat{y}\_{i}) + \\sum\_{k=1}^K \\Omega(f\_{k}) \\\obj(θ)=∑in​l(yi​,y^​i​)+∑k=1K​Ω(fk​)

Where:

- l(yi,y^i) l(y\_{i}, \\hat{y}\_{i}) l(yi​,y^​i​) is the loss function which computes the difference between the true value yiy\_iyi​ and the predicted value y^i\\hat{y}\_iy^​i​,
- Ω(fk) \\Omega(f\_{k}) \\\Ω(fk​) is the regularization term which discourages overly complex trees.

Now, instead of fitting the model all at once we optimize the model iteratively. We start with an initial prediction y^i(0)=0\\hat{y}\_i^{(0)} =0y^​i(0)​=0 and at each step we add a new tree to improve the model. The updated predictions after adding the tth tree can be written as:

y^i(t)=y^i(t−1)+ft(xi)\\\ \\hat{y}\_i^{(t)} = \\hat{y}\_i^{(t-1)} + f\_t(x\_i)y^​i(t)​=y^​i(t−1)​+ft​(xi​)

Where y^i(t−1) \\hat{y}\_i^{(t-1)} y^​i(t−1)​ is the prediction from the previous iteration and ft(xi) f\_t(x\_i)ft​(xi​) is the prediction of the tth tree for the ith data point.

The **regularization term** Ω(ft)\\Omega(f\_t) Ω(ft​) simplify complex trees by penalizing the number of leaves in the tree and the size of the leaf. It is defined as:

Ω(ft)=γT+12λ∑j=1Twj2\\Omega(f\_t) = \\gamma T + \\frac{1}{2}\\lambda \\sum\_{j=1}^T w\_j^2Ω(ft​)=γT+21​λ∑j=1T​wj2​

Where:

- T\\TauT is the number of leaves in the tree
- γ\\gammaγ is a regularization parameter that controls the complexity of the tree
- λ\\lambdaλ is a parameter that penalizes the squared weight of the leaves wjw\_jwj​​

Finally when deciding how to split the nodes in the tree we compute the **information gain** for every possible split. The information gain for a split is calculated as:

Gain=12\[GL2HL+λ+GR2HR+λ−(GL+GR)2HL+HR+λ\]–γGain = \\frac{1}{2} \\left\[\\frac{G\_L^2}{H\_L+\\lambda}+\\frac{G\_R^2}{H\_R+\\lambda}-\\frac{(G\_L+G\_R)^2}{H\_L+H\_R+\\lambda}\\right\] – \\gammaGain=21​\[HL​+λGL2​​+HR​+λGR2​​−HL​+HR​+λ(GL​+GR​)2​\]–γ

Where

- GL​,GR are the sums of gradients in the left and right child nodes
- HL,HR are the sums of Hessians in the left and right child nodes

By calculating the information gain for every possible split at each node XGBoost selects the split that results in the largest gain which effectively reduces the errors and improves the model’s performance.

## What Makes XGBoost “eXtreme”?

XGBoost extends traditional gradient boosting by including regularization elements in the objective function, XGBoost improves generalization and prevents overfitting.

### 1\. Preventing Overfitting

The learning rate, also known as **shrinkage**, is a new parameter introduced by XGBoost. It is represented by the symbol “ **eta**.” It quantifies each tree’s contribution to the total prediction. Because each tree has less of an influence, an optimization process with a lower learning rate is more resilient. By making the model more conservative, regularization terms combined with a low learning rate assist avoid overfitting.

XGBoost constructs trees level by level, assessing whether adding a new node (split) enhances the objective function as a whole at each level. The split is trimmed if not. This level growth along with trimming makes the trees easier to understand and easier to create.

The regularization terms, along with other techniques such as shrinkage and pruning, play a crucial role in preventing overfitting, improving generalization, and making XGBoost a robust and powerful algorithm for various machine learning tasks.

### 2\. Tree Structure

Conventional decision trees are frequently developed by expanding each branch until a stopping condition is satisfied, or in a depth-first fashion. On the other hand, XGBoost builds trees level-wise or breadth-first. This implies that it adds nodes for every feature at a certain depth before moving on to the next level, so growing the tree one level at a time.

- **Determining the Best Splits**: XGBoost assesses every split that might be made for every feature at every level and chooses the one that minimizes the objective function as much as feasible (e.g., minimizing the mean squared error for regression tasks or cross-entropy for classification tasks).

In contrast, a single feature is selected for a split at each level in depth-wise expansion.

- **Prioritizing Important Features**: The overhead involved in choosing the best split for each feature at each level is decreased by level-wise growth. XGBoost eliminates the need to revisit and assess the same feature more than once during tree construction because all features are taken into account at the same time.

This is particularly beneficial when there are complex interactions among features, as the algorithm can adapt to the intricacies of the data.

### 3\. Handling Missing Data

XGBoost functions well even with incomplete datasets because of its strong mechanism for handling missing data during training.

To effectively handle missing values, XGBoost employs a “ **Sparsity Aware Split Finding**” algorithm. The algorithm treats missing values as a separate value and assesses potential splits in accordance with them when determining the optimal split at each node. If a data point has a missing value for a particular feature during tree construction, it descends a different branch of the tree.

The potential gain from splitting the data based on the available feature values—including missing values—is taken into account by the algorithm to determine the ideal split. It computes the gain for every possible split, treating the cases where values are missing as a separate group.

If the algorithm’s path through the tree comes across a node that has missing values while generating predictions **for a new instance during inference**, it will proceed along the default branch made for instances that have missing values. This guarantees that the model can generate predictions in the event that there are missing values in the input data.

### 4\. Cache-Aware Access in XGBoost

Cache memory located closer to the CPU offers faster access times, and modern computer architectures consist of hierarchical memory systems, By making effective use of this cache hierarchy, computational performance can be greatly enhanced. This is why XGBoost’s cache-aware access was created, with the goal of reducing memory access times during the training stage.

The most frequently accessed data is always available for computations because XGBoost processes data by storing portions of the dataset in the CPU’s cache memory. This method makes use of the spatial locality principle, which states that adjacent memory locations are more likely to be accessed concurrently. Computations are sped up by XGBoost because it arranges data in a cache-friendly manner, reducing the need to fetch data from slower main memory.

### 5\. Approximate Greedy Algorithm

This algorithm uses weighted quantiles to find the optimal node split quickly rather than analyzing each possible split point in detail. When working with large datasets, XGBoost makes the algorithm more scalable and faster by approximating the optimal split, which dramatically lowers the computational cost associated with evaluating all candidate splits.

## Advantages of XGboost

- XGBoost is highly scalable and efficient as It is designed to handle large datasets with millions or even billions of instances and features.
- XGBoost implements parallel processing techniques and utilizes hardware optimization, such as GPU acceleration, to speed up the training process. This scalability and efficiency make XGBoost suitable for big data applications and real-time predictions.
- It provides a wide range of customizable parameters and regularization techniques, allowing users to fine-tune the model according to their specific needs.
- XGBoost offers built-in feature importance analysis, which helps identify the most influential features in the dataset. This information can be valuable for feature selection, dimensionality reduction, and gaining insights into the underlying data patterns.
- XGBoost has not only demonstrated exceptional performance but has also become a go-to tool for data scientists and machine learning practitioners across various languages. It has consistently outperformed other algorithms in Kaggle competitions, showcasing its effectiveness in producing high-quality predictive models.

## Disadvantages of XGBoost

- XGBoost can be computationally intensive especially when training complex models making it less suitable for resource-constrained systems.
- Despite its robustness, XGBoost can still be sensitive to noisy data or outliers, necessitating careful data preprocessing for optimal performance.
- XGBoost is prone to overfitting on small datasets or when too many trees are used in the model.
- While feature importance scores are available, the overall model can be challenging to interpret compared to simpler methods like linear regression or decision trees. This lack of transparency may be a drawback in fields like healthcare or finance where interpretability is critical.

XGBoost is a powerful and flexible tool that works well for many machine learning tasks. Its ability to handle large datasets and deliver high accuracy makes it useful.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/catboost-ml/)

[CatBoost in Machine Learning](https://www.geeksforgeeks.org/catboost-ml/)

[P](https://www.geeksforgeeks.org/user/pawangfg/)

[pawangfg](https://www.geeksforgeeks.org/user/pawangfg/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

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


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/xgboost/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1026752486.1745056441&gtm=45je54g3v884918195za200zb858768136&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=452776138)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)