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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/?type%3Darticle%26id%3D268668&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
ML \| Momentum-based Gradient Optimizer\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/)

# ML \| Mini-Batch Gradient Descent with Python

Last Updated : 02 Aug, 2022

Comments

Improve

Suggest changes

16 Likes

Like

Report

In machine learning, gradient descent is an optimization technique used for computing the model parameters (coefficients and bias) for algorithms like linear regression, logistic regression, neural networks, etc. In this technique, we repeatedly iterate through the training set and update the model parameters in accordance with the gradient of the error with respect to the training set. Depending on the number of training examples considered in updating the model parameters, we have 3-types of gradient descents:

1. **Batch Gradient Descent:** Parameters are updated after computing the gradient of the error with respect to the entire training set
2. **Stochastic Gradient Descent:** Parameters are updated after computing the gradient of the error with respect to a single training example
3. **Mini-Batch Gradient Descent:** Parameters are updated after computing the gradient of  the error with respect to a subset of the training set

| Batch Gradient Descent | Stochastic Gradient Descent | Mini-Batch Gradient Descent |
| --- | --- | --- |
| Since the entire training data is considered before taking a step in the direction of gradient, therefore it takes a lot of time for making a single update. | Since only a single training example is considered before taking a step in the direction of gradient, we are forced to loop over the training set and thus cannot exploit the speed associated with vectorizing the code. | Since a subset of training examples is considered, it can make quick updates in the model parameters and can also exploit the speed associated with vectorizing the code. |
| It makes smooth updates in the model parameters | It makes very noisy updates in the parameters | Depending upon the batch size, the updates can be made less noisy – greater the batch size less noisy is the update |

Thus, mini-batch gradient descent makes a compromise between the speedy convergence and the noise associated with gradient update which makes it a more flexible and robust algorithm.

![](https://media.geeksforgeeks.org/wp-content/uploads/20220615041457/resizedImage-660x187.png)

Convergence in BGD, SGD & MBGD

**Mini-Batch Gradient Descent:** **Algorithm-**

> Let theta = model parameters and max\_iters = number of epochs. for itr = 1, 2, 3, …, max\_iters:       for mini\_batch (X\_mini, y\_mini):
>
> - Forward Pass on the batch X\_mini:
>   - Make predictions on the mini-batch
>   - Compute error in predictions (J(theta)) with the current values of the parameters
> - Backward Pass:
>   - Compute gradient(theta) = partial derivative of J(theta) w.r.t. theta
> - Update parameters:
>   - theta = theta – learning\_rate\*gradient(theta)

**Below is the Python Implementation:**

**Step #1:** First step is to import dependencies, generate data for linear regression, and visualize the generated data. We have generated 8000 data examples, each having 2 attributes/features. These data examples are further divided into training sets (X\_train, y\_train) and testing set (X\_test, y\_test) having 7200 and 800 examples respectively.

- Python3

## Python3

|     |
| --- |
| `# importing dependencies`<br>`import` `numpy as np`<br>`import` `matplotlib.pyplot as plt`<br>`# creating data`<br>`mean ` `=` `np.array([` `5.0` `, ` `6.0` `])`<br>`cov ` `=` `np.array([[` `1.0` `, ` `0.95` `], [` `0.95` `, ` `1.2` `]])`<br>`data ` `=` `np.random.multivariate_normal(mean, cov, ` `8000` `)`<br>`# visualising data`<br>`plt.scatter(data[:` `500` `, ` `0` `], data[:` `500` `, ` `1` `], marker` `=` `'.'` `)`<br>`plt.show()`<br>`# train-test-split`<br>`data ` `=` `np.hstack((np.ones((data.shape[` `0` `], ` `1` `)), data))`<br>`split_factor ` `=` `0.90`<br>`split ` `=` `int` `(split_factor ` `*` `data.shape[` `0` `])`<br>`X_train ` `=` `data[:split, :` `-` `1` `]`<br>`y_train ` `=` `data[:split, ` `-` `1` `].reshape((` `-` `1` `, ` `1` `))`<br>`X_test ` `=` `data[split:, :` `-` `1` `]`<br>`y_test ` `=` `data[split:, ` `-` `1` `].reshape((` `-` `1` `, ` `1` `))`<br>`print` `(& quot`<br>`       ``Number of examples ` `in` `training ` `set` `=` `%` `d & quot`<br>`       ``%` `(X_train.shape[` `0` `]))`<br>`print` `(& quot`<br>`       ``Number of examples ` `in` `testing ` `set` `=` `%` `d & quot`<br>`       ``%` `(X_test.shape[` `0` `]))` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/in-1.png)

Number of examples in training set = 7200 Number of examples in testing set = 800

**Step #2:** Next, we write the code for implementing linear regression using mini-batch gradient descent. gradientDescent() is the main driver function and other functions are helper functions used for making predictions – hypothesis(), computing gradients – gradient(), computing error – cost() and creating mini-batches – create\_mini\_batches(). The driver function initializes the parameters, computes the best set of parameters for the model, and returns these parameters along with a list containing a history of errors as the parameters get updated.

**Example**

- Python3

## Python3

|     |
| --- |
| `# linear regression using "mini-batch" gradient descent`<br>`# function to compute hypothesis / predictions`<br>`def` `hypothesis(X, theta):`<br>`    ``return` `np.dot(X, theta)`<br>`# function to compute gradient of error function w.r.t. theta`<br>`def` `gradient(X, y, theta):`<br>`    ``h ` `=` `hypothesis(X, theta)`<br>`    ``grad ` `=` `np.dot(X.transpose(), (h ` `-` `y))`<br>`    ``return` `grad`<br>`# function to compute the error for current values of theta`<br>`def` `cost(X, y, theta):`<br>`    ``h ` `=` `hypothesis(X, theta)`<br>`    ``J ` `=` `np.dot((h ` `-` `y).transpose(), (h ` `-` `y))`<br>`    ``J ` `/` `=` `2`<br>`    ``return` `J[` `0` `]`<br>`# function to create a list containing mini-batches`<br>`def` `create_mini_batches(X, y, batch_size):`<br>`    ``mini_batches ` `=` `[]`<br>`    ``data ` `=` `np.hstack((X, y))`<br>`    ``np.random.shuffle(data)`<br>`    ``n_minibatches ` `=` `data.shape[` `0` `] ` `/` `/` `batch_size`<br>`    ``i ` `=` `0`<br>`    ``for` `i ` `in` `range` `(n_minibatches ` `+` `1` `):`<br>`        ``mini_batch ` `=` `data[i ` `*` `batch_size:(i ` `+` `1` `)` `*` `batch_size, :]`<br>`        ``X_mini ` `=` `mini_batch[:, :` `-` `1` `]`<br>`        ``Y_mini ` `=` `mini_batch[:, ` `-` `1` `].reshape((` `-` `1` `, ` `1` `))`<br>`        ``mini_batches.append((X_mini, Y_mini))`<br>`    ``if` `data.shape[` `0` `] ` `%` `batch_size !` `=` `0` `:`<br>`        ``mini_batch ` `=` `data[i ` `*` `batch_size:data.shape[` `0` `]]`<br>`        ``X_mini ` `=` `mini_batch[:, :` `-` `1` `]`<br>`        ``Y_mini ` `=` `mini_batch[:, ` `-` `1` `].reshape((` `-` `1` `, ` `1` `))`<br>`        ``mini_batches.append((X_mini, Y_mini))`<br>`    ``return` `mini_batches`<br>`# function to perform mini-batch gradient descent`<br>`def` `gradientDescent(X, y, learning_rate` `=` `0.001` `, batch_size` `=` `32` `):`<br>`    ``theta ` `=` `np.zeros((X.shape[` `1` `], ` `1` `))`<br>`    ``error_list ` `=` `[]`<br>`    ``max_iters ` `=` `3`<br>`    ``for` `itr ` `in` `range` `(max_iters):`<br>`        ``mini_batches ` `=` `create_mini_batches(X, y, batch_size)`<br>`        ``for` `mini_batch ` `in` `mini_batches:`<br>`            ``X_mini, y_mini ` `=` `mini_batch`<br>`            ``theta ` `=` `theta ` `-` `learning_rate ` `*` `gradient(X_mini, y_mini, theta)`<br>`            ``error_list.append(cost(X_mini, y_mini, theta))`<br>`    ``return` `theta, error_list` |

```

```

```

```

Calling the gradientDescent() function to compute the model parameters (theta) and visualize the change in the error function.

- Python3

## Python3

|     |
| --- |
| `theta, error_list ` `=` `gradientDescent(X_train, y_train)`<br>`print` `("Bias ` `=` `", theta[` `0` `])`<br>`print` `("Coefficients ` `=` `", theta[` `1` `:])`<br>`# visualising gradient descent`<br>`plt.plot(error_list)`<br>`plt.xlabel("Number of iterations")`<br>`plt.ylabel("Cost")`<br>`plt.show()` |

```

```

```

```

**Output:** Bias = \[0.81830471\] Coefficients = \[\[1.04586595\]\]

![](https://media.geeksforgeeks.org/wp-content/uploads/cost.png)

**Step #3:** Finally, we make predictions on the testing set and compute the mean absolute error in predictions.

- Python3

## Python3

|     |
| --- |
| `# predicting output for X_test`<br>`y_pred ` `=` `hypothesis(X_test, theta)`<br>`plt.scatter(X_test[:, ` `1` `], y_test[:, ], marker` `=` `'.'` `)`<br>`plt.plot(X_test[:, ` `1` `], y_pred, color` `=` `'orange'` `)`<br>`plt.show()`<br>`# calculating error in predictions`<br>`error ` `=` `np.` `sum` `(np.` `abs` `(y_test ` `-` `y_pred) ` `/` `y_test.shape[` `0` `])`<br>`print` `(& quot`<br>`       ``Mean absolute error ` `=` `& quot`<br>`       ``, error)` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/out-6.png)

> Mean absolute error = 0.4366644295854125

The orange line represents the final hypothesis function: **theta\[0\] + theta\[1\]\*X\_test\[:, 1\] + theta\[2\]\*X\_test\[:, 2\] = 0**

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/)

[ML \| Momentum-based Gradient Optimizer](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

16

Improve

Article Tags :

- [Computer Subject](https://www.geeksforgeeks.org/category/computer-subject/)
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


Like16

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=753271119.1745057054&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=1794433710)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057053903&cv=11&fst=1745057053903&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3h1v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&ptag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fml-mini-batch-gradient-descent-with-python%2F&hn=www.googleadservices.com&frm=0&tiba=ML%20%7C%20Mini-Batch%20Gradient%20Descent%20with%20Python%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1888432156.1745057054&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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