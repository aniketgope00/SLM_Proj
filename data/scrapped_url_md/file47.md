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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/?type%3Darticle%26id%3D490767&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Open AI GPT-3\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/open-ai-gpt-3/)

# Implementation of Logistic Regression from Scratch using Python

Last Updated : 25 Oct, 2020

Comments

Improve

Suggest changes

Like Article

Like

Report

#### Introduction:

Logistic Regression is a supervised learning algorithm that is used when the target variable is categorical. Hypothetical function h(x) of linear regression predicts unbounded values. But in the case of Logistic Regression, where the target variable is categorical we have to strict the range of predicted values. Consider a classification problem, where we need to classify whether an email is a spam or not. So, the hypothetical function of linear regression could not be used here to predict as it predicts unbound values, but we have to predict either 0 or 1.

To do, so we apply the sigmoid activation function on the hypothetical function of linear regression. So the resultant hypothetical function for logistic regression is given below :

```
h( x ) = sigmoid( wx + b )

Here, w is the weight vector.
x is the feature vector.
b is the bias.

sigmoid( z ) = 1 / ( 1 + e( - z ) )

```

#### Mathematical Intuition:

The cost function of linear regression ( or mean square error ) can’t be used in logistic regression because it is a non-convex function of weights. Optimizing algorithms like i.e gradient descent only converge convex function into a global minimum.

So, the simplified cost function we use :

```
J = - ylog( h(x) ) - ( 1 - y )log( 1 - h(x) )

here, y is the real target value

h( x ) = sigmoid( wx + b )

For y = 0,

J = - log( 1 - h(x) )

and y = 1,

J = - log( h(x) )

```

This cost function is because when we train, we need to maximize the probability by minimizing the loss function.

**Gradient Descent Calculation:**

```
repeat until convergence  {
       tmpi = wi - alpha * dwi
       wi = tmpi
}
where alpha is the learning rate.

```

The chain rule is used to calculate the gradients like i.e dw.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200915115103/chainrule.png)

Chain rule for dw

```
here, a = sigmoid( z ) and z = wx + b.

```

#### Implementation:

Diabetes Dataset used in this implementation can be downloaded from [link](https://github.com/mohit-baliyan/References).

It has 8 features columns like i.e “ _Age_“, “ _Glucose_” e.t.c, and the target variable “Outcome” for 108 patients. So in this, we will train a Logistic Regression Classifier model to predict the presence of diabetes or not for patients with such information.

|     |
| --- |
| `# Importing libraries `<br>`import` `numpy as np `<br>`import` `pandas as pd `<br>`from` `sklearn.model_selection ` `import` `train_test_split `<br>`import` `warnings `<br>`warnings.filterwarnings( ` `"ignore"` `) `<br>` `<br>`# to compare our model's accuracy with sklearn model `<br>`from` `sklearn.linear_model ` `import` `LogisticRegression `<br>`# Logistic Regression `<br>`class` `LogitRegression() : `<br>`    ``def` `__init__( ` `self` `, learning_rate, iterations ) :         `<br>`        ``self` `.learning_rate ` `=` `learning_rate         `<br>`        ``self` `.iterations ` `=` `iterations `<br>`         `<br>`    ``# Function for model training     `<br>`    ``def` `fit( ` `self` `, X, Y ) :         `<br>`        ``# no_of_training_examples, no_of_features         `<br>`        ``self` `.m, ` `self` `.n ` `=` `X.shape         `<br>`        ``# weight initialization         `<br>`        ``self` `.W ` `=` `np.zeros( ` `self` `.n )         `<br>`        ``self` `.b ` `=` `0`<br>`        ``self` `.X ` `=` `X         `<br>`        ``self` `.Y ` `=` `Y `<br>`         `<br>`        ``# gradient descent learning `<br>`                 `<br>`        ``for` `i ` `in` `range` `( ` `self` `.iterations ) :             `<br>`            ``self` `.update_weights()             `<br>`        ``return` `self`<br>`     `<br>`    ``# Helper function to update weights in gradient descent `<br>`     `<br>`    ``def` `update_weights( ` `self` `) :            `<br>`        ``A ` `=` `1` `/` `( ` `1` `+` `np.exp( ` `-` `( ` `self` `.X.dot( ` `self` `.W ) ` `+` `self` `.b ) ) ) `<br>`         `<br>`        ``# calculate gradients         `<br>`        ``tmp ` `=` `( A ` `-` `self` `.Y.T )         `<br>`        ``tmp ` `=` `np.reshape( tmp, ` `self` `.m )         `<br>`        ``dW ` `=` `np.dot( ` `self` `.X.T, tmp ) ` `/` `self` `.m          `<br>`        ``db ` `=` `np.` `sum` `( tmp ) ` `/` `self` `.m  `<br>`         `<br>`        ``# update weights     `<br>`        ``self` `.W ` `=` `self` `.W ` `-` `self` `.learning_rate ` `*` `dW     `<br>`        ``self` `.b ` `=` `self` `.b ` `-` `self` `.learning_rate ` `*` `db `<br>`         `<br>`        ``return` `self`<br>`     `<br>`    ``# Hypothetical function  h( x )  `<br>`     `<br>`    ``def` `predict( ` `self` `, X ) :     `<br>`        ``Z ` `=` `1` `/` `( ` `1` `+` `np.exp( ` `-` `( X.dot( ` `self` `.W ) ` `+` `self` `.b ) ) )         `<br>`        ``Y ` `=` `np.where( Z > ` `0.5` `, ` `1` `, ` `0` `)         `<br>`        ``return` `Y `<br>` `<br>` `<br>`# Driver code `<br>` `<br>`def` `main() : `<br>`     `<br>`    ``# Importing dataset     `<br>`    ``df ` `=` `pd.read_csv( ` `"diabetes.csv"` `) `<br>`    ``X ` `=` `df.iloc[:,:` `-` `1` `].values `<br>`    ``Y ` `=` `df.iloc[:,` `-` `1` `:].values `<br>`     `<br>`    ``# Splitting dataset into train and test set `<br>`    ``X_train, X_test, Y_train, Y_test ` `=` `train_test_split( `<br>`      ``X, Y, test_size ` `=` `1` `/` `3` `, random_state ` `=` `0` `) `<br>`     `<br>`    ``# Model training     `<br>`    ``model ` `=` `LogitRegression( learning_rate ` `=` `0.01` `, iterations ` `=` `1000` `) `<br>`     `<br>`    ``model.fit( X_train, Y_train )     `<br>`    ``model1 ` `=` `LogisticRegression()     `<br>`    ``model1.fit( X_train, Y_train) `<br>`     `<br>`    ``# Prediction on test set `<br>`    ``Y_pred ` `=` `model.predict( X_test )     `<br>`    ``Y_pred1 ` `=` `model1.predict( X_test ) `<br>`     `<br>`    ``# measure performance     `<br>`    ``correctly_classified ` `=` `0`<br>`    ``correctly_classified1 ` `=` `0`<br>`     `<br>`    ``# counter     `<br>`    ``count ` `=` `0`<br>`    ``for` `count ` `in` `range` `( np.size( Y_pred ) ) :   `<br>`       `<br>`        ``if` `Y_test[count] ` `=` `=` `Y_pred[count] :             `<br>`            ``correctly_classified ` `=` `correctly_classified ` `+` `1`<br>`         `<br>`        ``if` `Y_test[count] ` `=` `=` `Y_pred1[count] :             `<br>`            ``correctly_classified1 ` `=` `correctly_classified1 ` `+` `1`<br>`             `<br>`        ``count ` `=` `count ` `+` `1`<br>`         `<br>`    ``print` `( ` `"Accuracy on test set by our model       :  "` `, (  `<br>`      ``correctly_classified ` `/` `count ) ` `*` `100` `) `<br>`    ``print` `( ` `"Accuracy on test set by sklearn model   :  "` `, (  `<br>`      ``correctly_classified1 ` `/` `count ) ` `*` `100` `) `<br>` `<br>` `<br>`if` `__name__ ` `=` `=` `"__main__"` `:      `<br>`    ``main()` |

```

```

```

```

#### Output :

```
Accuracy on test set by our model       :   58.333333333333336
Accuracy on test set by sklearn model   :   61.111111111111114

```

**Note:** The above-trained model is to implement the mathematical intuition not just for improving accuracies.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/open-ai-gpt-3/)

[Open AI GPT-3](https://www.geeksforgeeks.org/open-ai-gpt-3/)

[![author](https://media.geeksforgeeks.org/auth/profile/ftl0h451b52f66f7iiya)](https://www.geeksforgeeks.org/user/mohitbaliyan/)

[mohitbaliyan](https://www.geeksforgeeks.org/user/mohitbaliyan/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Implementation of Lasso Regression From Scratch using Python\\
\\
\\
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that combines prediction with feature selection. It does this by adding a penalty term to the cost function shrinking less relevant feature's coefficients to zero. This makes it effective for high-dim\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/)
[Linear Regression Implementation From Scratch using Python\\
\\
\\
Linear Regression is a supervised learning algorithm which is both a statistical and a machine learning algorithm. It is used to predict the real-valued output y based on the given input value x. It depicts the relationship between the dependent variable y and the independent variables xi ( or featu\\
\\
4 min read](https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/)
[Polynomial Regression ( From Scratch using Python )\\
\\
\\
Prerequisites Linear RegressionGradient DescentIntroductionLinear Regression finds the correlation between the dependent variable ( or target variable ) and independent variables ( or features ). In short, it is a linear model to fit the data linearly. But it fails to fit and catch the pattern in no\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/)
[ML \| Naive Bayes Scratch Implementation using Python\\
\\
\\
Naive Bayes is a probabilistic machine learning algorithms based on the Bayes Theorem. It is a simple yet powerful algorithm because of its understanding, simplicity and ease of implementation. It is popular method for classification applications such as spam filtering and text classification. In th\\
\\
7 min read](https://www.geeksforgeeks.org/ml-naive-bayes-scratch-implementation-using-python/)
[Implementing SVM from Scratch in Python\\
\\
\\
Support Vector Machines (SVMs) are powerful supervised machine learning algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that separates data points of different classes with the maximum margin. We can use Scikit library of python to implement SVM b\\
\\
4 min read](https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/)
[Linear Regression (Python Implementation)\\
\\
\\
Linear regression is a statistical method that is used to predict a continuous dependent variable i.e target variable based on one or more independent variables. This technique assumes a linear relationship between the dependent and independent variables which means the dependent variable changes pr\\
\\
14 min read](https://www.geeksforgeeks.org/linear-regression-python-implementation/)
[Implementation of neural network from scratch using NumPy\\
\\
\\
Neural networks are a core component of deep learning models, and implementing them from scratch is a great way to understand their inner workings. we will demonstrate how to implement a basic Neural networks algorithm from scratch using the NumPy library in Python, focusing on building a three-lett\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/)
[Logistic Regression using Python\\
\\
\\
A basic machine learning approach that is frequently used for binary classification tasks is called logistic regression. Though its name suggests otherwise, it uses the sigmoid function to simulate the likelihood of an instance falling into a specific class, producing values between 0 and 1. Logisti\\
\\
8 min read](https://www.geeksforgeeks.org/ml-logistic-regression-using-python/)
[Text Classification using Logistic Regression\\
\\
\\
Text classification is a fundamental task in Natural Language Processing (NLP) that involves assigning predefined categories or labels to textual data. It has a wide range of applications, including spam detection, sentiment analysis, topic categorization, and language identification. Logistic Regre\\
\\
4 min read](https://www.geeksforgeeks.org/text-classification-using-logistic-regression/)
[Logistic Regression using PySpark Python\\
\\
\\
In this tutorial series, we are going to cover Logistic Regression using Pyspark. Logistic Regression is one of the basic ways to perform classification (donâ€™t be confused by the word â€œregressionâ€). Logistic Regression is a classification method. Some examples of classification are: Spam detectionDi\\
\\
3 min read](https://www.geeksforgeeks.org/logistic-regression-using-pyspark-python/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/)

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