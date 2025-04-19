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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/?type%3Darticle%26id%3D485668&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Top 5 Open-Source Online Machine Learning Environments\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/top-5-open-source-online-machine-learning-environments/)

# Linear Regression Implementation From Scratch using Python

Last Updated : 01 Oct, 2020

Comments

Improve

Suggest changes

Like Article

Like

Report

Linear Regression is a supervised learning algorithm which is both a statistical and a machine learning algorithm. It is used to predict the real-valued output y based on the given input value x. It depicts the relationship between the dependent variable y and the independent variables xi  ( or features ).  The hypothetical function used for prediction is represented by h( x ).

```
  h( x ) = w * x + b

  here, b is the bias.
  x represents the feature vector
  w represents the weight vector.

```

Linear regression with one variable is also called univariant linear regression.  After initializing the weight vector, we can find the weight vector to best fit the model by ordinary least squares method or gradient descent learning.

**Mathematical Intuition:** The cost function (or loss function) is used to measure the performance of a machine learning model or quantifies the error between the expected values and the values predicted by our hypothetical function. The cost function for Linear Regression is represented by J.

![\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}-h\left(x^{(i)}\right)\right)^{2}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ee804dd2ef914445d34e803be76167a2_l3.svg)

```
Here, m is the total number of training examples in the dataset.
y(i) represents the value of target variable for ith training example.

```

So, our objective is to minimize the cost function _**J**(_ or improve the performance of our machine learning model). To do this, we have to find the weights at which _**J**_ is minimum.  One such algorithm which can be used to minimize any differentiable function is Gradient Descent. It is a first-order iterative optimizing algorithm that takes us to a minimum of a function.

#### Gradient descent:

Pseudo Code:

1. Start with some w
2. Keep changing w to reduce J( w ) until we hopefully end up at a minimum.

Algorithm:

```
repeat until convergence  {
       tmpi = wi - alpha * dwi
       wi = tmpi
}
where alpha is the learning rate.

```

#### Implementation:

Dataset used in this implementation can be downloaded from [link](https://github.com/mohit-baliyan/References).

It has 2 columns — “ _YearsExperience_” and “ _Salary_” for 30 employees in a company. So in this, we will train a Linear Regression model to learn the correlation between the number of years of experience of each employee and their respective salary. Once the model is trained, we will be able to predict the salary of an employee on the basis of his years of experience.

- Python3

## Python3

|     |
| --- |
| `# Importing libraries `<br>` `<br>`import` `numpy as np `<br>` `<br>`import` `pandas as pd `<br>` `<br>`from` `sklearn.model_selection ` `import` `train_test_split `<br>` `<br>`import` `matplotlib.pyplot as plt `<br>` `<br>`# Linear Regression `<br>` `<br>`class` `LinearRegression() : `<br>`     `<br>`    ``def` `__init__( ` `self` `, learning_rate, iterations ) : `<br>`         `<br>`        ``self` `.learning_rate ` `=` `learning_rate `<br>`         `<br>`        ``self` `.iterations ` `=` `iterations `<br>`         `<br>`    ``# Function for model training `<br>`             `<br>`    ``def` `fit( ` `self` `, X, Y ) : `<br>`         `<br>`        ``# no_of_training_examples, no_of_features `<br>`         `<br>`        ``self` `.m, ` `self` `.n ` `=` `X.shape `<br>`         `<br>`        ``# weight initialization `<br>`         `<br>`        ``self` `.W ` `=` `np.zeros( ` `self` `.n ) `<br>`         `<br>`        ``self` `.b ` `=` `0`<br>`         `<br>`        ``self` `.X ` `=` `X `<br>`         `<br>`        ``self` `.Y ` `=` `Y `<br>`         `<br>`         `<br>`        ``# gradient descent learning `<br>`                 `<br>`        ``for` `i ` `in` `range` `( ` `self` `.iterations ) : `<br>`             `<br>`            ``self` `.update_weights() `<br>`             `<br>`        ``return` `self`<br>`     `<br>`    ``# Helper function to update weights in gradient descent `<br>`     `<br>`    ``def` `update_weights( ` `self` `) : `<br>`            `<br>`        ``Y_pred ` `=` `self` `.predict( ` `self` `.X ) `<br>`         `<br>`        ``# calculate gradients   `<br>`     `<br>`        ``dW ` `=` `-` `( ` `2` `*` `( ` `self` `.X.T ).dot( ` `self` `.Y ` `-` `Y_pred )  ) ` `/` `self` `.m `<br>`      `<br>`        ``db ` `=` `-` `2` `*` `np.` `sum` `( ` `self` `.Y ` `-` `Y_pred ) ` `/` `self` `.m  `<br>`         `<br>`        ``# update weights `<br>`     `<br>`        ``self` `.W ` `=` `self` `.W ` `-` `self` `.learning_rate ` `*` `dW `<br>`     `<br>`        ``self` `.b ` `=` `self` `.b ` `-` `self` `.learning_rate ` `*` `db `<br>`         `<br>`        ``return` `self`<br>`     `<br>`    ``# Hypothetical function  h( x )  `<br>`     `<br>`    ``def` `predict( ` `self` `, X ) : `<br>`     `<br>`        ``return` `X.dot( ` `self` `.W ) ` `+` `self` `.b `<br>`    `<br>` `<br>`# driver code `<br>` `<br>`def` `main() : `<br>`     `<br>`    ``# Importing dataset `<br>`     `<br>`    ``df ` `=` `pd.read_csv( ` `"salary_data.csv"` `) `<br>` `<br>`    ``X ` `=` `df.iloc[:,:` `-` `1` `].values `<br>` `<br>`    ``Y ` `=` `df.iloc[:,` `1` `].values `<br>`     `<br>`    ``# Splitting dataset into train and test set `<br>` `<br>`    ``X_train, X_test, Y_train, Y_test ` `=` `train_test_split(  `<br>`      ``X, Y, test_size ` `=` `1` `/` `3` `, random_state ` `=` `0` `) `<br>`     `<br>`    ``# Model training `<br>`     `<br>`    ``model ` `=` `LinearRegression( iterations ` `=` `1000` `, learning_rate ` `=` `0.01` `) `<br>` `<br>`    ``model.fit( X_train, Y_train ) `<br>`     `<br>`    ``# Prediction on test set `<br>` `<br>`    ``Y_pred ` `=` `model.predict( X_test ) `<br>`     `<br>`    ``print` `( ` `"Predicted values "` `, np.` `round` `( Y_pred[:` `3` `], ` `2` `) )  `<br>`     `<br>`    ``print` `( ` `"Real values      "` `, Y_test[:` `3` `] ) `<br>`     `<br>`    ``print` `( ` `"Trained W        "` `, ` `round` `( model.W[` `0` `], ` `2` `) ) `<br>`     `<br>`    ``print` `( ` `"Trained b        "` `, ` `round` `( model.b, ` `2` `) ) `<br>`     `<br>`    ``# Visualization on test set  `<br>`     `<br>`    ``plt.scatter( X_test, Y_test, color ` `=` `'blue'` `) `<br>`     `<br>`    ``plt.plot( X_test, Y_pred, color ` `=` `'orange'` `) `<br>`     `<br>`    ``plt.title( ` `'Salary vs Experience'` `) `<br>`     `<br>`    ``plt.xlabel( ` `'Years of Experience'` `) `<br>`     `<br>`    ``plt.ylabel( ` `'Salary'` `) `<br>`     `<br>`    ``plt.show() `<br>`    `<br>`if` `__name__ ` `=` `=` `"__main__"` `:  `<br>`     `<br>`    ``main()` |

```

```

```

```

#### Output:

```
Predicted values  [ 40594.69 123305.18  65031.88]
Real values       [ 37731 122391  57081]
Trained W         9398.92
Trained b         26496.31

```

![](https://media.geeksforgeeks.org/wp-content/uploads/20200913181446/LinearRegressionmodel.png)

Linear Regression Visualization

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/top-5-open-source-online-machine-learning-environments/)

[Top 5 Open-Source Online Machine Learning Environments](https://www.geeksforgeeks.org/top-5-open-source-online-machine-learning-environments/)

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

[Implementation of Logistic Regression from Scratch using Python\\
\\
\\
Introduction:Logistic Regression is a supervised learning algorithm that is used when the target variable is categorical. Hypothetical function h(x) of linear regression predicts unbounded values. But in the case of Logistic Regression, where the target variable is categorical we have to strict the\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/?ref=ml_lbp)
[Polynomial Regression ( From Scratch using Python )\\
\\
\\
Prerequisites Linear RegressionGradient DescentIntroductionLinear Regression finds the correlation between the dependent variable ( or target variable ) and independent variables ( or features ). In short, it is a linear model to fit the data linearly. But it fails to fit and catch the pattern in no\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/?ref=ml_lbp)
[Linear Regression (Python Implementation)\\
\\
\\
Linear regression is a statistical method that is used to predict a continuous dependent variable i.e target variable based on one or more independent variables. This technique assumes a linear relationship between the dependent and independent variables which means the dependent variable changes pr\\
\\
14 min read](https://www.geeksforgeeks.org/linear-regression-python-implementation/?ref=ml_lbp)
[Implementing SVM from Scratch in Python\\
\\
\\
Support Vector Machines (SVMs) are powerful supervised machine learning algorithms used for classification and regression tasks. They work by finding the optimal hyperplane that separates data points of different classes with the maximum margin. We can use Scikit library of python to implement SVM b\\
\\
4 min read](https://www.geeksforgeeks.org/implementing-svm-from-scratch-in-python/?ref=ml_lbp)
[ML \| Naive Bayes Scratch Implementation using Python\\
\\
\\
Naive Bayes is a probabilistic machine learning algorithms based on the Bayes Theorem. It is a simple yet powerful algorithm because of its understanding, simplicity and ease of implementation. It is popular method for classification applications such as spam filtering and text classification. In th\\
\\
7 min read](https://www.geeksforgeeks.org/ml-naive-bayes-scratch-implementation-using-python/?ref=ml_lbp)
[Implementation of neural network from scratch using NumPy\\
\\
\\
Neural networks are a core component of deep learning models, and implementing them from scratch is a great way to understand their inner workings. we will demonstrate how to implement a basic Neural networks algorithm from scratch using the NumPy library in Python, focusing on building a three-lett\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/?ref=ml_lbp)
[Python \| Linear Regression using sklearn\\
\\
\\
Prerequisite: Linear Regression Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecast\\
\\
3 min read](https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/?ref=ml_lbp)
[Solving Linear Regression in Python\\
\\
\\
Linear regression is a common method to model the relationship between a dependent variable and one or more independent variables. Linear models are developed using the parameters which are estimated from the data. Linear regression is useful in prediction and forecasting where a predictive model is\\
\\
4 min read](https://www.geeksforgeeks.org/solving-linear-regression-in-python/?ref=ml_lbp)
[Simple Linear Regression in Python\\
\\
\\
Simple linear regression models the relationship between a dependent variable and a single independent variable. In this article, we will explore simple linear regression and it's implementation in Python using libraries such as NumPy, Pandas, and scikit-learn. Understanding Simple Linear Regression\\
\\
7 min read](https://www.geeksforgeeks.org/simple-linear-regression-in-python/?ref=ml_lbp)
[Univariate Linear Regression in Python\\
\\
\\
In this article, we will explain univariate linear regression. It is one of the simplest types of regression. In this regression, we predict our target value on only one independent variable. Univariate Linear Regression in PythonUnivariate Linear Regression is a type of regression in which the targ\\
\\
7 min read](https://www.geeksforgeeks.org/univariate-linear-regression-in-python/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=460419195.1745055580&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=357423003)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055579958&cv=11&fst=1745055579958&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102665699~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Flinear-regression-implementation-from-scratch-using-python%2F&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Linear%20Regression%20Implementation%20From%20Scratch%20using%20Python%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1854497793.1745055580&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)