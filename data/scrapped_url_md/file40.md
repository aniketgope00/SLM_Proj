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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/?type%3Darticle%26id%3D307043&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Regularization in Machine Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/regularization-in-machine-learning/)

# Python \| Linear Regression using sklearn

Last Updated : 22 May, 2024

Comments

Improve

Suggest changes

29 Likes

Like

Report

**Prerequisite:** [Linear Regression](https://www.geeksforgeeks.org/ml-linear-regression/)

Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between the dependent and independent variables, they are considering and the number of independent variables being used. This article is going to demonstrate how to use the various Python libraries to implement linear regression on a given dataset. We will demonstrate a binary linear model as this will be easier to visualize. In this demonstration, the model will use Gradient Descent to learn. You can learn about it [here.](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)

**Step 1:** Importing all the required libraries

- Python3

## Python3

|     |
| --- |
| `import` `numpy as np `<br>`import` `pandas as pd `<br>`import` `seaborn as sns `<br>`import` `matplotlib.pyplot as plt `<br>`from` `sklearn ` `import` `preprocessing, svm `<br>`from` `sklearn.model_selection ` `import` `train_test_split `<br>`from` `sklearn.linear_model ` `import` `LinearRegression ` |

```

```

```

```

**Step 2:** Reading the dataset:

- Python3

## Python3

|     |
| --- |
| `df ` `=` `pd.read_csv(` `'bottle.csv'` `) `<br>`df_binary ` `=` `df[[` `'Salnty'` `, ` `'T_degC'` `]] `<br>` `<br>`# Taking only the selected two attributes from the dataset `<br>`df_binary.columns ` `=` `[` `'Sal'` `, ` `'Temp'` `] `<br>`#display the first 5 rows `<br>`df_binary.head()` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522133619/Capture143.png)

**Step 3:** Exploring the data scatter

- Python3

## Python3

|     |
| --- |
| `#plotting the Scatter plot to check relationship between Sal and Temp `<br>`sns.lmplot(x ` `=` `"Sal"` `, y ` `=` `"Temp"` `, data ` `=` `df_binary, order ` `=` `2` `, ci ` `=` `None` `) `<br>`plt.show()` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522134153/Capture224-300x244.png)

**Step 4:** Data cleaning

- Python3

## Python3

|     |
| --- |
| `# Eliminating NaN or missing input numbers `<br>`df_binary.fillna(method ` `=` `'ffill'` `, inplace ` `=` `True` `)` |

```

```

```

```

**Step 5:** Training our model

- Python3

## Python3

|     |
| --- |
| `X ` `=` `np.array(df_binary[` `'Sal'` `]).reshape(` `-` `1` `, ` `1` `) `<br>`y ` `=` `np.array(df_binary[` `'Temp'` `]).reshape(` `-` `1` `, ` `1` `) `<br>` `<br>`# Separating the data into independent and dependent variables `<br>`# Converting each dataframe into a numpy array  `<br>`# since each dataframe contains only one column `<br>`df_binary.dropna(inplace ` `=` `True` `) `<br>` `<br>`# Dropping any rows with Nan values `<br>`X_train, X_test, y_train, y_test ` `=` `train_test_split(X, y, test_size ` `=` `0.25` `) `<br>` `<br>`# Splitting the data into training and testing data `<br>`regr ` `=` `LinearRegression() `<br>` `<br>`regr.fit(X_train, y_train) `<br>`print` `(regr.score(X_test, y_test)) ` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522134357/Capture34-3.png)

**Step 6:** Exploring our results

- Python3

## Python3

|     |
| --- |
| `y_pred ` `=` `regr.predict(X_test) `<br>`plt.scatter(X_test, y_test, color ` `=` `'b'` `) `<br>`plt.plot(X_test, y_pred, color ` `=` `'k'` `) `<br>` `<br>`plt.show() `<br>`# Data scatter of predicted values ` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522134513/Capture48-300x179.png)

The low accuracy score of our model suggests that our regressive model has not fit very well with the existing data. This suggests that our data is not suitable for linear regression. But sometimes, a dataset may accept a linear regressor if we consider only a part of it. Let us check for that possibility.

**Step 7:** Working with a smaller dataset

- Python3

## Python3

|     |
| --- |
| `df_binary500 ` `=` `df_binary[:][:` `500` `] `<br>`   `<br>`# Selecting the 1st 500 rows of the data `<br>`sns.lmplot(x ` `=` `"Sal"` `, y ` `=` `"Temp"` `, data ` `=` `df_binary500, `<br>`                               ``order ` `=` `2` `, ci ` `=` `None` `)` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522140357/Capture57-300x237.png)

We can already see that the first 500 rows follow a linear model. Continuing with the same steps as before.

- Python3

## Python3

|     |
| --- |
| `df_binary500.fillna(method ` `=` `'fill'` `, inplace ` `=` `True` `) `<br>` `<br>`X ` `=` `np.array(df_binary500[` `'Sal'` `]).reshape(` `-` `1` `, ` `1` `) `<br>`y ` `=` `np.array(df_binary500[` `'Temp'` `]).reshape(` `-` `1` `, ` `1` `) `<br>` `<br>`df_binary500.dropna(inplace ` `=` `True` `) `<br>`X_train, X_test, y_train, y_test ` `=` `train_test_split(X, y, test_size ` `=` `0.25` `) `<br>` `<br>`regr ` `=` `LinearRegression() `<br>`regr.fit(X_train, y_train) `<br>`print` `(regr.score(X_test, y_test))` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522140536/Capture65.png)

- Python3

## Python3

|     |
| --- |
| `y_pred ` `=` `regr.predict(X_test) `<br>`plt.scatter(X_test, y_test, color ` `=` `'b'` `) `<br>`plt.plot(X_test, y_pred, color ` `=` `'k'` `) `<br>` `<br>`plt.show() ` |

```

```

```

```

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190522140720/Capture75-300x179.png)

**Step 8:** Evaluation Metrics For Regression

At last, we check the performance of the Linear Regression model with help of evaluation metrics. For Regression algorithms we widely use mean\_absolute\_error, and mean\_squared\_error metrics to check the model performance.

- Python3

## Python3

|     |
| --- |
| `from` `sklearn.metrics ` `import` `mean_absolute_error,mean_squared_error `<br>` `<br>`mae ` `=` `mean_absolute_error(y_true` `=` `y_test,y_pred` `=` `y_pred) `<br>`#squared True returns MSE value, False returns RMSE value. `<br>`mse ` `=` `mean_squared_error(y_true` `=` `y_test,y_pred` `=` `y_pred) ` `#default=True `<br>`rmse ` `=` `mean_squared_error(y_true` `=` `y_test,y_pred` `=` `y_pred,squared` `=` `False` `) `<br>` `<br>`print` `(` `"MAE:"` `,mae) `<br>`print` `(` `"MSE:"` `,mse) `<br>`print` `(` `"RMSE:"` `,rmse)` |

```

```

```

```

**Output:**

```
MAE: 0.7927322046360309
MSE: 1.0251137190180517
RMSE: 1.0124789968281078
```

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/regularization-in-machine-learning/)

[Regularization in Machine Learning](https://www.geeksforgeeks.org/regularization-in-machine-learning/)

[A](https://www.geeksforgeeks.org/user/AlindGupta/)

[AlindGupta](https://www.geeksforgeeks.org/user/AlindGupta/)

Follow

29

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Solving Linear Regression in Python\\
\\
\\
Linear regression is a common method to model the relationship between a dependent variable and one or more independent variables. Linear models are developed using the parameters which are estimated from the data. Linear regression is useful in prediction and forecasting where a predictive model is\\
\\
4 min read](https://www.geeksforgeeks.org/solving-linear-regression-in-python/?ref=ml_lbp)
[Python \| Decision Tree Regression using sklearn\\
\\
\\
When it comes to predicting continuous values, Decision Tree Regression is a powerful and intuitive machine learning technique. Unlike traditional linear regression, which assumes a straight-line relationship between input features and the target variable, Decision Tree Regression is a non-linear re\\
\\
4 min read](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/?ref=ml_lbp)
[Linear Regression using PyTorch\\
\\
\\
Linear Regression is a very commonly used statistical method that allows us to determine and study the relationship between two continuous variables. The various properties of linear regression and its Python implementation have been covered in this article previously. Now, we shall find out how to\\
\\
4 min read](https://www.geeksforgeeks.org/linear-regression-using-pytorch/?ref=ml_lbp)
[Simple Linear Regression in Python\\
\\
\\
Simple linear regression models the relationship between a dependent variable and a single independent variable. In this article, we will explore simple linear regression and it's implementation in Python using libraries such as NumPy, Pandas, and scikit-learn. Understanding Simple Linear Regression\\
\\
7 min read](https://www.geeksforgeeks.org/simple-linear-regression-in-python/?ref=ml_lbp)
[ML \| Multiple Linear Regression using Python\\
\\
\\
Linear regression is a fundamental statistical method widely used for predictive analysis. It models the relationship between a dependent variable and a single independent variable by fitting a linear equation to the data. Multiple Linear Regression is an extension of this concept that allows us to\\
\\
4 min read](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/?ref=ml_lbp)
[Univariate Linear Regression in Python\\
\\
\\
In this article, we will explain univariate linear regression. It is one of the simplest types of regression. In this regression, we predict our target value on only one independent variable. Univariate Linear Regression in PythonUnivariate Linear Regression is a type of regression in which the targ\\
\\
7 min read](https://www.geeksforgeeks.org/univariate-linear-regression-in-python/?ref=ml_lbp)
[Linear Regression using Turicreate\\
\\
\\
Linear Regression is a method or approach for Supervised Learning.Supervised Learning takes the historical or past data and then train the model and predict the things according to the past results.Linear Regression comes from the word 'Linear' and 'Regression'.Regression concept deals with predicti\\
\\
2 min read](https://www.geeksforgeeks.org/linear-regression-using-turicreate/?ref=ml_lbp)
[Weighted Least Squares Regression in Python\\
\\
\\
Weighted Least Squares (WLS) regression is a powerful extension of ordinary least squares regression, particularly useful when dealing with data that violates the assumption of constant variance. In this guide, we will learn brief overview of Weighted Least Squares regression and demonstrate how to\\
\\
6 min read](https://www.geeksforgeeks.org/weighted-least-squares-regression-in-python/?ref=ml_lbp)
[Logistic Regression using PySpark Python\\
\\
\\
In this tutorial series, we are going to cover Logistic Regression using Pyspark. Logistic Regression is one of the basic ways to perform classification (donâ€™t be confused by the word â€œregressionâ€). Logistic Regression is a classification method. Some examples of classification are: Spam detectionDi\\
\\
3 min read](https://www.geeksforgeeks.org/logistic-regression-using-pyspark-python/?ref=ml_lbp)
[Polynomial Regression ( From Scratch using Python )\\
\\
\\
Prerequisites Linear RegressionGradient DescentIntroductionLinear Regression finds the correlation between the dependent variable ( or target variable ) and independent variables ( or features ). In short, it is a linear model to fit the data linearly. But it fails to fit and catch the pattern in no\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/?ref=ml_lbp)

Like29

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=733788118.1745055808&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1533226474)

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

```

```

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)