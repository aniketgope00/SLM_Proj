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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/regression-in-machine-learning/?type%3Darticle%26id%3D165762&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Plotting graph using Seaborn \| Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/plotting-graph-using-seaborn-python/)

# Regression in machine learning

Last Updated : 13 Jan, 2025

Comments

Improve

Suggest changes

40 Likes

Like

Report

Regression in machine learning refers to a [**supervised learning**](https://www.geeksforgeeks.org/supervised-machine-learning/) technique where the goal is to predict a continuous numerical value based on one or more independent features. It finds relationships between variables so that predictions can be made. we have two types of variables present in regression:

- **Dependent Variable (Target)**: The variable we are trying to predict e.g house price.
- **Independent Variables (Features)**: The input variables that influence the prediction e.g locality, number of rooms.

Regression analysis problem works with if output variable is a real or continuous value such as “salary” or “weight”. Many different regression models can be used but the simplest model in them is linear regression.

## Types of Regression

Regression can be classified into different types based on the number of predictor variables and the nature of the relationship between variables:

### 1\. **Simple Linear Regression**

[**Linear regression**](https://www.geeksforgeeks.org/ml-linear-regression/) is one of the simplest and most widely used statistical models. This assumes that there is a linear relationship between the independent and dependent variables. This means that the change in the dependent variable is proportional to the change in the independent variables.For example predicting the price of a house based on its size.

### 2\. **Multiple Linear Regression**

[Multiple linear regression](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/) extends simple linear regression by using multiple independent variables to predict target variable.For example predicting the price of a house based on multiple features such as size, location, number of rooms, etc.

### 3\. **Polynomial Regression**

[Polynomial regression](https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/) is used to model with non-linear relationships between the dependent variable and the independent variables. It adds polynomial terms to the linear regression model to capture more complex relationships.For example when we want to predict a non-linear trend like population growth over time we use polynomial regression.

### 4\. **Ridge & Lasso Regression**

[Ridge & lasso regression](https://www.geeksforgeeks.org/ridge-regression-vs-lasso-regression/) are regularized versions of linear regression that help avoid overfitting by penalizing large coefficients.When there’s a risk of overfitting due to too many features we use these type of regression algorithms.

### 5\. **Support Vector Regression (SVR)**

SVR is a type of regression algorithm that is based on the [Support Vector Machine (SVM)](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) algorithm. SVM is a type of algorithm that is used for classification tasks but it can also be used for regression tasks. SVR works by finding a hyperplane that minimizes the sum of the squared residuals between the predicted and actual values.

### 6\. **Decision Tree Regression**

[Decision tree](https://www.geeksforgeeks.org/decision-tree/) Uses a tree-like structure to make decisions where each branch of tree represents a decision and leaves represent outcomes. For example predicting customer behavior based on features like age, income, etc there we use decison tree regression.

### 7\. **Random Forest Regression**

[Random Forest](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/) is a ensemble method that builds multiple decision trees and each tree is trained on a different subset of the training data. The final prediction is made by averaging the predictions of all of the trees. For example customer churn or sales data using this.

## Regression Evaluation Metrics

Evaluation in machine learning measures the performance of a model. Here are some popular evaluation metrics for regression:

- [**Mean Absolute Error (MAE):**](https://www.geeksforgeeks.org/how-to-calculate-mean-absolute-error-in-python/) The average absolute difference between the predicted and actual values of the target variable.
- [**Mean Squared Error (MSE):**](https://www.geeksforgeeks.org/python-mean-squared-error/) The average squared difference between the predicted and actual values of the target variable.
- [**Root Mean Squared Error (RMSE)**](https://www.geeksforgeeks.org/rmse-root-mean-square-error-in-matlab/) **:** Square root of the mean squared error.
- [**Huber Loss:**](https://www.geeksforgeeks.org/sklearn-different-loss-functions-in-sgd/) A hybrid loss function that transitions from MAE to MSE for larger errors, providing balance between robustness and MSE’s sensitivity to outliers.
- [R2 – Score](https://www.geeksforgeeks.org/python-coefficient-of-determination-r2-score/): Higher values indicate better fit ranging from 0 to 1.

## Regression Model Machine Learning

Let’s take an example of linear regression. We have a [**Housing data set**](https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv) and we want to predict the price of the house. Following is the python code for it.

Python`
import matplotlib
matplotlib.use('TkAgg')  # General backend for plots

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

# Load dataset
df = pd.read_csv("Housing.csv")
# Extract features and target variable
Y = df['price']
X = df['lotsize']
# Reshape for compatibility with scikit-learn
X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)
# Split data into training and testing sets
X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]
# Plot the test data
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
# Train linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
# Plot predictions
plt.plot(X_test, regr.predict(X_test), color='red', linewidth=3)
plt.show()
`

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/Screenshot-28-1-300x208.png)

Here in this graph we plot the test data. The red line indicates the best fit line for predicting the price.

To make an individual prediction using the linear regression model:

```
print("Predicted price for a lot size of 5000: " + str(round(regr.predict([[5000]])[0][0])))

```

## **Applications of Regression**

- **Predicting prices:** Used to predict the price of a house based on its size, location and other features.
- **Forecasting trends:** Model to forecast the sales of a product based on historical sales data.
- **Identifying risk factors:** Used to identify risk factors for heart patient based on patient medical data.
- **Making decisions:** It could be used to recommend which stock to buy based on market data.

## **Advantages of Regression**

- Easy to understand and interpret.
- Robust to outliers.
- Can handle both linear relationships easily.

## **Disadvantages of Regression**

- Assumes linearity.
- Sensitive to situation where two or more independent variables are highly correlated with each other i.e multicollinearity.
- May not be suitable for highly complex relationships.

## Conclusion

Regression in machine learning is a fundamental technique for predicting continuous outcomes based on input features. It is used in many real-world applications like price prediction, trend analysis and risk assessment. With its simplicity and effectiveness regression is used to understand relationships in data.

Suggested Quiz


10 Questions


Which of the following regression techniques is specifically designed to handle multicollinearity among independent variables?

- Ridge Regression

- Simple Linear Regression

- Polynomial Regression

- Decision Tree Regression


Explanation:

Ridge Regression addresses multicollinearity by adding a penalty term to the loss function, which is proportional to the square of the coefficients (L2 regularization).

Which of the following is the primary purpose of residual analysis in regression?

b)

c) d)

- To verify that the predictors are independent of each other

- To assess the normality of the predictor variables

- To identify the most important predictors

- The model's reliance on a single predictor variable


Explanation:

Residual analysis is used to check for patterns or trends in the residuals to assess the fit of the model.

Which evaluation metric is most appropriate for assessing regression models when outliers are present?

- Mean Squared Error (MSE)

- Mean Absolute Error (MAE)

- Root Mean Squared Error (RMSE)

- Huber Loss


Explanation:

Huber Loss combines the advantages of both MAE and MSE. It is quadratic for small errors (similar to MSE) and linear for large errors (similar to MAE). This means it is less sensitive to outliers than MSE while still providing a smooth gradient for optimization.

What is the main issue with using ordinary least squares (OLS) when there is autocorrelation in the residuals?

- OLS estimates become biased and inconsistent

- The assumption of normally distributed errors is violated

- The variance of the errors is no longer constant

- The residuals become homoscedastic


Explanation:

Autocorrelation disruptes the assumption of independent residuals, which can lead to biased and inconsistent estimates in OLS.

In the context of regression, what does the term "R-squared" represent?

- The proportion of variance in the dependent variable explained by the independent variable(s)

- The average of the absolute errors between predicted and actual values

- The maximum possible value of the regression coefficients

- The likelihood of the model fitting the data well


Explanation:

In the context of regression, **R-squared** (also known as the coefficient of determination) represents the proportion of the variance in the dependent variable that can be explained by the independent variable(s) in the model. It ranges from 0 to 1.

When using LASSO regression, what is the primary effect of applying L1 regularization?

- It increases all coefficients uniformly.

- It reduces multicollinearity by averaging coefficients.

- It can shrink some coefficients to exactly zero, effectively performing variable selection.

- It ensures that all predictors are included in the final model.


Explanation:

It can shrink some coefficients to exactly zero, effectively performing variable selection.

What is the primary purpose of using regularization techniques like Lasso and Ridge in regression?

- To increase the number of independent variables in the model to improve accuracy.

- To reduce the risk of overfitting by penalizing large coefficients and simplifying the model.

- To ensure that all predictor variables contribute equally to the model's predictions.

- To eliminate multicollinearity by removing correlated independent variables from the dataset.


Explanation:

To reduce the risk of overfitting by penalizing large coefficients and simplifying the model.

In regression analysis, what does "heteroscedasticity" indicate about the residuals?

- The residuals have constant variance across all levels of the independent variable.

- The residuals exhibit a systematic pattern.

- The variance of the residuals changes at different levels of the independent variable.

- The residuals are normally distributed.


Explanation:

Heteroscedasticity signifies that as the value of the independent variable(s) increases or decreases, the spread or variability of the residuals also changes systematically.

Which of the following is a potential consequence of multicollinearity in a regression model?

- Increased interpretability of the model

- Reduced statistical power of the model

- Enhanced prediction accuracy

- Simplification of the model structure


Explanation:

Because multicollinearity inflates the standard errors of the coefficient estimates, making it difficult to determine the individual effects of correlated predictors.

What is the primary difference between Support Vector Regression (SVR) and traditional linear regression?

- SVR can only handle linear relationships.

- SVR focuses on maximizing the margin around the predicted values.

- SVR requires a larger dataset to perform effectively.

- SVR is not suitable for high-dimensional data.


Explanation:

Support Vector Regression (SVR) aims to find a function that not only fits the data but also maintains a specified margin of tolerance (epsilon) around the predicted values.

![](https://media.geeksforgeeks.org/auth-dashboard-uploads/sucess-img.png)

Quiz Completed Successfully


Your Score :   2/10

Accuracy :  0%

Login to View Explanation


1/10 1/10
< Previous

Next >


Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/plotting-graph-using-seaborn-python/)

[Plotting graph using Seaborn \| Python](https://www.geeksforgeeks.org/plotting-graph-using-seaborn-python/)

[S](https://www.geeksforgeeks.org/user/Sagar%20Shukla/)

[Sagar Shukla](https://www.geeksforgeeks.org/user/Sagar%20Shukla/)

Follow

40

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Linear Regression in Machine learning\\
\\
\\
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It provides valuable insights for prediction and data analysis. This article will explore its types, assumptions, implementation, advantages and evaluation met\\
\\
15+ min read](https://www.geeksforgeeks.org/ml-linear-regression/?ref=ml_lbp)
[Logistic Regression in Machine Learning\\
\\
\\
In our previous discussion, we explored the fundamentals of machine learning and walked through a hands-on implementation of Linear Regression. Now, let's take a step forward and dive into one of the first and most widely used classification algorithms â€” Logistic Regression What is Logistic Regressi\\
\\
13 min read](https://www.geeksforgeeks.org/understanding-logistic-regression/?ref=ml_lbp)
[Regularization in Machine Learning\\
\\
\\
In the previous session, we learned how to implement linear regression. Now, weâ€™ll move on to regularization, which helps prevent overfitting and makes our models work better with new data. While developing machine learning models we may encounter a situation where model is overfitted. To avoid such\\
\\
8 min read](https://www.geeksforgeeks.org/regularization-in-machine-learning/?ref=ml_lbp)
[Multioutput Regression in Machine Learning\\
\\
\\
In machine learning we often encounter regression, these problems involve predicting a continuous target variable, such as house prices, or temperature. However, in many real-world scenarios, we need to predict not only single but many variables together, this is where we use multi-output regression\\
\\
11 min read](https://www.geeksforgeeks.org/multioutput-regression-in-machine-learning/?ref=ml_lbp)
[Machine learning in Marketing\\
\\
\\
Marketers are constantly seeking new ways to understand their customers and optimize campaigns for maximum impact. Enter machine learning (ML), a powerful technology that is transforming the marketing landscape. By leveraging ML algorithms, businesses can glean valuable insights from vast amounts of\\
\\
11 min read](https://www.geeksforgeeks.org/machine-learning-in-marketing/?ref=ml_lbp)
[Classification vs Regression in Machine Learning\\
\\
\\
Classification and regression are two primary tasks in supervised machine learning, where key difference lies in the nature of the output: classification deals with discrete outcomes (e.g., yes/no, categories), while regression handles continuous values (e.g., price, temperature). Both approaches re\\
\\
5 min read](https://www.geeksforgeeks.org/ml-classification-vs-regression/?ref=ml_lbp)
[Robust Regression for Machine Learning in Python\\
\\
\\
Simple linear regression aims to find the best fit line that describes the linear relationship between some input variables(denoted by X) and the target variable(denoted by y). This has some limitations as in real-world problems, there is a high probability that the dataset may have outliers. This r\\
\\
4 min read](https://www.geeksforgeeks.org/robust-regression-for-machine-learning-in-python/?ref=ml_lbp)
[Machine Learning Projects Using Regression\\
\\
\\
Regression analysis in machine learning aims to model the relationship between a dependent variable and one or more independent variables. The central goal is to predict the value of the dependent variable based on input features. Linear Regression assumes a linear relationship, finding the best-fit\\
\\
15 min read](https://www.geeksforgeeks.org/machine-learning-projects-using-regression/?ref=ml_lbp)
[10 Machine Learning Projects in Retail\\
\\
\\
In the modern-day dynamic retail landscape, maintaining a competitive edge goes beyond offering top-notch products and services. Retail businesses must harness the power of advanced technologies to decode consumer behavior. Machine Learning emerges as a game changer in the context that provides reta\\
\\
9 min read](https://www.geeksforgeeks.org/10-machine-learning-projects-in-retail/?ref=ml_lbp)
[Machine Learning with R\\
\\
\\
Machine Learning as the name suggests is the field of study that allows computers to learn and take decisions on their own i.e. without being explicitly programmed. These decisions are based on the available data that is available through experiences or instructions. It gives the computer that makes\\
\\
2 min read](https://www.geeksforgeeks.org/machine-learning-with-r/?ref=ml_lbp)

Like40

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/regression-in-machine-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=262789439.1745055546&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=1166042815)

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