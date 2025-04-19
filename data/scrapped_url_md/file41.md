- [Python](https://www.geeksforgeeks.org/python-programming-language/)
- [R Language](https://www.geeksforgeeks.org/r-tutorial/)
- [Python for Data Science](https://www.geeksforgeeks.org/data-science-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [Data Analysis](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [ML Math](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning-interview-questions/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-project-with-source-code/)
- [ML Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/what-is-ridge-regression/?type%3Darticle%26id%3D1268861&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Weighted Ridge Regression in R\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/weighted-ridge-regression-in-r/)

# Ridge Regression

Last Updated : 12 Feb, 2025

Comments

Improve

Suggest changes

5 Likes

Like

Report

Ridge regression, also known as  L2 regularization, is a technique used in linear regression to address the **problem of multicollinearity among predictor variables**. Multicollinearity occurs when independent variables in a regression model are highly correlated, which can lead to unreliable and unstable estimates of regression coefficients.

> Ridge regression mitigates this issue by adding a **regularization term to the ordinary least squares (OLS) objective function, which penalizes large coefficients and thus reduces their variance.**

![Understanding-Ridge-Regression-copy](https://media.geeksforgeeks.org/wp-content/uploads/20240611185740/Understanding-Ridge-Regression-copy.webp)What is ridge regression?

## How Ridge Regression **Addresses Overfitting and Multicollinearity?**

[Overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/) occurs when a model becomes too complex and fits the noise in the training data, leading to poor generalization on new data. Ridge regression combats overfitting by adding a penalty term (L2 regularization) to the ordinary least squares ( [OLS](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)) objective function.

> Imagine your model is _overreacting_ to tiny details in the data (like memorizing noise). Ridge regression "calms it down" by shrinking the model's weights (coefficients) toward zero. Think of it like adjusting a volume knob to get the perfect sound level—not too loud (overfitting), not too quiet (underfitting).

_**This penalty discourages the model from using large values for the coefficients (the numbers multiplying the features). It forces the model to keep these coefficients small**_. By making the coefficients smaller and closer to zero, ridge regression simplifies the model and reduces its sensitivity to random fluctuations or noise in the data. This makes the model less likely to overfit and helps it perform better on new, unseen data, improving its overall accuracy and reliability.

**For Example** \- We are predicting house prices based on multiple features such as square footage, number of bedrooms, and age of the house:

**Price=1000 Size−500⋅Age+Noise**

- **Ridge might adjust it to:**

**Price=800⋅Size−300⋅Age+Less Noise**

As lambda increases the model **places more emphasis on shrinking the coefficients of highly correlated features**, making their impact smaller and more stable. This reduces the effect of multicollinearity by preventing large fluctuations in coefficient estimates due to correlated predictors.

## Mathematical Formulation of Ridge Regression Estimator

Consider the multiple linear regression model:.

> y=Xβ+ϵ

where:

- y is an n×1 vector of observations,
- X is an n×p matrix of predictors,
- β is a p×1 vector of unknown regression coefficients,
- ϵ is an n×1 vector of random errors.

The [ordinary least squares](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/) (OLS) estimator of _β_ is given by:

\\hat{\\beta}\_{\\text{OLS}} = (X'X)^{-1}X'y

In the presence of multicollinearity, X^′X is nearly singular, leading to unstable estimates. ridge regression addresses this issue by adding a penalty term kI, where k is the ridge parameter and I is the identity matrix. The ridge regression estimator is:

\\hat{\\beta}\_k = (X'X + kI)^{-1}X'y

This modification stabilizes the estimates by shrinking the coefficients, improving generalization and mitigating multicollinearity effects.

## Bias-Variance Tradeoff in Ridge Regression

Ridge regression allows control over the [bias-variance trade-off.](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/) Increasing the value of λ increases the bias but reduces the variance, while decreasing λ does the opposite. The goal is to find an optimal λ that balances bias and variance, leading to a model that generalizes well to new data.

As we increase the penalty level in ridge regression, the estimates of β gradually change. The following simulation illustrates how the variation in β is affected by different penalty values, showing how estimated parameters deviate from the true values.

![Bias-Variance-Tradeoff-in-Ridge-Regression](https://media.geeksforgeeks.org/wp-content/uploads/20240611185648/Bias-Variance-Tradeoff-in-Ridge-Regression.webp)Bias-Variance Tradeoff in Ridge Regression

Ridge regression introduces bias into the estimates to reduce their variance. The [mean squared error (MSE)](https://www.geeksforgeeks.org/python-mean-squared-error/) of the ridge estimator can be decomposed into bias and variance components:

\\text{MSE}(\\hat{\\beta}\_k) = \\text{Bias}^2(\\hat{\\beta}\_k) + \\text{Var}(\\hat{\\beta}\_k)

- **Bias**: Measures the error introduced by approximating a real-world problem, which may be complex, by a simplified model. In ridge regression, as the regularization parameter k increases, the model becomes simpler, which increases bias but reduces variance.
- **Variance**: Measures how much the ridge regression model's predictions would vary if we used different training data. As the regularization parameter k decreases, the model becomes more complex, fitting the training data more closely, which reduces bias but increases variance.
- **Irreducible Error**: Represents the noise in the data that cannot be reduced by any model.

As k increases, the bias increases, but the variance decreases. The optimal value of k balances this tradeoff, minimizing the MSE.

## Selection of the Ridge Parameter in Ridge Regression

Choosing an appropriate value for the ridge parameter k is crucial in ridge regression, as it directly influences the bias-variance tradeoff and the overall performance of the model. Several methods have been proposed for selecting the optimal ridge parameter, each with its own advantages and limitations. Methods for Selecting the Ridge Parameter are:

**1\. Cross-Validation**

[Cross-validation](https://www.geeksforgeeks.org/cross-validation-machine-learning/) is a common method for selecting the ridge parameter by dividing data into subsets. The model trains on some subsets and validates on others, repeating this process and averaging the results to find the optimal value of k.

- K-Fold Cross-Validation: The data is split into K subsets, training on K-1 folds and validating on the remaining fold. This is repeated K times, with each fold serving as the validation set once.
- [**Leave-One-Out Cross-Validation (LOOCV)**](https://www.geeksforgeeks.org/loocvleave-one-out-cross-validation-in-r-programming/) A special case of K-fold where K equals the number of observations, training on all but one observation and validating on the remaining one. It’s computationally intensive but unbiased.

### 2\. Generalized Cross-Validation (GCV)

[Generalized Cross-Validation](https://www.geeksforgeeks.org/mastering-generalized-cross-validation-gcv-theory-applications-and-best-practices/) is an extension of cross-validation that provides a more efficient way to estimate the optimal k without explicitly dividing the data. GCV is based on the idea of minimizing a function that approximates the leave-one-out cross-validation error. It is computationally less intensive and often yields similar results to traditional cross-validation methods.

### 3\. Information Criteria

Information criteria such as the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC) can also be used to select the ridge parameter. These criteria balance the goodness of fit of the model with its complexity, penalizing models with more parameters.

### 4\. Empirical Bayes Methods

Empirical Bayes methods involve estimating the ridge parameter by treating it as a hyperparameter in a Bayesian framework. These methods use prior distributions and observed data to estimate the posterior distribution of the ridge parameter.

**Empirical Bayes Estimation**: This method involves specifying a prior distribution for k and using the observed data to update this prior to obtain a posterior distribution. The mode or mean of the posterior distribution is then used as the estimate of k.

### **5\. Stability Selection**

Stability selection improves ridge parameter robustness by subsampling data and fitting the model multiple times. The most frequently selected parameter across all subsamples is chosen as the final estimate.

### Practical Considerations for Selecting Ridge Parameter

- **Tradeoff Between Bias and Variance:** The choice of the ridge parameter k involves a tradeoff between bias and variance. A larger k introduces more bias but reduces variance, while a smaller _k_ reduces bias but increases variance. The optimal k balances this tradeoff to minimize the mean squared error (MSE) of the model.
- **Computational Efficiency**: Some methods for selecting k, such as cross-validation and empirical Bayes methods, can be computationally intensive, especially for large datasets. Generalized cross-validation and analytical methods offer more computationally efficient alternatives.
- **Interpretability:** The interpretability of the selected ridge parameter is also an important consideration. Methods that provide explicit criteria or formulas for selecting k can offer more insight into the relationship between the data and the model.

Read about [Implementation of Ridge Regression from Scratch using Python.](https://www.geeksforgeeks.org/implementation-of-ridge-regression-from-scratch-using-python/)

## Applications of Ridge Regression

- **Forecasting Economic Indicators:** Ridge regression helps predict economic factors like GDP, inflation, and unemployment by managing multicollinearity between predictors like interest rates and consumer spending, leading to more accurate forecasts.
- **Medical Diagnosis**: In healthcare, it aids in building diagnostic models by controlling multicollinearity among biomarkers, improving disease diagnosis and prognosis.
- **Sales Prediction**: In marketing, ridge regression forecasts sales based on factors like advertisement costs and promotions, handling correlations between these variables for better sales planning.
- **Climate Modeling:** Ridge regression improves climate models by eliminating interference between variables like temperature and precipitation, ensuring more accurate predictions.
- **Risk Management**: In credit scoring and financial risk analysis, ridge regression evaluates creditworthiness by addressing multicollinearity among financial ratios, enhancing accuracy in risk management.

## Advantages and Disadvantages of Ridge Regression

### Advantages:

- **Stability**: Ridge regression provides more stable estimates in the presence of multicollinearity.
- **Bias-Variance Tradeoff**: By introducing bias, ridge regression reduces the variance of the estimates, leading to lower MSE.
- **Interpretability**: Unlike principal component regression, ridge regression retains the original predictors, making the results easier to interpret.

### Disadvantages:

- **Bias Introduction**: The introduction of bias can lead to underestimation of the true effects of the predictors.
- **Parameter Selection**: Choosing the optimal ridge parameter k can be challenging and computationally intensive.
- **Not Suitable for Variable Selection**: Ridge regression does not perform variable selection, meaning all predictors remain in the model, even those with negligible effects.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/weighted-ridge-regression-in-r/)

[Weighted Ridge Regression in R](https://www.geeksforgeeks.org/weighted-ridge-regression-in-r/)

[I](https://www.geeksforgeeks.org/user/indradhm0hg/)

[indradhm0hg](https://www.geeksforgeeks.org/user/indradhm0hg/)

Follow

5

Improve

Article Tags :

- [Data Science](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [data-science](https://www.geeksforgeeks.org/tag/data-science/)
- [ML-Regression](https://www.geeksforgeeks.org/tag/ml-regression/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+3 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[What is Ridge Regression?\\
\\
\\
Ridge regression, also known as L2 regularization, is a technique used in linear regression to address the problem of multicollinearity among predictor variables. Multicollinearity occurs when independent variables in a regression model are highly correlated, which can lead to unreliable and unstabl\\
\\
8 min read](https://www.geeksforgeeks.org/what-is-ridge-regression/?ref=ml_lbp)
[Weighted Ridge Regression in R\\
\\
\\
Ridge Regression is a key method used in statistics and machine learning to deal with a common problem called multicollinearity in regression analysis. It does this by adding a penalty to the regression equation, which helps to make the estimates more stable, especially when the predictor variables\\
\\
5 min read](https://www.geeksforgeeks.org/weighted-ridge-regression-in-r/?ref=ml_lbp)
[Ridge Regression in R Programming\\
\\
\\
Ridge regression is a classification algorithm that works in part as it doesnâ€™t require unbiased estimators. Ridge regression minimizes the residual sum of squares of predictors in a given model. Ridge regression includes a shrinks the estimate of the coefficients towards zero. Ridge Regression in R\\
\\
5 min read](https://www.geeksforgeeks.org/ridge-regression-in-r-programming/?ref=ml_lbp)
[Ridge Regression vs Lasso Regression\\
\\
\\
Ridge and Lasso Regression are two popular techniques in machine learning used for regularizing linear models to avoid overfitting and improve predictive performance. Both methods add a penalty term to the modelâ€™s cost function to constrain the coefficients, but they differ in how they apply this pe\\
\\
6 min read](https://www.geeksforgeeks.org/ridge-regression-vs-lasso-regression/?ref=ml_lbp)
[ML \| Ridge Regressor using sklearn\\
\\
\\
Ridge regression is a powerful technique used in statistics and machine learning to improve the performance of linear regression models. In this article we will understand the concept of ridge regression with its implementation in sklearn. Ridge RegressionA Ridge regressor is basically a regularized\\
\\
4 min read](https://www.geeksforgeeks.org/ml-ridge-regressor-using-sklearn/?ref=ml_lbp)
[Ridge Classifier\\
\\
\\
Supervised Learning is the type of Machine Learning that uses labelled data to train the model. Both Regression and Classification belong to the category of Supervised Learning. Regression: This is used to predict a continuous range of values using one or more features. These features act as the ind\\
\\
10 min read](https://www.geeksforgeeks.org/ridge-classifier/?ref=ml_lbp)
[ML \| Locally weighted Linear Regression\\
\\
\\
Linear Regression is a supervised learning algorithm used for computing linear relationships between input (X) and output (Y). The steps involved in ordinary linear regression are: Training phase: Compute \[Tex\]\\theta \[/Tex\]to minimize the cost. \[Tex\]J(\\theta) = $\\sum\_{i=1}^{m} (\\theta^Tx^{(i)} - y^{\\
\\
3 min read](https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/?ref=ml_lbp)
[Understanding Kernel Ridge Regression With Sklearn\\
\\
\\
Kernel ridge regression (KRR) is a powerful technique in scikit-learn for tackling regression problems, particularly when dealing with non-linear relationships between features and the target variable. Â This technique allows for the modeling of complex, nonlinear relationships between variables, mak\\
\\
7 min read](https://www.geeksforgeeks.org/understanding-kernel-ridge-regression-with-sklearn/?ref=ml_lbp)
[How to create a 3D ridge border using CSS?\\
\\
\\
In CSS, the border-style property is used to set the line style of the border of an element. The border-style property may have one, two, three, or four values. When the specified value is one, the same style is applied to all four sides. When the specified value is two, the first style is applied t\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-create-a-3d-ridge-border-using-css/?ref=ml_lbp)
[Lasso vs Ridge vs Elastic Net \| ML\\
\\
\\
Regularization methods such as Lasso, Ridge and Elastic Net are important in improving linear regression models by avoiding overfitting, solving multicollinearity and feature selection. These methods enhance the model's predictive accuracy and robustness. Below is a concise explanation of how each t\\
\\
5 min read](https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/?ref=ml_lbp)

Like5

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/what-is-ridge-regression/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1126946741.1745055811&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=914931478)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055811472&cv=11&fst=1745055811472&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fwhat-is-ridge-regression%2F&hn=www.googleadservices.com&frm=0&tiba=Ridge%20Regression%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1718296672.1745055811&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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