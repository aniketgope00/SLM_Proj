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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/what-is-lasso-regression/?type%3Darticle%26id%3D1240441&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is Data Architecture?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/what-is-data-architecture/)

# What is Lasso Regression?

Last Updated : 27 Mar, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Lasso Regression is a regression method based on **Least Absolute Shrinkage** and **Selection Operator** and its an important technique in regression analysis for variables selection and regularization. It helps remove irrelevant data features and preventing overfitting. This allows features with weak influence to be clearly identified as the coefficients of less important variables are shrunk toward zero.

In this guide, we will understand the core concepts of lasso regression as well as how it works to mitigate overfitting.

Table of Content

- [Understanding Lasso Regression](https://www.geeksforgeeks.org/what-is-lasso-regression/#understanding-lasso-regression)
- [Bias-Variance Tradeoff in Lasso Regression](https://www.geeksforgeeks.org/what-is-lasso-regression/#biasvariance-tradeoff-in-lasso-regression)
- [When to use Lasso Regression](https://www.geeksforgeeks.org/what-is-lasso-regression/#when-to-use-lasso-regression)
- [Advantages of Lasso Regression](https://www.geeksforgeeks.org/what-is-lasso-regression/#advantages-of-lasso-regression)
- [Disadvantages of Lasso Regression](https://www.geeksforgeeks.org/what-is-lasso-regression/#disadvantages-of-lasso-regression)

## Understanding Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) regression typically belongs to regularization techniques category, which is usually applied to avoid [overfitting](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/). Lasso Regression enhance the [**linear regression**](https://www.geeksforgeeks.org/ml-linear-regression/) concept by making use of a [regularization](https://www.geeksforgeeks.org/regularization-in-machine-learning/) process in the standard regression equation. Linear Regression operates by minimizing the sum of squared discrepancies between the observed and predicted values by fitting a line to the data points.

However, [multicollinearity](https://www.geeksforgeeks.org/multicollinearity-in-data/) is a condition in which features have a strong correlation with one another occurs in real-words datasets. This is when the regularization approach of Lasso Regression comes in handy. Regularization in simple term add penalty term to model preventing it from overfitting.

> For example if you're attempting with a model to forecast house prices based on features such as location, square footage and the number of bedrooms.
>
> Lasso Regression will let us determine which feature is more important or whereas location and square footage are major determinants of price. By zeroing out the coefficient for the bedroom feature it simplifies the model in order to increase the model accuracy.

## Bias-Variance Tradeoff in Lasso Regression

The balance between bias (error resulting from oversimplified assumptions in the model) and variance (error resulting from sensitivity to little variations in the training data) is known as the [bias-variance tradeoff](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/).

When implementing Lasso Regression, the penalty term i.e., [**L1 regularization**](https://www.geeksforgeeks.org/regularization-in-machine-learning/) helps lower the **variance** of the model by shrinking the coefficients of less significant features towards zero. This reduces the chance of overfitting where the model identifies noise in the training set instead of the underlying patterns.

However, increasing the regularization strength (i.e., raising the **lambda** value) may increase **bias**. This happens because a stronger penalty can cause the model to oversimplify, making it unable to capture the true relationships in the data.

Thus, bias and variance are traded off in lasso regression just like in other regularization strategies. **Achieving the ideal balance usually entails minimizing the total prediction error by adjusting the regularization parameter using methods like** [**cross-validation**](https://www.geeksforgeeks.org/cross-validation-machine-learning/) **.**

![bias-varianec-trade-off1-copy](https://media.geeksforgeeks.org/wp-content/uploads/20240515184306/bias-varianec-trade-off1-copy.webp)Bias Variance Tradeoff

## Understanding Lasso Regression Working

Lasso regression is fundamentally an extension of linear regression. The goal of traditional linear regression is to minimize the sum of squared differences between the observed and predicted values in order to determine the line that best fits the data points.

But the complexity of real-world data is not taken into account by linear regression, particularly when there are many factors.

### 1\. **Ordinary Least Squares (OLS) Regression**

Lasso regression builds on the [**Ordinary Least Squares (OLS) Regression**](https://www.geeksforgeeks.org/ordinary-least-squares-and-ridge-regression-variance-in-scikit-learn/) method by adding a penalty term. The basic equation for OLS is:

min RSS=Σ(yᵢ−y^ᵢ)²RSS = Σ(yᵢ - ŷᵢ)²RSS=Σ(yᵢ−y^​ᵢ)²

Where

- yiy\_iyi​ is the observed value.
- y^ᵢŷᵢy^​ᵢ is the predicted value for each data point iii.

### **2\. Penalty Term for Lasso Regression**

In Lasso regression, a penalty term is added to the OLS equation. The penalty is the sum of the absolute values of the coefficients. The updated cost function becomes:

RSS+λ×∑∣βi∣RSS + \\lambda \\times \\sum \|\\beta\_i\|RSS+λ×∑∣βi​∣

Where,

- βi\\beta\_iβi​ represents the coefficients of the predictors
- λ\\lambdaλ is the tuning parameter that controls the strength of the penalty. As λ\\lambdaλ increases more coefficients are pushed towards zero

### **3\. Shrinking Coefficients**:

The key feature of Lasso regression is its ability to shrink the coefficients of less significant features to zero. This effectively removes these features from the model performing **variable selection**. This is particularly useful when working with high-dimensional data where there are many predictors relative to the number of observations.

### **4\. Selecting the optimal** λ\\lambdaλ:

Selecting the correct **lambda** value is crucial. Cross-validation techniques are often employed to find the optimal value, balancing model complexity and predictive performance.

The primary objective of Lasso regression is to minimize the [**residual sum of squares (RSS)**](https://www.geeksforgeeks.org/residual-sum-of-squares/) along with a penalty term multiplied by the sum of the absolute values of the coefficients.

![bias-varianec-trade-off-copy](https://media.geeksforgeeks.org/wp-content/uploads/20240515184421/bias-varianec-trade-off-copy.webp)Graphical Representation of Lasso Regression

In the plot, the equation for the Lasso Regression of cost function combines the residual sum of squares (RSS) and an L1 penalty on the coefficients βjβ\_jβj​.

- **RSS measures:** The squared difference between the expected and actual values is measured.
- **L1 penalty:** Penalizes the coefficients' absolute values, bringing some of them to zero and simplifying the model. The L1 penalty's strength is managed via the lambda term. Stronger penalties result from greater lambdas which may both increase the RSS and make the model having more coefficients equal to zero.
- **y-axis:** represents the value of the cost function which Lasso Regression tries to minimize.
- **x-axis:** represents the value of the lambda (λ) parameter which controls the strength of the L1 penalty in the cost function.
- **Green to orange curve:** This curve depicts how the cost function (y-axis) changes with increasing lambda (x-axis). As lambda increases the curve transitions from green to orange. This represents the cost function value going up as the L1 penalty becomes stronger forcing more coefficients to zero.

## When to use Lasso Regression

Lasso Regression is particularly useful in the following situations:

- **Feature Selection**: It automatically selects the most important features by reducing the coefficients of less significant features to zero.
- **Collinearity:** When there is multicollinearity, it can help us by reducing the coefficients of correlated variables and selecting only one of them.
- **Regularization**: It helps preventing overfitting by penalizing large coefficients which is especially useful when the number of predictors is large.
- **Interpretability**: Compared to conventional linear regression models that incorporate all features, lasso regression generates a sparse models with fewer non-zero coefficients making model simpler to understand.

For its implementation you can refer to:

- [Implementation of Lasso Regression From Scratch using Python](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/)
- [Lasso Regression in R Programming](https://www.geeksforgeeks.org/lasso-regression-in-r-programming/)

## Advantages of Lasso Regression

- **Feature Selection:** Lasso regression eliminates the need to manually select the most relevant features, hence, the developed regression model becomes simpler and more explainable.
- **Regularization:** Lasso constrains large coefficients, so a less biased model is generated, which is robust and general in its predictions.
- **Interpretability:** With lasso, models are often sparsity induced, therefore, they are easier to interpret and explain, which is essential in fields like health care and finance.
- **Handles Large Feature Spaces:** Lasso lends itself to dealing with high-dimensional data like we have in genomic as well as imaging studies.

## Disadvantages of Lasso Regression

- **Selection Bias:** Lasso, might arbitrarily choose one variable in a group of highly correlated variables rather than the other, thereby yielding a biased model in the end.
- **Sensitive to Scale:** Lasso is demanding in the respect that features of different orders have a tendency to affect the regularization line and the model's precision.
- **Impact of Outliers:** Lasso can be easily affected by the outliers in the given data, resulting into the overfitting of the coefficients.
- **Model Instability:** In the environment of multiple correlated variables the lasso's selection of variable may be unstable, which results in different variable subsets each time in tiny data change.
- **Tuning Parameter Selection:** Analyzing different λ (alpha) values may be problematic and maybe solved by cross-validation.

By introducing a penalty term to the coefficients, Lasso encourages sparsity, simplifying the model and enhancing its interpretability. Understanding when and how to apply Lasso Regression can significantly improve model accuracy and prevent overfitting.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/what-is-data-architecture/)

[What is Data Architecture?](https://www.geeksforgeeks.org/what-is-data-architecture/)

[K](https://www.geeksforgeeks.org/user/kolisuszprv/)

[kolisuszprv](https://www.geeksforgeeks.org/user/kolisuszprv/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [ML-Regression](https://www.geeksforgeeks.org/tag/ml-regression/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+1 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Weighted Lasso Regression in R\\
\\
\\
In the world of data analysis and prediction, regression techniques are essential for understanding relationships between variables and making accurate forecasts. One standout method among many is Lasso regression. It not only helps in finding these relationships but also aids in creating models tha\\
\\
6 min read](https://www.geeksforgeeks.org/weighted-lasso-regression-in-r/?ref=ml_lbp)
[What is Regression Analysis?\\
\\
\\
In this article, we discuss about regression analysis, types of regression analysis, its applications, advantages, and disadvantages. What is regression?Regression Analysis is a supervised learning analysis where supervised learning is the analyzing or predicting the data based on the previously ava\\
\\
15+ min read](https://www.geeksforgeeks.org/what-is-regression-analysis/?ref=ml_lbp)
[What is Ridge Regression?\\
\\
\\
Ridge regression, also known as L2 regularization, is a technique used in linear regression to address the problem of multicollinearity among predictor variables. Multicollinearity occurs when independent variables in a regression model are highly correlated, which can lead to unreliable and unstabl\\
\\
8 min read](https://www.geeksforgeeks.org/what-is-ridge-regression/?ref=ml_lbp)
[Multi-task lasso regression\\
\\
\\
MultiTaskLasso Regression is an enhanced version of Lasso regression. MultiTaskLasso is a model provided by sklearn that is used for multiple regression problems to work together by estimating their sparse coefficients. There is the same feature for all the regression problems called tasks. This mod\\
\\
2 min read](https://www.geeksforgeeks.org/multi-task-lasso-regression/?ref=ml_lbp)
[Understanding LARS Lasso Regression\\
\\
\\
A regularization method called LARS Lasso (Least Angle Regression Lasso) is used in linear regression to decrease the number of features and enhance the model's predictive ability. It is a variation on the Lasso (Least Absolute Shrinkage and Selection Operator) regression, in which certain regressio\\
\\
8 min read](https://www.geeksforgeeks.org/understanding-lars-lasso-regression/?ref=ml_lbp)
[Local Regression in R\\
\\
\\
In this article, we will discuss what local regression is and how we implement it in the R Programming Language. What is Local Regression in R?Local regression is also known as LOESS (locally estimated scatterplot smoothing) regression. It is a flexible non-parametric method for fitting regression m\\
\\
4 min read](https://www.geeksforgeeks.org/local-regression-in-r/?ref=ml_lbp)
[Non-Linear Regression in R\\
\\
\\
Non-Linear Regression is a statistical method that is used to model the relationship between a dependent variable and one of the independent variable(s). In non-linear regression, the relationship is modeled using a non-linear equation. This means that the model can capture more complex and non-line\\
\\
6 min read](https://www.geeksforgeeks.org/non-linear-regression-in-r/?ref=ml_lbp)
[Lasso Regression in R Programming\\
\\
\\
Lasso regression is a classification algorithm that uses shrinkage in simple and sparse models(i.e models with fewer parameters). In Shrinkage, data values are shrunk towards a central point like the mean. Lasso regression is a regularized regression algorithm that performs L1 regularization which a\\
\\
11 min read](https://www.geeksforgeeks.org/lasso-regression-in-r-programming/?ref=ml_lbp)
[Weighted Ridge Regression in R\\
\\
\\
Ridge Regression is a key method used in statistics and machine learning to deal with a common problem called multicollinearity in regression analysis. It does this by adding a penalty to the regression equation, which helps to make the estimates more stable, especially when the predictor variables\\
\\
5 min read](https://www.geeksforgeeks.org/weighted-ridge-regression-in-r/?ref=ml_lbp)
[Least Angle Regression (LARS)\\
\\
\\
Regression is a supervised machine learning task that can predict continuous values (real numbers), as compared to classification, that can predict categorical or discrete values. Before we begin, if you are a beginner, I highly recommend this article. Least Angle Regression (LARS) is an algorithm u\\
\\
3 min read](https://www.geeksforgeeks.org/least-angle-regression-lars/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/what-is-lasso-regression/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=309607148.1745055830&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=996219618)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=v78603kwtecn)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=normal&cb=9w37zk30bx2a)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055830711&cv=11&fst=1745055830711&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tcfd=10000&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025~103130495~103130497&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=720&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fwhat-is-lasso-regression%2F&hn=www.googleadservices.com&frm=0&tiba=What%20is%20Lasso%20Regression%3F%20%7C%20GeeksforGeeks&npa=0&us_privacy=1---&pscdl=noapi&auid=1743192040.1745055831&uaa=x86&uab=64&uafvl=Chromium%3B131.0.6778.33%7CNot_A%2520Brand%3B24.0.0.0&uamb=0&uam=&uap=Windows&uapv=10.0&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=ItfkQiGBlJCsN5gUMmHbpLEb&size=invisible&cb=k4m37ky62bhk)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)