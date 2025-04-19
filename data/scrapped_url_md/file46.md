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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-cost-function-in-logistic-regression/?type%3Darticle%26id%3D299749&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Boosting in Machine Learning \| Boosting and AdaBoost\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/)

# Cost function in Logistic Regression in Machine Learning

Last Updated : 05 Apr, 2025

Comments

Improve

Suggest changes

18 Likes

Like

Report

Logistic Regression is one of the simplest classification algorithms we learn while exploring machine learning algorithms. In this article, we will explore cross-entropy, a cost function used for logistic regression.

## What is Logistic Regression?

[Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/) is a statistical method used for binary classification. Despite its name, it is employed for predicting the probability of an instance belonging to a particular class. It models the relationship between input features and the log odds of the event occurring, where the event is typically represented by the binary outcome (0 or 1).

The output of logistic regression is transformed using the logistic function (sigmoid), which maps any real-valued number to a value between 0 and 1. This transformed value can be interpreted as the probability of the instance belonging to the positive class.

### Why do we need Logistic Regression?

#### Even when we have a [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) algorithm, why do we need another logistic regression algorithm?

To answer this question first we need to understand the **problem behind the linear regression for the classification task.** Let’s consider an example of predicting diabetes based on sugar levels. In this example, we have a dataset with two variables: sugar levels (independent variable) and the likelihood of having diabetes (dependent variable, 1 for diabetic and 0 for non-diabetic).

![linear-regression](https://media.geeksforgeeks.org/wp-content/uploads/20231204141502/linear-regression.png)

As observed, the linear regression line (y=mx+c y=mx+c y=mx+c) fails to accurately capture the relationship between sugar levels and the likelihood of diabetes. It assumes a linear relationship, which is not suitable for this binary classification problem. **Extending this line will result in values that are less than 0 and greater than 1, which are not very useful in our classification problem.**

This is where logistic regression comes into play, using the sigmoid function to model the probability of an instance belonging to a particular class.

#### **Logistic function**

Logistic function is also known as [sigmoid function](https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/). It compressed the output of the linear regression between 0 & 1. It can be defined as:

σ=11+e−z(i)\\sigma=\\frac{1}{1+e^{-z^{(i)}}}σ=1+e−z(i)1​

- Here, z(i)=hθ=θ1xi+θ2z^{(i)} = h\_\\theta=\\theta\_1x\_i+\\theta\_2z(i)=hθ​=θ1​xi​+θ2​ is the linear combination of input features and model parameters.
- The predicted probability σ\\sigmaσ is the output of this sigmoid function.
- e is the base of the natural logarithm, approximately equal to 2.718

When we applied the logistic functions to the output of the linear regression. it can be observed like:

![](https://media.geeksforgeeks.org/wp-content/uploads/20210422175332/LRmodel.png)

As evident in the plot, sigmoid function is accurately separating the classes for binary classification tasks. Also, it produces continuous values exclusively within the 0 to 1 range, which can be employed for predictive purposes.

### What is Cost Function?

A [cost function](https://www.geeksforgeeks.org/what-is-cost-function/) is a mathematical function that calculates the difference between the target actual values (ground truth) and the values predicted by the model. A function that assesses a machine learning model’s performance also referred to as a **loss function or objective function**. Usually, the objective of a machine learning algorithm is to reduce the error or output of cost function.

**When it comes to Linear Regression**, the conventional Cost Function employed is the **Mean Squared Error**. The cost function (J) for m training samples can be written as:

J(θ)=12m∑i=1m\[z(i)–y(i)\]2=12m∑i=1m\[hθ(x(i))–y(i)\]2\\begin{aligned}J(\\theta) &= \\frac{1}{2m} \\sum\_{i = 1}^{m} \[z^{(i)} – y^{(i)}\]^{2}  \\\&= \\frac{1}{2m} \\sum\_{i = 1}^{m} \[h\_{\\theta}(x^{(i)}) – y^{(i)}\]^{2} \\end{aligned}J(θ)​=2m1​i=1∑m​\[z(i)–y(i)\]2=2m1​i=1∑m​\[hθ​(x(i))–y(i)\]2​

where,

- y(i)y^{(i)}


y(i) is the actual value of the target variable for the i-th training example.
- z(i)=hθ(x(i))z^{(i)}=h\_\\theta(x^{(i)})z(i)=hθ​(x(i)) is the predicted value of the target variable for the i-th training example, calculated using the linear regression model with parameters θ.
- x(i)x^{(i)}


x(i) is the i-th training example.
- m is number of training examples.

Plotting this specific error function against the linear regression model’s weight parameters results in a convex shape. This convexity is important because it allows the [Gradient Descent](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) Algorithm to be used to optimize the function. Using this algorithm, we can locate the global minima on the graph and modify the model’s weights to systematically lower the error. In essence, it’s a means of optimizing the model to raise its accuracy in making predictions.

![Gradient-Descent-for-ML-Linear-Regression](https://media.geeksforgeeks.org/wp-content/uploads/20231204224802/Gradient-Descent-for-ML-Linear-Regression.webp)

### Why Mean Squared Error cannot be used as cost function for Logistic Regression

Let’s consider the [Mean Squared Error (MSE)](https://www.geeksforgeeks.org/python-mean-squared-error/) as a cost function, but it is not suitable for logistic regression due to its nonlinearity introduced by the sigmoid function.

In logistic regression, if we substitute the sigmoid function into the above MSE equation, we get:

J(θ)=12m∑i=1m\[σ(i)–y(i)\]2=12m∑i=1m(11+e−z(i)–y(i))2=12m∑i=1m(11+e−(θ1x(i)+θ2)–y(i))2\\begin{aligned} J(\\theta) &= \\frac{1}{2m} \\sum\_{i = 1}^{m}\[\\sigma^{(i)} – y^{(i)}\]^{2} \\\&= \\frac{1}{2m} \\sum\_{i=1}^{m} \\left(\\frac{1}{1 + e^{-z^{(i)}}} – y^{(i)}\\right)^2\\\&= \\frac{1}{2m} \\sum\_{i=1}^{m} \\left(\\frac{1}{1 + e^{-(\\theta\_1x^{(i)}+\\theta\_2)}} – y^{(i)}\\right)^2\\end{aligned}J(θ)​=2m1​i=1∑m​\[σ(i)–y(i)\]2=2m1​i=1∑m​(1+e−z(i)1​–y(i))2=2m1​i=1∑m​(1+e−(θ1​x(i)+θ2​)1​–y(i))2​

The equation 11+e−z\\frac{1}{1 + e^{-z}}1+e−z1​ is a nonlinear transformation, and evaluating this term within the Mean Squared Error formula results in a non-convex cost function. A non-convex function, have multiple [local minima](https://www.geeksforgeeks.org/local-and-global-optimum-in-uni-variate-optimization/) which can make it difficult to optimize using traditional [gradient descent](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) algorithms as shown below.

![Cost_func_Non_Convex](https://media.geeksforgeeks.org/wp-content/uploads/20231205004205/Cost_func_Non_Convex.jpg)

Let’s break down the nonlinearity introduced by the sigmoid function and the impact on the cost function:

### **Sigmoid Function Nonlinearity**

#### **1\.** When _z_ is large and positive, e-z becomes very small, and11+e−z\\frac{1}{1+e^{-z}}  1+e−z1​ approaches 1.

Example:

> when z=5,
>
> 11+e−5=11+1e5=1(e5+1e5)=e5e5+1=0.9933071490757153 \\begin{aligned} \\frac{1}{1+e^{-5}}&= \\frac{1}{{1+\\frac{1}{e^{5}}}} \\\ &=\\frac{1}{(\\frac{e^{5}+1}{e^{5}})} \\\ &=\\frac{e^{5}}{e^{5}+1} \\\& = 0.9933071490757153 \\end{aligned}1+e−51​​=1+e51​1​=(e5e5+1​)1​=e5+1e5​=0.9933071490757153​
>
> Which is close to 1 indicating that the sigmoid function, when applied to a large positive value like 5, outputs a probability close to 1, suggesting a high confidence in the positive class.

**2\. When z is large and negative,** e-z **dominates, and** 11+e−z\\frac{1}{1+e^{-z}}1+e−z1​ **approaches 0.**

Example,

> when z= -5, the expression becomes,
>
> 11+e−(−5)=11+e5=0.0066928509242848554 \\begin{aligned} \\frac{1}{1+e^{-(-5)}}&= \\frac{1}{1+e^{5}} \\\& = 0.0066928509242848554 \\end{aligned}1+e−(−5)1​​=1+e51​=0.0066928509242848554​
>
> Which is close to 0 indicating that sigmoid function, when applied to a large negative value like -5, outputs a probability close to 0. In other words, when z is a large negative number, the exponential term eze^zez dominates the denominator, causing the sigmoid function to approach 0.

### **Squaring within the MSE Formula**

The Mean Squared Error (MSE) formula involves squaring the difference between the predicted probability and the true label. In general Squaring the term 11+e−z−y\\frac{1}{1+e^{-z}}-y1+e−z1​−y magnifies the errors when the predicted probability is far from the true label. but if the difference between the actual and predicted is in between 0 & 1 squaring the values will get lesser values.

**Example 1:**

> If the true label y is 1 and the predicted probability 11+e−z=0.2\\frac{1}{{1+{e^{-z}}}} = 0.21+e−z1​=0.2,
>
> Squaring the difference :
>
> (1−0.2)2=0.82=0.64<0.8(1-0.2)^2= 0.8^2 = 0.64 <0.8(1−0.2)2=0.82=0.64<0.8
>
> Gives a lesser error than the predicted probability was, say, 0.8.
>
> **The squaring operation intensifies the impact of misclassifications, especially when** y^\\hat{y}y^​ **​ is close to 0 or 1.**

While dealing with an optimization problem with Non-convex graph we face the problem of getting stuck at the local minima instead of the global minima. The presence of multiple local minima can make it challenging to find the optimal solution for a machine learning model. If the model gets trapped in a local minimum, it will not achieve the best possible performance. That’s where comes [Log Loss or Cross Entropy Function](https://www.geeksforgeeks.org/sigmoid-cross-entropy-function-of-tensorflow/) most important term in the case of logistic regression.

### Log Loss for Logistic regression

[**Log loss**](https://www.geeksforgeeks.org/ml-log-loss-and-mean-squared-error/) **is a classification evaluation metric** that is used to compare different models which we build during the process of model development. It is considered one of the efficient metrics for evaluation purposes while dealing with the soft probabilities predicted by the model.

The log of corrected probabilities, in logistic regression, is obtained by taking the natural logarithm (base _e_) of the predicted probabilities.

log loss=ln⁡σ=ln⁡11+e−z\\begin{aligned}\\text{log loss} &= \\ln{\\sigma}\\\ &= \\ln{\\frac{1}{1+e^{-z}}}\\end{aligned}log loss​=lnσ=ln1+e−z1​​

> Let’s find the log odds for the examples, z=5
>
> ln⁡(z=5)=ln⁡11+e−5=ln⁡(11+1e5)=ln⁡(e51+e5)=ln⁡e5−ln⁡(1+e5)=5–ln⁡(1+2.785)=5–ln⁡(1+2.718285)=5−5.006712007735388≈−0.00671200 \\begin{aligned} \\ln{(z=5)}&=\\ln{\\frac{1}{1+e^{-5}}}\\\ &= \\ln{\\left (\\frac{1}{{1+\\frac{1}{e^{5}}}} \\right )} \\\ &=\\ln{\\left (\\frac{e^{5}}{1+e^{5}} \\right )} \\\ &=\\ln{e^5}-\\ln{(1+e^5)} \\\ &= 5 – \\ln{(1+2.78^5)}\\\ &= 5 – \\ln{(1+2.71828^5)}\\\ &= 5-5.006712007735388\\\&\\approx -0.00671200\\end{aligned}ln(z=5)​=ln1+e−51​=ln(1+e51​1​)=ln(1+e5e5​)=lne5−ln(1+e5)=5–ln(1+2.785)=5–ln(1+2.718285)=5−5.006712007735388≈−0.00671200​
>
> and for z=-5,
>
> ln⁡(z=−5)=ln⁡11+e−(−5)=ln⁡(11+e5)=ln⁡1−ln⁡(1+e5)=0–ln⁡(1+2.785)=–ln⁡(1+2.718285)≈−5.00671200 \\begin{aligned} \\ln{(z=-5)}&=\\ln{\\frac{1}{1+e^{-(-5)}}}\\\ &= \\ln{\\left (\\frac{1}{1+e^{5}} \\right )} \\\ &=\\ln{1}-\\ln{(1+e^5)} \\\ &= 0 – \\ln{(1+2.78^5)}\\\ &= – \\ln{(1+2.71828^5)}\\\&\\approx -5.00671200\\end{aligned}ln(z=−5)​=ln1+e−(−5)1​=ln(1+e51​)=ln1−ln(1+e5)=0–ln(1+2.785)=–ln(1+2.718285)≈−5.00671200​
>
> These log values are negative. In order to maintain the common convention that lower loss scores are better, we take the negative average of these values to deal with the negative sign.

Hence, The Log Loss can be summarized with the following formula:

J=−∑i=1myilog⁡(hθ(xi))+(1−yi)log⁡(1−hθ(xi))J = -\\sum\_{i=1}^{m}y\_i\\log \\left ( h\_\\theta\\left ( x\_i \\right ) \\right ) + \\left ( 1-y\_i \\right )\\log \\left (1- h\_\\theta\\left ( x\_i \\right ) \\right )J=−∑i=1m​yi​log(hθ​(xi​))+(1−yi​)log(1−hθ​(xi​))

where,

- m is the number of training examples
- yiy\_iyi​ is the true class label for the i-th example (either 0 or 1).
- hθ(xi)h\_\\theta(x\_i)hθ​(xi​) is the predicted probability for the i-th example, as calculated by the logistic regression model.
- θ\\thetaθ is the model parameters

The first term in the sum represents the cross-entropy for the positive class (yi=1y\_i = 1yi​=1), and the second term represents the cross-entropy for the negative class (yi=0y\_i = 0yi​=0). The goal of logistic regression is to minimize the cost function by adjusting the model parametersθ\\thetaθ

In summary:

- Calculate predicted probabilities using the sigmoid function.
- Apply the natural logarithm to the corrected probabilities.
- Sum up and average the log values, then negate the result to get the Log Loss.

### Cost function for Logistic Regression

Cost(hθ(x),y)={−log(hθ(x))ify=1−log(1−hθ(x))ify=0Cost(h\_{\\theta}(x),y) = \\left\\{\\begin{matrix} -log(h\_{\\theta}(x)) & if&y=1\\\ -log(1-h\_{\\theta}(x))& if& y = 0 \\end{matrix}\\right. Cost(hθ​(x),y)={−log(hθ​(x))−log(1−hθ​(x))​ifif​y=1y=0​

- **Case 1:** If y = 1, that is the true label of the class is 1. Cost = 0 if the predicted value of the label is 1 as well. But as hθ(x) deviates from 1 and approaches 0 cost function increases exponentially and tends to infinity which can be appreciated from the below graph as well.
- **Case 2:** If y = 0, that is the true label of the class is 0. Cost = 0 if the predicted value of the label is 0 as well. But as hθ(x) deviates from 0 and approaches 1 cost function increases exponentially and tends to infinity which can be appreciated from the below graph as well.

![download-(4)](https://media.geeksforgeeks.org/wp-content/uploads/20231205010859/download-(4).png)

With the modification of the cost function, we have achieved a loss function that penalizes the model weights more and more as the predicted value of the label deviates more and more from the actual label.

### Conclusion

The choice of cost function, log loss or cross-entropy, is significant for logistic regression. It quantifies the disparity between predicted probabilities and actual outcomes, providing a measure of how well the model aligns with the ground truth.

### Frequently Asked Questions (FAQs)

#### 1\. What is the difference between the cost function and the loss function for logistic regression?

> The terms are often used interchangeably, but the cost function typically refers to the average loss over the entire dataset, while the loss function calculates the error for a single data point.

#### 2.Why logistic regression cost function is non convex?

> The sigmoid function introduces non-linearity, resulting in a non-convex cost function. It has multiple local minima, making optimization challenging, as traditional gradient descent may converge to suboptimal solutions.

#### 3.What is a cost function in simple terms?

> A cost function measures the disparity between predicted values and actual values in a machine learning model. It quantifies how well the model aligns with the ground truth, guiding optimization.

#### 4.Is the cost function for logistic regression always negative?

> No, the cost function for logistic regression is not always negative. It includes terms like -log(h(x)) and -log(1 – h(x)), and the overall value depends on the predicted probabilities and actual labels, yielding positive or negative values.

#### 5.Why only sigmoid function is used in logistic regression?

> The sigmoid function maps real-valued predictions to probabilities between 0 and 1, facilitating the interpretation of results as probabilities. Its range and smoothness make it suitable for binary classification, ensuring stable optimization.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/)

[Boosting in Machine Learning \| Boosting and AdaBoost](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/)

[![author](https://media.geeksforgeeks.org/auth/profile/71uvm7pynx4cnk0rtuk6)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

[mohit gupta\_omg :)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

Follow

18

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Logistic Regression in Machine Learning\\
\\
\\
In our previous discussion, we explored the fundamentals of machine learning and walked through a hands-on implementation of Linear Regression. Now, let's take a step forward and dive into one of the first and most widely used classification algorithms â€” Logistic Regression What is Logistic Regressi\\
\\
13 min read](https://www.geeksforgeeks.org/understanding-logistic-regression/?ref=ml_lbp)
[Loss function for Linear regression in Machine Learning\\
\\
\\
The loss function quantifies the disparity between the prediction value and the actual value. In the case of linear regression, the aim is to fit a linear equation to the observed data, the loss function evaluate the difference between the predicted value and true values. By minimizing this differen\\
\\
7 min read](https://www.geeksforgeeks.org/loss-function-for-linear-regression/?ref=ml_lbp)
[Naive Bayes vs Logistic Regression in Machine Learning\\
\\
\\
Logistic Regression is a statistical model that predicts the probability of a binary outcome by modeling the relationship between the dependent variable and one or more independent variables. Despite its name, Logistic Regression is used for classification rather than regression tasks. It assumes a\\
\\
7 min read](https://www.geeksforgeeks.org/naive-bayes-vs-logistic-regression-in-machine-learning/?ref=ml_lbp)
[Classification vs Regression in Machine Learning\\
\\
\\
Classification and regression are two primary tasks in supervised machine learning, where key difference lies in the nature of the output: classification deals with discrete outcomes (e.g., yes/no, categories), while regression handles continuous values (e.g., price, temperature). Both approaches re\\
\\
5 min read](https://www.geeksforgeeks.org/ml-classification-vs-regression/?ref=ml_lbp)
[Linear Regression in Machine learning\\
\\
\\
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It provides valuable insights for prediction and data analysis. This article will explore its types, assumptions, implementation, advantages and evaluation met\\
\\
15+ min read](https://www.geeksforgeeks.org/ml-linear-regression/?ref=ml_lbp)
[Regression in machine learning\\
\\
\\
Regression in machine learning refers to a supervised learning technique where the goal is to predict a continuous numerical value based on one or more independent features. It finds relationships between variables so that predictions can be made. we have two types of variables present in regression\\
\\
5 min read](https://www.geeksforgeeks.org/regression-in-machine-learning/?ref=ml_lbp)
[CART (Classification And Regression Tree) in Machine Learning\\
\\
\\
CART( Classification And Regression Trees) is a variation of the decision tree algorithm. It can handle both classification and regression tasks. Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train Decision Trees (also called â€œgrowingâ€ trees). CART was first produced b\\
\\
11 min read](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/?ref=ml_lbp)
[Logistic Regression vs K Nearest Neighbors in Machine Learning\\
\\
\\
Machine learning algorithms play a crucial role in training the data and decision-making processes. Logistic Regression and K Nearest Neighbors (KNN) are two popular algorithms in machine learning used for classification tasks. In this article, we'll delve into the concepts of Logistic Regression an\\
\\
4 min read](https://www.geeksforgeeks.org/logistic-regression-vs-k-nearest-neighbors-in-machine-learning/?ref=ml_lbp)
[What is the Cost Function in Linear Regression?\\
\\
\\
Linear Regression is a method used to make accurate predictions by finding the best-fit line for the model. However when we first create a model the predictions may not match the actual data. To address this a cost function is used to measure how far off these predictions are. This article explore t\\
\\
7 min read](https://www.geeksforgeeks.org/what-is-the-cost-function-in-linear-regression/?ref=ml_lbp)
[Pros and Cons of Decision Tree Regression in Machine Learning\\
\\
\\
Decision tree regression is a widely used algorithm in machine learning for predictive modeling tasks. It is a powerful tool that can handle both classification and regression problems, making it versatile for various applications. However, like any other algorithm, decision tree regression has its\\
\\
5 min read](https://www.geeksforgeeks.org/pros-and-cons-of-decision-tree-regression-in-machine-learning/?ref=ml_lbp)

Like18

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-cost-function-in-logistic-regression/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=853399525.1745055858&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1931749630)