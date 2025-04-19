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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/understanding-logistic-regression/?type%3Darticle%26id%3D146807&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Understanding Activation Functions in Depth\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

# Logistic Regression in Machine Learning

Last Updated : 03 Feb, 2025

Comments

Improve

Suggest changes

103 Likes

Like

Report

In our previous discussion, we explored the fundamentals of machine learning and walked through a hands-on implementation of **Linear Regression**. Now, let’s take a step forward and dive into one of the first and most widely used classification algorithms — **Logistic Regression**

## What is Logistic Regression?

**Logistic regression** is a **supervised machine learning algorithm** used for **classification tasks** where the goal is to predict the probability that an instance belongs to a given class or not. Logistic regression is a statistical algorithm which analyze the relationship between two data factors. The article explores the fundamentals of logistic regression, it’s types and implementations.

Logistic regression is used for binary [classification](https://www.geeksforgeeks.org/getting-started-with-classification/) where we use [sigmoid function](https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/), that takes input as independent variables and produces a probability value between 0 and 1.

For example, we have two classes Class 0 and Class 1 if the value of the logistic function for an input is greater than 0.5 (threshold value) then it belongs to Class 1 otherwise it belongs to Class 0. It’s referred to as regression because it is the extension of [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) but is mainly used for classification problems.

### Key Points:

- Logistic regression predicts the output of a categorical dependent variable. Therefore, the outcome must be a categorical or discrete value.
- It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.
- In Logistic regression, instead of fitting a regression line, we fit an “S” shaped logistic function, which predicts two maximum values (0 or 1).

## Types of Logistic Regression

On the basis of the categories, Logistic Regression can be classified into three types:

1. **Binomial:** In binomial Logistic regression, there can be only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc.
2. **Multinomial:** In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as “cat”, “dogs”, or “sheep”
3. **Ordinal:** In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as “low”, “Medium”, or “High”.

## Assumptions of Logistic Regression

We will explore the assumptions of logistic regression as understanding these assumptions is important to ensure that we are using appropriate application of the model. The assumption include:

1. Independent observations: Each observation is independent of the other. meaning there is no correlation between any input variables.
2. Binary dependent variables: It takes the assumption that the dependent variable must be binary or dichotomous, meaning it can take only two values. For more than two categories SoftMax functions are used.
3. Linearity relationship between independent variables and log odds: The relationship between the independent variables and the log odds of the dependent variable should be linear.
4. No outliers: There should be no outliers in the dataset.
5. Large sample size: The sample size is sufficiently large

## **Understanding Sigmoid Function**

So far, we’ve covered the basics of logistic regression, but now let’s focus on the most important function that forms the core of logistic regression **.**

- The **sigmoid function** is a mathematical function used to map the predicted values to probabilities.
- It maps any real value into another value within a range of **0 and 1**. The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve like the “ **S**” form.
- The S-form curve is called the **Sigmoid function or the logistic function.**
- In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0.

## How does Logistic Regression work?

The logistic regression model transforms the [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) function continuous value output into categorical value output using a sigmoid function, which maps any real-valued set of independent variables input into a value between 0 and 1. This function is known as the logistic function.

Let the independent input features be:

X=\[x11…x1mx21…x2m⋮⋱⋮xn1…xnm\]X = \\begin{bmatrix} x\_{11}  & … & x\_{1m}\\\ x\_{21}  & … & x\_{2m} \\\  \\vdots & \\ddots  & \\vdots  \\\ x\_{n1}  & … & x\_{nm} \\end{bmatrix}X=​x11​x21​⋮xn1​​……⋱…​x1m​x2m​⋮xnm​​​

and the dependent variable is Y having only binary value i.e. 0 or 1.

Y={0 if Class11 if Class2Y = \\begin{cases} 0 & \\text{ if } Class\\;1 \\\ 1 & \\text{ if } Class\\;2 \\end{cases}Y={01​ if Class1 if Class2​

then, apply the multi-linear function to the input variables X.

z=(∑i=1nwixi)+bz = \\left(\\sum\_{i=1}^{n} w\_{i}x\_{i}\\right) + bz=(∑i=1n​wi​xi​)+b

Here xix\_ixi​ is the ith observation of X, wi=\[w1,w2,w3,⋯,wm\]w\_i = \[w\_1, w\_2, w\_3, \\cdots,w\_m\]wi​=\[w1​,w2​,w3​,⋯,wm​\] is the weights or Coefficient, and b is the bias term also known as intercept. simply this can be represented as the dot product of weight and bias.

z=w⋅X+bz = w\\cdot X +bz=w⋅X+b

whatever we discussed above is the [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/).

### Sigmoid Function

Now we use the [sigmoid function](https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/) where the input will be z and we find the probability between 0 and 1. i.e. predicted y.

σ(z)=11+e−z\\sigma(z) = \\frac{1}{1+e^{-z}}σ(z)=1+e−z1​

![sigmoid function - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20190522162153/sigmoid-function-300x138.png)

Sigmoid function

As shown above, the figure sigmoid function converts the continuous variable data into the [probability](https://www.geeksforgeeks.org/probability-gq/) i.e. between 0 and 1.

- σ(z)\\sigma(z)







σ(z) tends towards 1 as z→∞z\\rightarrow\\infty







z→∞
- σ(z)\\sigma(z)







σ(z) tends towards 0 as z→−∞z\\rightarrow-\\infty







z→−∞
- σ(z)\\sigma(z)







σ(z) is always bounded between 0 and 1

where the probability of being a class can be measured as:

P(y=1)=σ(z)P(y=0)=1−σ(z)P(y=1) = \\sigma(z) \\\ P(y=0) = 1-\\sigma(z)P(y=1)=σ(z)P(y=0)=1−σ(z)

### Equation of Logistic Regression:

The odd is the ratio of something occurring to something not occurring. it is different from probability as the probability is the ratio of something occurring to everything that could possibly occur. so odd will be:

p(x)1−p(x)=ez\\frac{p(x)}{1-p(x)}  = e^z1−p(x)p(x)​=ez

Applying natural log on odd. then log odd will be:

log⁡\[p(x)1−p(x)\]=zlog⁡\[p(x)1−p(x)\]=w⋅X+bp(x)1−p(x)=ew⋅X+b⋯Exponentiate both sidesp(x)=ew⋅X+b⋅(1−p(x))p(x)=ew⋅X+b−ew⋅X+b⋅p(x))p(x)+ew⋅X+b⋅p(x))=ew⋅X+bp(x)(1+ew⋅X+b)=ew⋅X+bp(x)=ew⋅X+b1+ew⋅X+b\\begin{aligned}\\log \\left\[\\frac{p(x)}{1-p(x)} \\right\] &= z \\\ \\log \\left\[\\frac{p(x)}{1-p(x)} \\right\] &= w\\cdot X +b\\\ \\frac{p(x)}{1-p(x)}&= e^{w\\cdot X +b} \\;\\;\\cdots\\text{Exponentiate both sides}\\\ p(x) &=e^{w\\cdot X +b}\\cdot (1-p(x))\\\p(x) &=e^{w\\cdot X +b}-e^{w\\cdot X +b}\\cdot p(x))\\\p(x)+e^{w\\cdot X +b}\\cdot p(x))&=e^{w\\cdot X +b}\\\p(x)(1+e^{w\\cdot X +b}) &=e^{w\\cdot X +b}\\\p(x)&= \\frac{e^{w\\cdot X +b}}{1+e^{w\\cdot X +b}}\\end{aligned}log\[1−p(x)p(x)​\]log\[1−p(x)p(x)​\]1−p(x)p(x)​p(x)p(x)p(x)+ew⋅X+b⋅p(x))p(x)(1+ew⋅X+b)p(x)​=z=w⋅X+b=ew⋅X+b⋯Exponentiate both sides=ew⋅X+b⋅(1−p(x))=ew⋅X+b−ew⋅X+b⋅p(x))=ew⋅X+b=ew⋅X+b=1+ew⋅X+bew⋅X+b​​

then the final logistic regression equation will be:

p(X;b,w)=ew⋅X+b1+ew⋅X+b=11+e−w⋅X+bp(X;b,w) = \\frac{e^{w\\cdot X +b}}{1+e^{w\\cdot X +b}} = \\frac{1}{1+e^{-w\\cdot X +b}}p(X;b,w)=1+ew⋅X+bew⋅X+b​=1+e−w⋅X+b1​

### Likelihood Function for Logistic Regression

The predicted probabilities will be:

- for y=1 The predicted probabilities will be: p(X;b,w) = p(x)
- for y = 0 The predicted probabilities will be: 1-p(X;b,w) = 1-p(x)

L(b,w)=∏i=1np(xi)yi(1−p(xi))1−yiL(b,w) = \\prod\_{i=1}^{n}p(x\_i)^{y\_i}(1-p(x\_i))^{1-y\_i}L(b,w)=∏i=1n​p(xi​)yi​(1−p(xi​))1−yi​

Taking natural logs on both sides

log⁡(L(b,w))=∑i=1nyilog⁡p(xi)+(1−yi)log⁡(1−p(xi))=∑i=1nyilog⁡p(xi)+log⁡(1−p(xi))−yilog⁡(1−p(xi))=∑i=1nlog⁡(1−p(xi))+∑i=1nyilog⁡p(xi)1−p(xi=∑i=1n−log⁡1−e−(w⋅xi+b)+∑i=1nyi(w⋅xi+b)=∑i=1n−log⁡1+ew⋅xi+b+∑i=1nyi(w⋅xi+b)\\begin{aligned}\\log(L(b,w)) &= \\sum\_{i=1}^{n} y\_i\\log p(x\_i)\\;+\\; (1-y\_i)\\log(1-p(x\_i)) \\\ &=\\sum\_{i=1}^{n} y\_i\\log p(x\_i)+\\log(1-p(x\_i))-y\_i\\log(1-p(x\_i)) \\\ &=\\sum\_{i=1}^{n} \\log(1-p(x\_i)) +\\sum\_{i=1}^{n}y\_i\\log \\frac{p(x\_i)}{1-p(x\_i} \\\ &=\\sum\_{i=1}^{n} -\\log1-e^{-(w\\cdot x\_i+b)} +\\sum\_{i=1}^{n}y\_i (w\\cdot x\_i +b) \\\ &=\\sum\_{i=1}^{n} -\\log1+e^{w\\cdot x\_i+b} +\\sum\_{i=1}^{n}y\_i (w\\cdot x\_i +b) \\end{aligned}log(L(b,w))​=i=1∑n​yi​logp(xi​)+(1−yi​)log(1−p(xi​))=i=1∑n​yi​logp(xi​)+log(1−p(xi​))−yi​log(1−p(xi​))=i=1∑n​log(1−p(xi​))+i=1∑n​yi​log1−p(xi​p(xi​)​=i=1∑n​−log1−e−(w⋅xi​+b)+i=1∑n​yi​(w⋅xi​+b)=i=1∑n​−log1+ew⋅xi​+b+i=1∑n​yi​(w⋅xi​+b)​

### Gradient of the log-likelihood function

To find the maximum likelihood estimates, we differentiate w.r.t w,

∂J(l(b,w)∂wj=−∑i=nn11+ew⋅xi+bew⋅xi+bxij+∑i=1nyixij=−∑i=nnp(xi;b,w)xij+∑i=1nyixij=∑i=nn(yi−p(xi;b,w))xij\\begin{aligned} \\frac{\\partial J(l(b,w)}{\\partial w\_j}&=-\\sum\_{i=n}^{n}\\frac{1}{1+e^{w\\cdot x\_i+b}}e^{w\\cdot x\_i+b} x\_{ij} +\\sum\_{i=1}^{n}y\_{i}x\_{ij} \\\&=-\\sum\_{i=n}^{n}p(x\_i;b,w)x\_{ij}+\\sum\_{i=1}^{n}y\_{i}x\_{ij} \\\&=\\sum\_{i=n}^{n}(y\_i -p(x\_i;b,w))x\_{ij} \\end{aligned}∂wj​∂J(l(b,w)​​=−i=n∑n​1+ew⋅xi​+b1​ew⋅xi​+bxij​+i=1∑n​yi​xij​=−i=n∑n​p(xi​;b,w)xij​+i=1∑n​yi​xij​=i=n∑n​(yi​−p(xi​;b,w))xij​​

## **Terminologies involved in Logistic Regression**

Here are some common terms involved in logistic regression:

- **Independent variables:** The input characteristics or predictor factors applied to the dependent variable’s predictions.
- **Dependent variable:** The target variable in a logistic regression model, which we are trying to predict.
- **Logistic function:** The formula used to represent how the independent and dependent variables relate to one another. The logistic function transforms the input variables into a probability value between 0 and 1, which represents the likelihood of the dependent variable being 1 or 0.
- **Odds:** It is the ratio of something occurring to something not occurring. it is different from probability as the probability is the ratio of something occurring to everything that could possibly occur.
- **Log-odds:** The log-odds, also known as the logit function, is the natural logarithm of the odds. In logistic regression, the log odds of the dependent variable are modeled as a linear combination of the independent variables and the intercept.
- **Coefficient:** The logistic regression model’s estimated parameters, show how the independent and dependent variables relate to one another.
- **Intercept:** A constant term in the logistic regression model, which represents the log odds when all independent variables are equal to zero.
- [**Maximum likelihood estimation**](https://www.geeksforgeeks.org/probability-density-estimation-maximum-likelihood-estimation/) **:** The method used to estimate the coefficients of the logistic regression model, which maximizes the likelihood of observing the data given the model

## Code Implementation for Logistic Regression

So far, we’ve covered the basics of logistic regression with all the theoritical concepts, but now let’s focus on the hands on code implementation part which makes you understand the logistic regression more clearly. We will dicuss **Binomial Logistic regression** and **Multinomial Logistic Regression** one by one.

### **Binomial Logistic regression:**

Target variable can have only 2 possible types: “0” or “1” which may represent “ **win**” vs “ **loss**”, “ **pass**” vs “ **fail**”, “ **dead**” vs “ **alive**”, etc., in this case, sigmoid functions are used, which is already discussed above.

Importing necessary libraries based on the requirement of model. This Python code shows how to use the **breast cancer dataset** to implement a Logistic Regression model for classification.

Python`
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#load the following dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
`

**Output**:

```
Logistic Regression model accuracy (in %): 96.49%
```

This code loads the **breast cancer dataset** from s **cikit-learn**, splits it into training and testing sets, and then trains a Logistic Regression model on the training data. The model is used to predict the labels for the test data, and the accuracy of these predictions is calculated by comparing the predicted values with the actual labels from the test set. Finally, the accuracy is printed as a percentage.

### **Multinomial Logistic Regression:**

Target variable can have 3 or more possible types which are not ordered (i.e. types have no quantitative significance) like “ **disease A**” vs “ **disease B**” vs “ **disease C**”.

In this case, the softmax function is used in place of the sigmoid function. [Softmax function](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/) for K classes will be:

softmax(zi)=ezi∑j=1Kezj\\text{softmax}(z\_i) =\\frac{ e^{z\_i}}{\\sum\_{j=1}^{K}e^{z\_{j}}}softmax(zi​)=∑j=1K​ezj​ezi​​

Here, **K** represents the number of elements in the vector z, and i, j iterates over all the elements in the vector.

Then the probability for class c will be:

P(Y=c∣X→=x)=ewc⋅x+bc∑k=1Kewk⋅x+bkP(Y=c \| \\overrightarrow{X}=x) = \\frac{e^{w\_c \\cdot x + b\_c}}{\\sum\_{k=1}^{K}e^{w\_k \\cdot x + b\_k}}P(Y=c∣X=x)=∑k=1K​ewk​⋅x+bk​ewc​⋅x+bc​​

In Multinomial Logistic Regression, the output variable can have **more than two possible discrete outputs**. Consider the Digit Dataset.

Python`
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
`

**Output:**

```
Logistic Regression model accuracy(in %): 96.66%
```

## How to Evaluate Logistic Regression Model?

So far, we’ve covered the implementation of logistic regression. Now, let’s dive into the evaluation of logistic regression and understand why it’s important

Evaluating the model helps us assess the model’s performance and ensure it generalizes well to new data

We can evaluate the logistic regression model using the following metrics:

- **Accuracy:** [Accuracy](https://www.geeksforgeeks.org/techniques-to-evaluate-accuracy-of-classifier-in-data-mining/) provides the proportion of correctly classified instances.

Accuracy=TruePositives+TrueNegativesTotalAccuracy = \\frac{True \\, Positives + True \\, Negatives}{Total}

Accuracy=TotalTruePositives+TrueNegatives​

- **Precision:** [Precision](https://www.geeksforgeeks.org/precision-recall-and-f1-score-using-r/) focuses on the accuracy of positive predictions.

Precision=TruePositivesTruePositives+FalsePositivesPrecision = \\frac{True \\, Positives }{True\\, Positives + False \\, Positives}

Precision=TruePositives+FalsePositivesTruePositives​

- **Recall (Sensitivity or True Positive Rate):** [Recall](https://www.geeksforgeeks.org/precision-and-recall-in-information-retrieval/) measures the proportion of correctly predicted positive instances among all actual positive instances.

Recall=TruePositivesTruePositives+FalseNegativesRecall = \\frac{ True \\, Positives}{True\\, Positives + False \\, Negatives}

Recall=TruePositives+FalseNegativesTruePositives​

- **F1 Score:** [F1 score](https://www.geeksforgeeks.org/f1-score-in-machine-learning/) is the harmonic mean of precision and recall.

F1Score=2∗Precision∗RecallPrecision+RecallF1 \\, Score = 2 \* \\frac{Precision \* Recall}{Precision + Recall}

F1Score=2∗Precision+RecallPrecision∗Recall​

- **Area Under the Receiver Operating Characteristic Curve (AUC-ROC):** The ROC curve plots the true positive rate against the false positive rate at various thresholds. [AUC-ROC](https://www.geeksforgeeks.org/auc-roc-curve/) measures the area under this curve, providing an aggregate measure of a model’s performance across different classification thresholds.
- **Area Under the Precision-Recall Curve (AUC-PR):** Similar to AUC-ROC, [AUC-PR](https://www.geeksforgeeks.org/precision-recall-curve-ml/) measures the area under the precision-recall curve, providing a summary of a model’s performance across different precision-recall trade-offs.

## Differences Between Linear and Logistic Regression

Now lets dive into the key differences of Linear Regression and Logistic Regression and evaluate that how they are different from each other.

The difference between linear regression and logistic regression is that linear regression output is the continuous value that can be anything while logistic regression predicts the probability that an instance belongs to a given class or not.

| Linear Regression | Logistic Regression |
| --- | --- |
| Linear regression is used to predict the continuous dependent variable using a given set of independent variables. | Logistic regression is used to predict the categorical dependent variable using a given set of independent variables. |
| Linear regression is used for solving regression problem. | It is used for solving classification problems. |
| In this we predict the value of continuous variables | In this we predict values of categorical variables |
| In this we find best fit line. | In this we find S-Curve. |
| Least square estimation method is used for estimation of accuracy. | Maximum likelihood estimation method is used for Estimation of accuracy. |
| The output must be continuous value, such as price, age, etc. | Output must be categorical value such as 0 or 1, Yes or no, etc. |
| It required linear relationship between dependent and independent variables. | It not required linear relationship. |
| There may be collinearity between the independent variables. | There should be little to no collinearity between independent variables. |

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

[Understanding Activation Functions in Depth](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

103

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

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


Like103

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/understanding-logistic-regression/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1882053201.1745055855&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=1575113481)

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