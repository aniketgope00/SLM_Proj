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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-gradient-boosting/?type%3Darticle%26id%3D459222&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Advantages and Disadvantages of Logistic Regression\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/)

# Gradient Boosting in ML

Last Updated : 11 Mar, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Gradient Boosting is a ensemble learning method used for classification and regression tasks. It is a [**boosting**](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/) algorithm which combine multiple weak learner to create a strong predictive model. It works by sequentially training models where each new model tries to correct the errors made by its predecessor.

In gradient boosting each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using [gradient descent](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/). In each iteration the algorithm computes the gradient of the loss function with respect to the predictions and then trains a new weak model to minimize this gradient. The predictions of the new model are then added to the ensemble and the process is repeated until a stopping criterion is met.

### **Shrinkage and Model Complexity**

A key feature of Gradient Boosting is shrinkage which scales the contribution of each new model by a factor called the **learning rate** (denoted as η).

- **Smaller learning rates:** mean the contribution of each tree is smaller which reduces the risk of overfitting but requires more trees to achieve the same performance.
- **Larger learning rates:** mean each tree has a more significant impact but this can lead to overfitting.

There’s a trade-off between the learning rate and the number of estimators (trees), a smaller learning rate usually means more trees are required to achieve optimal performance.

## Working of Gradient Boosting

### **1\. Sequential Learning Process**

The ensemble consists of multiple trees each trained to correct the errors of the previous one. In the first iteration **Tree 1** is trained on the original data xxx and the true labels yyy. It makes predictions which are used to compute the **residuals** (the difference between the actual and predicted values).

### **2\. Residuals Calculation**

In the second iteration **Tree 2** is trained using the feature matrix xxx and the **residuals** from **Tree 1** as labels. This means Tree 2 is trained to predict the errors of Tree 1. This process continues for all the trees in the ensemble. Each subsequent tree is trained to predict the residual errors of the previous tree.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200721214745/gradientboosting.PNG)

Gradient Boosted Trees

### **3\. Shrinkage**

After each tree is trained its predictions are **shrunk** by multiplying them with the learning rate η (which ranges from 0 to 1). This prevents overfitting by ensuring each tree has a smaller impact on the final model.

Once all trees are trained predictions are made by summing the contributions of all the trees. The final prediction is given by the formula:

ypred=y1+η⋅r1+η⋅r2+⋯+η⋅rNy\_{\\text{pred}} = y\_1 + \\eta \\cdot r\_1 + \\eta \\cdot r\_2 + \\cdots + \\eta \\cdot r\_Nypred​=y1​+η⋅r1​+η⋅r2​+⋯+η⋅rN​

Where r1,r2,…,rNr\_1, r\_2, \\dots, r\_Nr1​,r2​,…,rN​are the residuals (errors) predicted by each tree.

## Difference between Adaboost and Gradient Boosting

The difference between [AdaBoost](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/) and gradient boosting are as follows:

| AdaBoost | Gradient Boosting |
| --- | --- |
| During each iteration in AdaBoost, the weights of incorrectly classified samples are increased so that the next weak learner focuses more on these samples. | Gradient Boosting updates the weights by computing the negative gradient of the loss function with respect to the predicted output. |
| AdaBoost uses simple decision trees with one split known as the decision stumps of weak learners. | Gradient Boosting can use a wide range of base learners such as decision trees and linear models. |
| AdaBoost is more susceptible to noise and outliers in the data as it assigns high weights to misclassified samples | Gradient Boosting is generally more robust as it updates the weights based on the gradients which are less sensitive to outliers. |

## **Implementing Gradient Boosting for Classification and Regression**

Here are two examples to demonstrate how **Gradient Boosting** works for both classification and regression. But before that let’s understand gradient boosting parameters.

- **n\_estimators:** This specifies the number of trees (estimators) to be built. A higher value typically improves model performance but increases computation time.
- **learning\_rate:** This is the **shrinkage parameter**. It scales the contribution of each tree.
- **random\_state:** It ensures **reproducibility** of results. Setting a fixed value for `random_state` ensures that you get the same results every time you run the model.
- **max\_features**: This parameter limits the number of features each tree can use for splitting. It helps prevent overfitting by limiting the complexity of each tree and promoting diversity in the model.

Now we start building our models with Gradient Boosting.

### **Example 1: Classification**

We use **Gradient Boosting Classifier** to predict digits from the popular **Digits dataset**.

Steps:

- Import the necessary libraries
- Setting SEED for reproducibility
- Load the digit dataset and split it into train and test.
- Instantiate Gradient Boosting classifier and fit the model.
- Predict the test set and compute the accuracy score.

Python`
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
SEED = 23
X, y = load_digits(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = SEED)
gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )

gbc.fit(train_X, train_y)
pred_y = gbc.predict(test_X)
acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))
`

**Output:**

> Gradient Boosting Classifier accuracy is : 0.98

### Example 2: Regression

We use **Gradient Boosting Regressor** on the **Diabetes dataset** to predict continuous values:

Steps:

- Import the necessary libraries
- Setting SEED for reproducibility
- Load the diabetes dataset and split it into train and test.
- Instantiate Gradient Boosting Regressor and fit the model.
- Predict on the test set and compute RMSE.

python`
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
SEED = 23
X, y = load_diabetes(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = SEED)
gbr = GradientBoostingRegressor(loss='absolute_error',
                                learning_rate=0.1,
                                n_estimators=300,
                                max_depth = 1,
                                random_state = SEED,
                                max_features = 5)
gbr.fit(train_X, train_y)
pred_y = gbr.predict(test_X)
test_rmse = mean_squared_error(test_y, pred_y) ** (1 / 2)
print('Root mean Square error: {:.2f}'.format(test_rmse))
`

**Output:**

> Root mean Square error: 56.39

Gradient Boosting is an effective and widely-used machine learning technique for both classification and regression problems. It builds models sequentially focusing on correcting errors made by previous models which leads to improved performance. While it can be computationally expensive, tuning parameters like the **learning rate** and **number of estimators** can help optimize the model and prevent overfitting.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/)

[Advantages and Disadvantages of Logistic Regression](https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/)

[N](https://www.geeksforgeeks.org/user/nikki2398/)

[nikki2398](https://www.geeksforgeeks.org/user/nikki2398/)

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

[Gradient Boosting in R\\
\\
\\
In this article, we will explore how to implement Gradient Boosting in R, its theory, and practical examples using various R packages, primarily gbm and xgboost. Gradient Boosting in RGradient Boosting is a powerful machine-learning technique for regression and classification problems. It builds mod\\
\\
6 min read](https://www.geeksforgeeks.org/gradient-boosting-in-r/?ref=ml_lbp)
[ML \| XGBoost (eXtreme Gradient Boosting)\\
\\
\\
In machine learning we often combine different algorithms to get better and optimize results. Our main goal is to minimize loss function for which, one of the famous algorithm is XGBoost (Extreme boosting) technique which works by building an ensemble of decision trees sequentially where each new tr\\
\\
6 min read](https://www.geeksforgeeks.org/ml-xgboost-extreme-gradient-boosting/?ref=ml_lbp)
[GrowNet: Gradient Boosting Neural Networks\\
\\
\\
GrowNet was proposed in 2020 by students from Purdue, UCLA, and Virginia Tech in collaboration with engineers from Amazon and LinkedIn California. They proposed a new gradient boosting algorithm where they used a shallow neural network as the weak learners, a general loss function for training the g\\
\\
6 min read](https://www.geeksforgeeks.org/grownet-gradient-boosting-neural-networks/?ref=ml_lbp)
[What is Gradient descent?\\
\\
\\
Gradient Descent is a fundamental algorithm in machine learning and optimization. It is used for tasks like training neural networks, fitting regression lines, and minimizing cost functions in models. In this article we will understand what gradient descent is, how it works , mathematics behind it a\\
\\
8 min read](https://www.geeksforgeeks.org/what-is-gradient-descent/?ref=ml_lbp)
[Custom gradients in TensorFlow\\
\\
\\
Custom gradients in TensorFlow allow you to define your gradient functions for operations, providing flexibility in how gradients are computed for complex or non-standard operations. This can be useful for tasks such as implementing custom loss functions, incorporating domain-specific knowledge into\\
\\
6 min read](https://www.geeksforgeeks.org/custom-gradients-in-tensorflow/?ref=ml_lbp)
[Gradient Descent Algorithm in R\\
\\
\\
Gradient Descent is a fundamental optimization algorithm used in machine learning and statistics. It is designed to minimize a function by iteratively moving toward the direction of the steepest descent, as defined by the negative of the gradient. The goal is to find the set of parameters that resul\\
\\
7 min read](https://www.geeksforgeeks.org/gradient-descent-algorithm-in-r/?ref=ml_lbp)
[Understanding Gradient Clipping\\
\\
\\
Gradient Clipping is the process that helps maintain numerical stability by preventing the gradients from growing too large. When training a neural network, the loss gradients are computed through backpropagation. However, if these gradients become too large, the updates to the model weights can als\\
\\
12 min read](https://www.geeksforgeeks.org/understanding-gradient-clipping/?ref=ml_lbp)
[ML \| Mini-Batch Gradient Descent with Python\\
\\
\\
In machine learning, gradient descent is an optimization technique used for computing the model parameters (coefficients and bias) for algorithms like linear regression, logistic regression, neural networks, etc. In this technique, we repeatedly iterate through the training set and update the model\\
\\
5 min read](https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/?ref=ml_lbp)
[Stochastic Gradient Descent In R\\
\\
\\
Gradient Descent is an iterative optimization process that searches for an objective functionâ€™s optimum value (Minimum/Maximum). It is one of the most used methods for changing a modelâ€™s parameters to reduce a cost function in machine learning projects. In this article, we will learn the concept of\\
\\
10 min read](https://www.geeksforgeeks.org/stochastic-gradient-descent-in-r/?ref=ml_lbp)
[Applying Gradient Clipping in TensorFlow\\
\\
\\
In deep learning, gradient clipping is an essential technique to prevent gradients from becoming too large during backpropagation, which can lead to unstable training and exploding gradients. This article provides a detailed overview of how to apply gradient clipping in TensorFlow, starting from the\\
\\
5 min read](https://www.geeksforgeeks.org/applying-gradient-clipping-in-tensorflow/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-gradient-boosting/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=939071777.1745056434&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=470613963)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)