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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/machine-learning-model-evaluation/?type%3Darticle%26id%3D932252&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Steps to Build a Machine Learning Model\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/steps-to-build-a-machine-learning-model/)

# Machine Learning Model Evaluation

Last Updated : 12 Feb, 2025

Comments

Improve

Suggest changes

13 Likes

Like

Report

Model evaluation is a process that uses some metrics which help us to analyze the performance of the model. Think of training a model like teaching a student. **Model evaluation** is like giving them a test to see if they _truly_ learned the subject—or just memorized answers. It helps us answer:

- **Did the model learn patterns?**
- **Will it fail on new questions?**

Model development is a multi-step process and we need to keep a check on how well the model do future predictions and analyze a models weaknesses. There are many metrics for that. Cross Validation is one technique that is followed during the training phase and it is a model evaluation technique.

## **Cross-Validation: The Ultimate Practice Test**

[Cross Validation](https://www.geeksforgeeks.org/cross-validation-machine-learning/) is a method in which we do not use the whole dataset for training. In this technique some part of the dataset is reserved for testing the model. There are many types of Cross-Validation out of which [K Fold Cross Validation](https://www.geeksforgeeks.org/how-k-fold-prevents-overfitting-in-a-model/) is mostly used. In **K Fold Cross Validation the original dataset is divided into k subsets. The subsets are known as folds**. This is repeated k times where 1 fold is used for testing purposes, rest k-1 folds are used for training the model. It is seen that this technique generalizes the model well and reduces the error rate.

[Holdout](https://www.geeksforgeeks.org/introduction-of-holdout-method/) is the simplest approach. It is used in neural networks as well as in many classifiers. In this technique the dataset is divided into train and test datasets. The dataset is usually divided into ratios like 70:30 or 80:20. Normally a large percentage of data is used for training the model and a small portion of dataset is used for testing the model.

## Evaluation Metrics for Classification Task

Classification is used to categorize data into predefined labels or classes. To evaluate the performance of a classification model we commonly use metrics such as accuracy, precision, recall, F1 score and confusion matrix. These metrics are useful in assessing how well model distinguishes between classes especially in cases of imbalanced datasets. By understanding the strengths and weaknesses of each metric, we can select the most appropriate one for a given classification problem.

In this Python code, we have imported the iris dataset which has features like the length and width of sepals and petals. The target values are Iris setosa, Iris virginica, and Iris versicolor. After importing the dataset we divided the dataset into train and test datasets in the ratio 80:20. Then we called [Decision Trees](https://www.geeksforgeeks.org/decision-tree/) and trained our model. After that, we performed the prediction and calculated the accuracy score, precision, recall, and f1 score. We also plotted the confusion matrix.

#### Importing Libraries and Dataset

Python`
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,\
recall_score, f1_score, accuracy_score
`

Now let’s load the toy dataset iris flowers from the [sklearn.datasets library](https://www.geeksforgeeks.org/python-create-test-datasets-using-sklearn/) and then split it into training and testing parts (for model evaluation) in the 80:20 ratio.

Python`
iris = load_iris()
X = iris.data
y = iris.target
# Holdout method.Dividing the data into train and test
X_train, X_test,\
    y_train, y_test = train_test_split(X, y,
                                       random_state=20,
                                       test_size=0.20)
`

Now, let’s train a Decision Tree Classifier model on the training data, and then we will move on to the evaluation part of the model using different metrics.

Python`
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
`

### **1\. Accuracy**

[Accuracy](https://www.geeksforgeeks.org/metrics-for-machine-learning-model/) is defined as the ratio of number of correct predictions to the total number of predictions. This is the most fundamental metric used to evaluate the model. The formula is given by:

Accuracy=TP+TNTP+TN+FP+FN\\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}}Accuracy=TP+TN+FP+FNTP+TN​

However Accuracy has a drawback. It cannot perform well on an [imbalanced dataset](https://www.geeksforgeeks.org/what-is-imbalanced-dataset/). Suppose a model classifies that the majority of the data belongs to the major class label. It gives higher accuracy, but in general model cannot classify on minor class labels and has poor performance.

Python`
print("Accuracy:", accuracy_score(y_test,
                                  y_pred))
`

**Output:**

```
Accuracy: 0.9333333333333333
```

### **2\. Precision and Recall**

[Precision](https://www.geeksforgeeks.org/precision-recall-curve-ml/) is the ratio of true positives to the summation of true positives and false positives. It basically analyses the positive predictions.

Precision=TPTP+FP\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}Precision=TP+FPTP​

The drawback of Precision is that it does not consider the True  Negatives and False Negatives.

Recall is the ratio of true positives to the summation of true positives and false negatives. It basically analyses the number of correct positive samples.

Recall=TPTP+FN\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}Recall=TP+FNTP​

The drawback of Recall is that often it leads to a higher false positive rate.

Python`
print("Precision:", precision_score(y_test,
                                    y_pred,
                                    average="weighted"))
print('Recall:', recall_score(y_test,
                              y_pred,
                              average="weighted"))
`

**Output:**

```
Precision: 0.9435897435897436
Recall: 0.9333333333333333
```

### **3\. F1 score**

[F1 score](https://www.geeksforgeeks.org/f1-score-in-machine-learning/) is the harmonic mean of precision and recall. It is seen that during the precision-recall trade-off if we increase the precision, recall decreases and vice versa. The goal of the F1 score is to combine precision and recall.

F1 Score=2×Precision×RecallPrecision+Recall\\text{F1 Score} = \\frac{2 \\times \\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}F1 Score=Precision+Recall2×Precision×Recall​

Python`
# calculating f1 score
print('F1 score:', f1_score(y_test, y_pred,
                            average="weighted"))
`

**Output:**

```
F1 score: 0.9327777777777778
```

### **4\. Confusion Matrix**

[Confusion matrix](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/) is a N x N matrix where N is the number of target classes. It represents number of actual outputs and predicted outputs. Some terminologies in the matrix are as follows:

- **True Positives**: It is also known as TP. It is the output in which the actual and the predicted values are YES.
- **True Negatives:**  It is also known as TN. It is the output in which the actual and the predicted values are NO.
- **False Positives:** It is also known as FP. It is the output in which the actual value is NO but the predicted value is YES.
- **False Negatives:**  It is also known as FN. It is the output in which the actual value is YES but the predicted value is NO.

Python`
confusion_matrix = metrics.confusion_matrix(y_test,
                                            y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=[0, 1, 2])
cm_display.plot()
plt.show()
`

**Output:**

![Confusion matrix for the output of the model](https://media.geeksforgeeks.org/wp-content/uploads/20221228120316/download-(3).png)

Confusion matrix for the output of the model

In the output the accuracy of model is 93.33%. Precision is approximately 0.944  and Recall is 0.933. F1 score is approximately 0.933. Finally the confusion matrix is plotted. Here class labels denote the target classes:

```
0 = Setosa
1 = Versicolor
2 = Virginica
```

From the confusion matrix, we see that 8 setosa classes were correctly predicted. 11 Versicolor test cases were also correctly predicted by the model and 2 virginica test cases were misclassified. In contrast, the rest 9 were correctly predicted.

### **5\. AUC-ROC Curve**

[AUC (Area Under Curve)](https://www.geeksforgeeks.org/auc-roc-curve/) is an evaluation metric that is used to analyze the classification model at different threshold values. The [Receiver Operating Characteristic (ROC)](https://www.geeksforgeeks.org/receiver-operating-characteristic-roc-with-cross-validation-in-scikit-learn/) curve is a probabilistic curve used to highlight the model’s performance. The curve has two parameters:

- **TPR:** It stands for True positive rate. It basically follows the formula of Recall.
- **FPR:** It stands for False Positive rate. It is defined as the ratio of False positives to the summation of false positives and True negatives.

This curve is useful as it helps us to determine the model’s capacity to distinguish between different classes. Let us illustrate this with the help of a simple Python example

Python`
import numpy as np
from sklearn .metrics import roc_auc_score
y_true = [1, 0, 0, 1]
y_pred = [1, 0, 0.9, 0.2]
auc = np.round(roc_auc_score(y_true,
                             y_pred), 3)
print("Auc", (auc))
`

**Output:**

```
Auc 0.75
```

AUC score is a useful metric to evaluate the model. It highlights model’s capacity to separate the classes. In the above code 0.75 is a good AUC score. A model is considered good if the AUC score is greater than 0.5 and approaches 1.

## **Evaluation Metrics for Regression Task**

Regression is used to determine continuous values. It is mostly used to find a relation between a dependent and independent variable. For classification we use a confusion matrix, accuracy, f1 score, etc. But for regression analysis since we are predicting a numerical value it may differ from the actual output.  So we consider the error calculation as it helps to summarize how close the prediction is to the actual value. There are many metrics available for evaluating the regression model.

In this Python Code we have implemented a simple regression model using the Mumbai weather CSV file. This file comprises Day, Hour, Temperature, Relative Humidity, Wind Speed and Wind Direction. The link for the dataset is [here](https://media.geeksforgeeks.org/wp-content/uploads/20250121113221908989/weather.csv).

We are interested in finding relationship between Temperature and Relative Humidity. Here Relative Humidity is the dependent variable and Temperature is the independent variable. We performed [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) and use different metrics to evaluate the performance of our model. To calculate the metrics we make extensive use of sklearn library.

Python`
# importing the libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,\
    mean_squared_error, mean_absolute_percentage_error
`

Now let’s load the data into the panda’s data frame and then split it into training and testing parts (for model evaluation) in the 80:20 ratio.

Python`
df = pd.read_csv('weather.csv')
X = df.iloc[:, 2].values
Y = df.iloc[:, 3].values
X_train, X_test,\
    Y_train, Y_test = train_test_split(X, Y,
                                       test_size=0.20,
                                       random_state=0)
`

Now, let’s train a simple linear regression model. On the training data and we will move to the evaluation part of the model using different metrics.

Python`
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
regression = LinearRegression()
regression.fit(X_train, Y_train)
Y_pred = regression.predict(X_test)
`

### **1\. Mean Absolute Error (MAE)**

This is the simplest metric used to analyze the loss over the whole dataset. As we know that error is basically the difference between the predicted and actual values. Therefore [MAE](https://www.geeksforgeeks.org/how-to-calculate-mean-absolute-error-in-python/) is defined as the average of the errors calculated. Here we calculate the modulus of the error, perform summation and then divide the result by the total number of data points.  It is a positive value. The formula of MAE is given by

MAE=∑i=1N∣ypred–yactual∣N\\text{MAE} = \\frac{\\sum\_{i=1}^{N} \|\\text{y}\_{\\text{pred}} – \\text{y}\_{\\text{actual}}\|}{N}MAE=N∑i=1N​∣ypred​–yactual​∣​

Python`
mae = mean_absolute_error(y_true=Y_test,
                          y_pred=Y_pred)
print("Mean Absolute Error", mae)
`

**Output:**

```
Mean Absolute Error 1.7236295632503873
```

### **2\. Mean Squared Error(MSE)**

The most commonly used metric is Mean Square error or [MSE](https://www.geeksforgeeks.org/python-mean-squared-error/). It is a function used to calculate the loss. We find the difference between the predicted values and actual variable, square the result and then find the average by all datapoints present in dataset. MSE is always positive as we square the values. Small the value of MSE better is the performance of our model. The formula of MSE is given:

MSE=∑i=1N(ypred–yactual)2N\\text{MSE} = \\frac{\\sum\_{i=1}^{N} (\\text{y}\_{\\text{pred}} – \\text{y}\_{\\text{actual}})^2}{N}MSE=N∑i=1N​(ypred​–yactual​)2​

Python`
mse = mean_squared_error(y_true=Y_test,
                         y_pred=Y_pred)
print("Mean Square Error", mse)
`

**Output:**

```
Mean Square Error 3.9808057060106954
```

### **3\. Root Mean Squared Error(RMSE)**

[RMSE](https://www.geeksforgeeks.org/ml-mathematical-explanation-of-rmse-and-r-squared-error/) is a popular method and is the extended version of MSE. It indicates how much the data points are spread around the best line. It is the standard deviation of the MSE. A lower value means that the data point lies closer to the best fit line.

RMSE=∑i=1N(ypred–yactual)2N\\text{RMSE} = \\sqrt{\\frac{\\sum\_{i=1}^{N} (\\text{y}\_{\\text{pred}} – \\text{y}\_{\\text{actual}})^2}{N}}RMSE=N∑i=1N​(ypred​–yactual​)2​​

Python`
rmse = mean_squared_error(y_true=Y_test,
                          y_pred=Y_pred,
                          squared=False)
print("Root Mean Square Error", rmse)
`

**Output:**

```
Root Mean Square Error 1.9951956560725306
```

### **4\. Mean Absolute Percentage Error (MAPE)**

[MAPE](https://www.geeksforgeeks.org/how-to-calculate-mape-in-python/) is used to express the error in terms of percentage. It is defined as the difference between the actual and predicted value. The error is then divided by the actual value. The results are then summed up and finally and we calculate the average. Smaller the percentage better the performance of the model. The formula is given by

MAPE=1N∑i=1N(∣ypred–yactual∣∣yactual∣)×100%\\text{MAPE} = \\frac{1}{N} \\sum\_{i=1}^{N} \\left( \\frac{\|\\text{y}\_{\\text{pred}} – \\text{y}\_{\\text{actual}}\|}{\|\\text{y}\_{\\text{actual}}\|} \\right) \\times 100 \\%MAPE=N1​∑i=1N​(∣yactual​∣∣ypred​–yactual​∣​)×100%

Python`
mape = mean_absolute_percentage_error(Y_test,
                                      Y_pred,
                                      sample_weight=None,
                                      multioutput='uniform_average')
print("Mean Absolute Percentage Error", mape)
`

**Output:**

```
Mean Absolute Percentage Error 0.02334408993333347
```

Evaluating machine learning models is a important step in ensuring their effectiveness and reliability in real-world applications. Using appropriate metrics such as accuracy, precision, recall, F1 score for classification and regression-specific measures like MAE, MSE, RMSE and MAPE can assess model performance for different tasks. Moreover adopting evaluation techniques like cross-validation and holdout ensures that models generalize well to unseen data.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/steps-to-build-a-machine-learning-model/)

[Steps to Build a Machine Learning Model](https://www.geeksforgeeks.org/steps-to-build-a-machine-learning-model/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

13

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Technical Scripter](https://www.geeksforgeeks.org/category/technical-scripter/)
- [Technical Scripter 2022](https://www.geeksforgeeks.org/tag/technical-scripter-2022/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Machine Learning Tutorial\\
\\
\\
Machine learning is a subset of Artificial Intelligence (AI) that enables computers to learn from data and make predictions without being explicitly programmed. If you're new to this field, this tutorial will provide a comprehensive understanding of machine learning, its types, algorithms, tools, an\\
\\
8 min read](https://www.geeksforgeeks.org/machine-learning/)

## Prerequisites for Machine Learning

[Python for Machine Learning\\
\\
\\
Welcome to "Python for Machine Learning," a comprehensive guide to mastering one of the most powerful tools in the data science toolkit. Python is widely recognized for its simplicity, versatility, and extensive ecosystem of libraries, making it the go-to programming language for machine learning. I\\
\\
6 min read](https://www.geeksforgeeks.org/python-for-machine-learning/)
[SQL for Machine Learning\\
\\
\\
Integrating SQL with machine learning can provide a powerful framework for managing and analyzing data, especially in scenarios where large datasets are involved. By combining the structured querying capabilities of SQL with the analytical and predictive capabilities of machine learning algorithms,\\
\\
6 min read](https://www.geeksforgeeks.org/sql-for-machine-learning/)

## Getting Started with Machine Learning

[Advantages and Disadvantages of Machine Learning\\
\\
\\
Machine learning (ML) has revolutionized industries, reshaped decision-making processes, and transformed how we interact with technology. As a subset of artificial intelligence ML enables systems to learn from data, identify patterns, and make decisions with minimal human intervention. While its pot\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-machine-learning/)
[Why ML is Important ?\\
\\
\\
Machine learning (ML) has become a cornerstone of modern technology, revolutionizing industries and reshaping the way we interact with the world. As a subset of artificial intelligence (AI), ML enables systems to learn and improve from experience without being explicitly programmed. Its importance s\\
\\
4 min read](https://www.geeksforgeeks.org/why-ml-is-important/)
[Real- Life Examples of Machine Learning\\
\\
\\
Machine learning plays an important role in real life, as it provides us with countless possibilities and solutions to problems. It is used in various fields, such as health care, financial services, regulation, and more. Importance of Machine Learning in Real-Life ScenariosThe importance of machine\\
\\
13 min read](https://www.geeksforgeeks.org/real-life-applications-of-machine-learning/)
[What is the Role of Machine Learning in Data Science\\
\\
\\
In today's world, the collaboration between machine learning and data science plays an important role in maximizing the potential of large datasets. Despite the complexity, these concepts are integral in unraveling insights from vast data pools. Let's delve into the role of machine learning in data\\
\\
9 min read](https://www.geeksforgeeks.org/role-of-machine-learning-in-data-science/)
[Top Machine Learning Careers/Jobs\\
\\
\\
Machine Learning (ML) is one of the fastest-growing fields in technology, driving innovations across healthcare, finance, e-commerce, and more. As companies increasingly adopt AI-based solutions, the demand for skilled ML professionals is Soaring. This article delves into the Type of Machine Learnin\\
\\
10 min read](https://www.geeksforgeeks.org/top-career-paths-in-machine-learning/)

Like13

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/machine-learning-model-evaluation/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=750755846.1745055243&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1017802237)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055243114&cv=11&fst=1745055243114&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fmachine-learning-model-evaluation%2F%3Fref%3Dlbp&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Machine%20Learning%20Model%20Evaluation%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=2106880144.1745055243&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=ikx9hqswbzt4)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=normal&cb=alm2bqmosa48)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOlTnqb9r_mc_r5R&size=invisible&cb=8wu7iy4j74cj)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)