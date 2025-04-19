- [Python for Machine Learning](https://www.geeksforgeeks.org/python-for-machine-learning/)
- [Machine Learning with R](https://www.geeksforgeeks.org/introduction-to-machine-learning-in-r/)
- [Machine Learning Algorithms](https://www.geeksforgeeks.org/machine-learning-algorithms/)
- [EDA](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/)
- [Math for Machine Learning](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-projects/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Computer vision](https://www.geeksforgeeks.org/computer-vision/)
- [Data Science](https://www.geeksforgeeks.org/data-science-for-beginners/)
- [Artificial Intelligence](https://www.geeksforgeeks.org/artificial-intelligence/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/a-comprehensive-guide-to-ensemble-learning/?type%3Darticle%26id%3D1102718&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What are Descriptive Analytics? Working and Examples\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/what-are-descriptive-analytics/)

# Ensemble Learning

Last Updated : 23 Jan, 2025

Comments

Improve

Suggest changes

2 Likes

Like

Report

**Ensemble learning combines the predictions of multiple models (called "weak learners" or "base models") to make a stronger, more reliable prediction. The goal is to reduce errors and improve performance.**

> It is like asking a group of experts for their opinions instead of relying on just one person. Each expert might make mistakes, but when you combine their knowledge, the final decision is often better and more accurate.

## Types of Ensemble Learning in Machine Learning

There are two main types of ensemble methods:

1. **Bagging (Bootstrap Aggregating):** Models are trained independently on different subsets of the data, and their results are averaged or voted on.
2. **Boosting:** Models are trained sequentially, with each one learning from the mistakes of the previous model.

Think of it like asking multiple doctors for a diagnosis (bagging) or consulting doctors who specialize in correcting previous misdiagnoses (boosting).

## 1\. Bagging Algorithm

[Bagging classifier](https://www.geeksforgeeks.org/ml-bagging-classifier/) can be used for both regression and classification tasks. Here is an overview of **Bagging classifier algorithm:**

- **Bootstrap Sampling:** Divides the original training data into ‘N’ subsets and randomly selects a subset with replacement in some rows from other subsets. This step ensures that the base models are trained on diverse subsets of the data and there is no class imbalance.
- **Base Model Training**: For each bootstrapped sample we train a base model independently on that subset of data. These weak models are trained in parallel to increase computational efficiency and reduce time consumption. **We can use different base learners i.e different ML models as base learners to bring variety and robustness.**
- **Prediction Aggregation:** To make a prediction on testing data combine the predictions of all base models. For classification tasks it can include majority voting or weighted majority while for regression it involves averaging the predictions.
- **Out-of-Bag (OOB) Evaluation**: Some samples are excluded from the training subset of particular base models during the bootstrapping method. These “out-of-bag” samples can be used to estimate the model’s performance without the need for cross-validation.
- **Final Prediction:** After aggregating the predictions from all the base models, Bagging produces a final prediction for each instance.

**Python pseudo code for Bagging Estimator implementing libraries:**

1\. **Importing Libraries and Loading Data**

Python`
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
`

2\. **Loading and Splitting the Iris Dataset**

Python`
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
`

3\. **Creating a Base Classifier**

Python`
base_classifier = DecisionTreeClassifier()
`

Decision tree is chosen as the base model. They are prone to overfitting when trained on small datasets making them good candidates for bagging.

4\. **Creating and Training the Bagging Classifier**

Python`
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)
`

- A **`BaggingClassifier`** is created using the decision tree as the base classifier.
- **`n_estimators=10`** specifies that 10 decision trees will be trained on different bootstrapped subsets of the training data.

5\. **Making Predictions and Evaluating Accuracy**

Python`
y_pred = bagging_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
`

- The trained bagging model predicts labels for test data.
- The accuracy of the predictions is calculated by comparing the predicted labels ( `y_pred`) to the actual labels ( `y_test`).

**Output:**

```
Accuracy: 1.0

```

## **2\. Boosting Algorithm**

[Boosting](https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/) is an ensemble technique that combines multiple weak learners to create a strong learner. Weak models are trained in series such that each next model tries to correct errors of the previous model until the entire training dataset is predicted correctly. One of the most well-known boosting algorithms is [AdaBoost (Adaptive Boosting).](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/) Here is an overview of Boosting algorithm:

- **Initialize Model Weights**: Begin with a single weak learner and assign equal weights to all training examples.
- **Train Weak Learner**: Train weak learners on these dataset.
- **Sequential Learning**: Boosting works by training models sequentially where each model focuses on correcting the errors of its predecessor. Boosting typically uses a single type of weak learner like decision trees.
- **Weight Adjustment**: Boosting assigns weights to training datapoints. Misclassified examples receive higher weights in the next iteration so that next models pay more attention to them.

**Python pseudo code for boosting Estimator implementing libraries:**

1\. **Importing Libraries and Modules**

Python`
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
`

**`AdaBoostClassifier`** **:** Implements the AdaBoost algorithm.

2\. **Loading and Splitting the Dataset**

Python`
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
`

3\. **Defining the Weak Learner**

Python`
base_classifier = DecisionTreeClassifier(max_depth=1)
`

Decision tree with **`max_depth`** `=1` is used as the weak learner.

4\. **Creating and Training the AdaBoost Classifier**

Python`
adaboost_classifier = AdaBoostClassifier(
    base_classifier, n_estimators=50, learning_rate=1.0, random_state=42
)
adaboost_classifier.fit(X_train, y_train)
`

- **`base_classifier`** **:** The weak learner used in boosting.
- **`n_estimators=50`**: Number of weak learners to train sequentially.
- **`learning_rate=1.0`**: Controls the contribution of each weak learner to the final model.
- **`random_state=42`** **:** Ensures reproducibility.

**5**. **Making Predictions and Calculating Accuracy**

Python`
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
`

**Output:**

```
Accuracy: 1.0
```

## Benefits of Ensemble Learning in Machine Learning

Ensemble learning is a versatile approach that can be applied to machine learning model for:-

- **Reduction in Overfitting**: By aggregating predictions of multiple models ensembles can reduce overfitting that individual complex models might exhibit.
- **Improved Generalization**: It generalize better to unseen data by minimizing variance and bias.
- **Increased Accuracy**: Combining multiple models gives higher predictive accuracy.
- **Robustness to Noise**: It mitigates the effect of noisy or incorrect data points by averaging out predictions from diverse models.
- **Flexibility**: It can work with diverse models including decision trees, neural networks and support vector machines making them highly adaptable.
- **Bias-Variance Tradeoff**: Techniques like bagging reduce variance, while boosting reduces bias leading to better overall performance.

There are various ensemble learning techniques we can use as each one of them has their own pros and cons.

## Ensemble Learning Techniques

| Technique | Category | Description |
| --- | --- | --- |
| Random Forest | Bagging | [Random forest](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/) constructs multiple decision trees on bootstrapped subsets of the data and aggregates their predictions for final output, reducing overfitting and variance. |
| Random Subspace Method | Bagging | Trains models on random subsets of input features to enhance diversity and improve generalization while reducing overfitting. |
| Gradient Boosting Machines (GBM) | Boosting | [Gradient Boosting Machines](https://www.geeksforgeeks.org/ml-gradient-boosting/) sequentially builds decision trees, with each tree correcting errors of the previous ones, enhancing predictive accuracy iteratively. |
| Extreme Gradient Boosting (XGBoost) | Boosting | [XGBoost](https://www.geeksforgeeks.org/xgboost/) do optimizations like tree pruning, regularization, and parallel processing for robust and efficient predictive models. |
| AdaBoost (Adaptive Boosting) | Boosting | [AdaBoost](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/) focuses on challenging examples by assigning weights to data points. Combines weak classifiers with weighted voting for final predictions. |
| CatBoost | Boosting | [CatBoost](https://www.geeksforgeeks.org/catboost-ml/) specialize in handling categorical features natively without extensive preprocessing with high predictive accuracy and automatic overfitting handling. |

Selecting the right ensemble technique depends on the nature of the data, specific problem we are trying to solve and computational resources available. It often requires experimentation and changes to achieve the best results.

In conclusion ensemble learning is an method that uses the strengths and diversity of multiple models to enhance prediction accuracy in various machine learning applications. This technique is widely applicable in areas such as classification, regression, time series forecasting and other domains where reliable and precise predictions are crucial. It also used to mitigate overfitting issue.

Suggested Quiz


10 Questions


In ensemble learning, what is the primary purpose of combining multiple models?

- To reduce the computational complexity

- To increase the interpretability of the model

- To improve predictive accuracy and robustness

- To ensure uniformity in predictions


Explanation:

Which of the following ensemble techniques is specifically designed to improve the performance of weak learners by sequentially correcting errors?

- Bagging

- Stacking

- Boosting

- Random Forest


Explanation:

Boosting improves weak learner by training them sequentially , focusing on correcting the errors made by previous model to create a stronger overall model

What is the primary mechanism through which Bagging reduces overfitting in machine learning models?

- By increasing model complexity

- By combining predictions from different algorithms

- By training on random subsets of data

- By using a single model for predictions


Explanation:

In the context of ensemble learning, what does the term "meta-learner" refer to?

- A model that is trained on raw data

- A model that combines predictions from base models

- A model that is used for feature extraction

- A model that performs data preprocessing


Explanation:

The meta-learner takes the outputs of base models as inputs and learns how to combine them to improve the overall prediction accuracy.

Which ensemble technique is characterized by its ability to handle categorical features without extensive preprocessing?

- Gradient Boosting

- Random Forest

- CatBoost

- AdaBoost


Explanation:

CatBoost is characterized by its ability to handle categorical features without extensive preprocessing

In ensemble methods, what is the role of "Out-of-Bag" (OOB) evaluation?

- To estimate model performance without cross-validation

- To increase the training dataset size

- To validate the model on unseen data

- To combine predictions from multiple models


Explanation:

In ensemble methods, the role of "Out-of-Bag" (OOB) evaluation is to assess the performance and accuracy of models, particularly in random forests

Which of the following statements best describes the "bias-variance trade-off" in ensemble learning?

- It is a method to enhance model interpretability.

- It balances errors from model complexity and data noise sensitivity.

- It ensures uniform predictions across different models.

- It involves selecting a single best model for predictions.


Explanation:

ensemble learning balances errors from model complexity and data noise sensitivity.

What is a common application of ensemble methods in the field of finance?

- Time series analysis of stock prices

- Predicting customer preferences

- Portfolio optimization

- Image recognition tasks


Explanation:

In the context of ensemble learning, what does "stacking" involve?

- Combining predictions through majority voting

- Training multiple models independently

- Using a single model to predict outcomes

- Training a new model to combine the predictions of base models


Explanation:

The new model learns how to best mix the predictions from multiple models to get a more accurate final result.

Which ensemble technique is known for its ability to reduce model variance by averaging predictions?

- Boosting

- Bagging

- Stacking

- Blending


Explanation:

Because bagging, or bootstrap aggregating, reduces model variance by averaging the predictions from multiple models trained on different random samples of the data.

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

[Next Article](https://www.geeksforgeeks.org/what-are-descriptive-analytics/)

[What are Descriptive Analytics? Working and Examples](https://www.geeksforgeeks.org/what-are-descriptive-analytics/)

[V](https://www.geeksforgeeks.org/user/viswajeetray/)

[viswajeetray](https://www.geeksforgeeks.org/user/viswajeetray/)

Follow

2

Improve

Article Tags :

- [Data Science](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/tag/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Types of Ensemble Learning\\
\\
\\
Ensemble Learning in machine learning that integrates multiple models called as weak learners to create a single effective model for prediction. This technique is used to enhance accuracy, minimizing variance and removing overfitting. Here we will learn different ensemble techniques and their algori\\
\\
5 min read](https://www.geeksforgeeks.org/types-of-ensemble-learning/?ref=ml_lbp)
[Extreme Learning Machine\\
\\
\\
Extreme Learning Machine commonly referred to as ELM, is one of the machine learning algorithms introduced by Huang et al in 2006. This algorithm has gained widespread recognition in recent years, primarily due to its lightning-fast learning capabilities, exceptional generalization performance, and\\
\\
12 min read](https://www.geeksforgeeks.org/extreme-learning-machine/?ref=ml_lbp)
[Easy Ensemble Classifier in Machine Learning\\
\\
\\
The Easy Ensemble Classifier (EEC) is an advanced ensemble learning algorithm specifically designed to address class imbalance issues in classification tasks. It enhances the performance of models on imbalanced datasets by leveraging oversampling and ensembling techniques to improve classification a\\
\\
6 min read](https://www.geeksforgeeks.org/easy-ensemble-classifier-in-machine-learning/?ref=ml_lbp)
[Ensemble Learning with SVM and Decision Trees\\
\\
\\
Ensemble learning is a machine learning technique that combines multiple individual models to improve predictive performance. Two popular algorithms used in ensemble learning are Support Vector Machines (SVMs) and Decision Trees. What is Ensemble Learning?By merging many models (also referred to as\\
\\
5 min read](https://www.geeksforgeeks.org/ensemble-learning-with-svm-and-decision-trees/?ref=ml_lbp)
[Ensemble Classifier \| Data Mining\\
\\
\\
Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model. Basic idea is to learn a set of classifiers (experts) and to allow them to vote. Advantage : Improvement in predictiv\\
\\
3 min read](https://www.geeksforgeeks.org/ensemble-classifier-data-mining/?ref=ml_lbp)
[Multiagent Planning in AI\\
\\
\\
In the vast landscape of Artificial Intelligence (AI), multiagent planning emerges as a pivotal domain that orchestrates the synergy among multiple autonomous agents to achieve collective goals. It encompasses a spectrum of strategies and methodologies aimed at coordinating the decision-making proce\\
\\
12 min read](https://www.geeksforgeeks.org/multiagent-planning-in-ai/?ref=ml_lbp)
[Ensemble Methods in Python\\
\\
\\
Ensemble means a group of elements viewed as a whole rather than individually. An Ensemble method creates multiple models and combines them to solve it. Ensemble methods help to improve the robustness/generalizability of the model. In this article, we will discuss some methods with their implementat\\
\\
11 min read](https://www.geeksforgeeks.org/ensemble-methods-in-python/?ref=ml_lbp)
[ML \| Voting Classifier using Sklearn\\
\\
\\
A Voting Classifier is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output. It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output cl\\
\\
3 min read](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/?ref=ml_lbp)
[Mean Shift Clustering using Sklearn\\
\\
\\
Clustering is a fundamental method in unsupervised device learning, and one powerful set of rules for this venture is Mean Shift clustering. Mean Shift is a technique for grouping comparable data factors into clusters primarily based on their inherent characteristics, with our previous understanding\\
\\
9 min read](https://www.geeksforgeeks.org/mean-shift-clustering-using-sklearn/?ref=ml_lbp)
[What is Generative Machine Learning?\\
\\
\\
Generative Machine Learning is an interesting subset of artificial intelligence, where models are trained to generate new data samples similar to the original training data. In this article, we'll explore the fundamentals of generative machine learning, compare it with discriminative models, delve i\\
\\
4 min read](https://www.geeksforgeeks.org/what-is-generative-machine-learning/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/a-comprehensive-guide-to-ensemble-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2085627308.1745056427&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102015665~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=914816407)