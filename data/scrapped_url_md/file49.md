- [DSA](https://www.geeksforgeeks.org/learn-data-structures-and-algorithms-dsa-tutorial/)
- [Practice Problems](https://www.geeksforgeeks.org/explore)
- [Python](https://www.geeksforgeeks.org/python-programming-language/)
- [C](https://www.geeksforgeeks.org/c-programming-language/)
- [C++](https://www.geeksforgeeks.org/c-plus-plus/)
- [Java](https://www.geeksforgeeks.org/java/)
- [Courses](https://www.geeksforgeeks.org/courses)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [DevOps](https://www.geeksforgeeks.org/devops-tutorial/)
- [Web Development](https://www.geeksforgeeks.org/web-development/)
- [System Design](https://www.geeksforgeeks.org/system-design-tutorial/)
- [Aptitude](https://www.geeksforgeeks.org/aptitude-questions-and-answers/)
- [Projects](https://www.geeksforgeeks.org/computer-science-projects/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/feature-selection-using-decision-tree/?type%3Darticle%26id%3D1181910&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Bagging and Random Forest for Imbalanced Classification\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/bagging-and-random-forest-for-imbalanced-classification/)

# Feature selection using Decision Tree

Last Updated : 11 Mar, 2024

Comments

Improve

Suggest changes

Like Article

Like

Report

**Feature selection using decision trees** involves identifying the most important features in a dataset based on their contribution to the decision tree's performance. The article aims to explore **feature selection using decision trees and how decision trees evaluate feature importance.**

## **What is feature selection?**

[Feature selection](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) involves choosing a subset of important features for building a model. It aims to enhance model performance by reducing overfitting, improving interpretability, and cutting computational complexity.

### **Need for Feature Selection**

Datasets can have hundreds, thousands, or sometimes millions of features in the case of image- or text-based models. If we build an ML model using all the given features, it will lead to model overfitting and ultimately a low-performance rate.

Feature selection helps in:

- Increasing model performance.
- Increasing model interpretability.
- Reducing model complexity.
- Enhancing data visualization.

## What are decision trees ?

[Decision trees](https://www.geeksforgeeks.org/decision-tree/) are a popular machine learning algorithm used for both classification and regression tasks. They model decisions based on the features of the data and their outcomes.

## How do decision trees play a role in feature selection?

- Decision trees select the 'best' feature for splitting at each node based on [information gain.](https://www.geeksforgeeks.org/decision-tree-introduction-example/)
- Information gain measures the reduction in [entropy](https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/)(disorder) in a set of data points.
- Features with higher information gain are considered more important for splitting, thus aiding in feature selection.
- By recursively selecting features for splitting, decision trees inherently prioritize the most relevant features for the model.

## Implementation: Feature Selection using Decision Tree

In this implementation, we are going to discuss a practical approach to feature selection using decision trees, allowing for more efficient and interpretable models by focusing on the most relevant features. You can download the dataset from [here](https://media.geeksforgeeks.org/wp-content/uploads/20240304171136/apple_quality.csv).

### Step 1: Importing Libraries

We need to import the below libraries for implementing decision trees.

Python`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
`

### Step 2: Dataset Description

Getting data descriptions by df.info().

Python`
df = pd.read_csv('apple_quality.csv')
df.info()
`

**Output:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4001 entries, 0 to 4000
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   A_id         4000 non-null   float64
 1   Size         4000 non-null   float64
 2   Weight       4000 non-null   float64
 3   Sweetness    4000 non-null   float64
 4   Crunchiness  4000 non-null   float64
 5   Juiciness    4000 non-null   float64
 6   Ripeness     4000 non-null   float64
 7   Acidity      4001 non-null   object
 8   Quality      4000 non-null   object
dtypes: float64(7), object(2)
memory usage: 281.4+ KB

```

### Step 3: Data Preproccessing

Python3`
df = df.dropna()
df.isnull().sum()
`

**Output:**

```
A_id           0
Size           0
Weight         0
Sweetness      0
Crunchiness    0
Juiciness      0
Ripeness       0
Acidity        0
Quality        0
dtype: int64

```

### Step 4: Splitting the data

Splitting the dataset into train and test sets.

Python`
X = df.drop(['Quality'], axis = 1)
y = df['Quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8, test_size = 0.3)
`

### Step 5: Scaling the data

Python`
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
`

### Step 6: Training the Decision Tree Classifier

- The DecisionTreeClassifier is trained with a maximum depth of 16 and a random state of 8, which helps control the randomness for reproducibility.

Python3`
# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=16, random_state=8)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
`

### Step 7: Feature selection

- The feature importances are calculated using the trained classifier, indicating the relative importance of each feature in the model's decision-making process
- A threshold of 0.1 is set to select features with importance greater than this value, potentially reducing the number of features considered for the final model.
- The selected\_features variable contains the names of the features that meet the importance threshold, which can be used for further analysis or model refinement.

And, then we use only the selected columns.

Python3`
# Get feature importances
importances = clf.feature_importances_
# Select features with importance greater than a threshold
threshold = 0.1  # Adjust as needed
selected_features = X.columns[importances &gt; threshold]
# Use only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
`

### Step 8: Train a model using the selected features

Python3`
# Train a new model using the selected features
clf_selected = DecisionTreeClassifier(max_depth=16, random_state=8)
clf_selected.fit(X_train_selected, y_train)
`

### Step 9: Comparing the accuracies

Python3`
# Make predictions on the test set using the model trained with all features
y_pred_all_features = clf.predict(X_test_scaled)
# Calculate the accuracy of the model with all features
accuracy_all_features = accuracy_score(y_test, y_pred_all_features)
print(f&quot;Accuracy with all features: {accuracy_all_features}&quot;)
# Make predictions on the test set using the model trained with selected features
y_pred_selected_features = clf_selected.predict(X_test_selected)
# Calculate the accuracy of the model with selected features
accuracy_selected_features = accuracy_score(y_test, y_pred_selected_features)
print(f&quot;Accuracy with selected features: {accuracy_selected_features}&quot;)
`

**Output:**

```
Accuracy with all features: 0.7983333333333333
Accuracy with selected features: 0.8241666666666667

```

These accuracy scores provide insights into the performance of the models. The accuracy score represents the proportion of correctly classified instances out of the total instances in the test set.

**Comparing the two accuracies:**

- The model trained with all features achieved an accuracy of approximately 79.83%.
- However, after feature selection, the model trained with selected features achieved a higher accuracy of approximately 82.42%.

## Conclusion

Feature selection using decision trees offers a powerful and intuitive approach to enhancing model performance and interpretability. Following the outlined steps, we can easily select features using decision trees to build more robust and efficient models for various applications.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/bagging-and-random-forest-for-imbalanced-classification/)

[Bagging and Random Forest for Imbalanced Classification](https://www.geeksforgeeks.org/bagging-and-random-forest-for-imbalanced-classification/)

[A](https://www.geeksforgeeks.org/user/areeshaanjum748/)

[areeshaanjum748](https://www.geeksforgeeks.org/user/areeshaanjum748/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Data Prediction using Decision Tree of rpart\\
\\
\\
Decision trees are a popular choice due to their simplicity and interpretation, and effectiveness at handling both numerical and categorical data. The rpart (Recursive Partitioning) package in R specializes in constructing these trees, offering a robust framework for building predictive models. Over\\
\\
3 min read](https://www.geeksforgeeks.org/data-prediction-using-decision-tree-of-rpart/)
[Python \| Decision Tree Regression using sklearn\\
\\
\\
When it comes to predicting continuous values, Decision Tree Regression is a powerful and intuitive machine learning technique. Unlike traditional linear regression, which assumes a straight-line relationship between input features and the target variable, Decision Tree Regression is a non-linear re\\
\\
4 min read](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/)
[Sequential Feature Selection\\
\\
\\
Feature selection is a process of identifying and selecting the most relevant features from a dataset for a particular predictive modeling task. This can be done for a variety of reasons, such as to improve the predictive accuracy of a model, to reduce the computational complexity of a model, or to\\
\\
3 min read](https://www.geeksforgeeks.org/sequential-feature-selection/)
[Feature Selection Using Random forest Classifier\\
\\
\\
Feature selection is a crucial step in the machine learning pipeline that involves identifying the most relevant features for building a predictive model. One effective method for feature selection is using a Random Forest classifier, which provides insights into feature importance. In this article,\\
\\
5 min read](https://www.geeksforgeeks.org/feature-selection-using-random-forest-classifier/)
[Feature Descriptor in Image Processing\\
\\
\\
In image processing, a feature descriptor is a representation of an image region or key point that captures relevant information about the image content. In this article, we are going to discuss one of the image processing algorithms i.e. Feature Descriptor Image processingImage processing is a comp\\
\\
5 min read](https://www.geeksforgeeks.org/feature-descriptor-in-image-processing/)
[ML \| Extra Tree Classifier for Feature Selection\\
\\
\\
Prerequisites: Decision Tree Classifier Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a "forest" to output it's classification result. In concept, it is very si\\
\\
6 min read](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/)
[Feature Selection using Branch and Bound Algorithm\\
\\
\\
Feature selection is a very important factor in Machine Learning. To get the algorithms to work properly and give near about perfect predictions, i.e to enhance the performance of a predictive model, feature selection is required. Large set of features or redundant ones should be removed. Up to a ce\\
\\
4 min read](https://www.geeksforgeeks.org/feature-selection-using-branch-and-bound-algorithm/)
[Top Decision Tree Interview Questions\\
\\
\\
Decision Tree is one of the most popular and intuitive machine learning algorithms used for both classification and regression tasks where it creates flowchart-like structure where each internal node represents a decision or a test on a feature, each branch represents the outcome of that test, and e\\
\\
11 min read](https://www.geeksforgeeks.org/top-decision-tree-interview-questions/)
[Feature Selection Using Random Forest\\
\\
\\
Feature selection is a crucial step in building machine learning models. It involves selecting the most important features from your dataset that contribute to the predictive power of the model. Random Forest, an ensemble learning method, is widely used for feature selection due to its inherent abil\\
\\
4 min read](https://www.geeksforgeeks.org/feature-selection-using-random-forest/)
[Pruning decision trees\\
\\
\\
Decision tree pruning is a critical technique in machine learning used to optimize decision tree models by reducing overfitting and improving generalization to new data. In this guide, we'll explore the importance of decision tree pruning, its types, implementation, and its significance in machine l\\
\\
6 min read](https://www.geeksforgeeks.org/pruning-decision-trees/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/feature-selection-using-decision-tree/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=275812867.1745055866&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=316724138)

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