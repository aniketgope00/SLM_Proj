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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/?type%3Darticle%26id%3D1280780&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Least Mean Squares Filter in Signal Processing\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/least-mean-squares-filter-in-signal-processing/)

# Building and Implementing Decision Tree Classifiers with Scikit-Learn: A Comprehensive Guide

Last Updated : 27 Jan, 2025

Comments

Improve

Suggest changes

3 Likes

Like

Report

Decision Tree Classifiers is a fundamental machine learning algorithm for classification tasks. They organize data into a tree-like structure where internal nodes represent decisions, branches represent outcomes and leaf node represent class labels. This article introduces how to build and implement these classifiers using [Scikit-Learn](https://www.geeksforgeeks.org/learning-model-building-scikit-learn-python-machine-learning-library/) a Python library for machine learning.

## **Implementing Decision Tree Classifiers with Scikit-Learn**

The **DecisionTreeClassifier** from **Sklearn** has the ability to perform multi-class classification on a dataset. The syntax for DecisionTreeClassifier is as follows:

> _class_ sklearn.tree.DecisionTreeClassifier **(** _\*_ **,** _criterion='gini'_ **,** _splitter='best'_ **,** _max\_depth=None_ **,** _min\_samples\_split=2_ **,** _min\_samples\_leaf=1_ **,** _min\_weight\_fraction\_leaf=0.0_ **,** _max\_features=None_ **,** _random\_state=None_ **,** _max\_leaf\_nodes=None_ **,** _min\_impurity\_decrease=0.0_ **,** _class\_weight=None_ **,** _ccp\_alpha=0.0_ **,** _monotonic\_cst=None)_

**Let's go through the parameters:**

- criterion: It meeasure the quality of a split. Supported values are 'gini', 'entropy' and 'log\_loss'. The default value is 'gini'
- splitter: This parameter is used to choose the split at each node. Supported values are 'best' & 'random'. The default value is 'best'
- max\_features: It defines the number of features to consider when looking for the best split.
- max\_depth: This parameter denotes maximum depth of the tree (default=None).
- min\_samples\_split: It defines the minimum number of samples reqd. to split an internal node (default=2).
- min\_samples\_leaf: The minimum number of samples required to be at a leaf node (default=1)
- max\_leaf\_nodes: It defines the maximum number of possible leaf nodes.
- min\_impurity\_split: It defines the threshold for early stopping tree growth.
- class\_weight: It defines the weights associated with classes.
- ccp\_alpha: It is a complexity parameter used for minimal cost-complexity pruning

## Steps to train a DecisionTreeClassifier Using Sklearn

Let's look at how to train a DecisionTreeClassifier using Sklearn on **Iris dataset**. The phase by phase execution as follows:

**Step 1: Import Libraries**

To start, import the libraries you'll need such as Scikit-Learn (sklearn) for machine learning tasks.

Python`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
`

**Step 2: Data Loading**

In order to perform classification load a dataset. For demonstration one can utilize sample datasets from Scikit-Learn such as Iris or Breast Cancer.

Python`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
`

**Step 3: Splitting Data**

Use the train\_test\_split method from sklearn.model\_selection to split the dataset into training and testing sets.

Python`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
`

**Step 4: Starting the Model**

Using DecisionTreeClassifier from sklearn.tree create an object for the Decision Tree Classifier.

Python`
clf = DecisionTreeClassifier(random_state=1)
`

**Step 5: Training the Model**

Apply the fit method to match the classifier to the training set of data.

Python`
clf.fit(X_train, y_train)
`

**Step 6: Making Predictions**

Apply the predict method to the test data and use the trained model to create predictions.

Python`
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
`

Let's implement the complete code based on above steps. The code is as follows:

Python`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# split dataset to training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 99)
# initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=1)
# train the classifier
clf.fit(X_train, y_train)
# predict using classifier
y_pred = clf.predict(X_test)
# claculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
`

**Output:**

> Accuracy: 0.9555555555555556

## Hyperparameter Tuning with Decision Tree Classifier

Hyperparameters are configuration settings that control the behavior of a decision tree model and significantly affect its performance. Proper tuning can improve accuracy, reduce overfitting and enhance generalization of model.

Popular methods for tuning include Grid Search, Random Search, and Bayesian Optimization, which explore different combinations to find the best configuration.

> For more details, you can check out the article [How to tune a Decision Tree in Hyperparameter tuning?](https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/)

### Hyperparmater tuning using GridSearchCV

Let's make use of Scikit-Learn's GridSearchCV to find the best combination of of hyperparameter values. The code is as follows:

Python`
from sklearn.model_selection import GridSearchCV
# Hyperparameter to fine tune
param_grid = {
    'max_depth': range(1, 10, 1),
    'min_samples_leaf': range(1, 20, 2),
    'min_samples_split': range(2, 20, 2),
    'criterion': ["entropy", "gini"]
}
tree = DecisionTreeClassifier(random_state=1)
# GridSearchCV
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid,
                           cv=5, verbose=True)
grid_search.fit(X_train, y_train)
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)
`

**Output:**

> Fitting 5 folds for each of 1620 candidates, totalling 8100 fits
>
> best accuracy 0.9714285714285715
>
> DecisionTreeClassifier(criterion='entropy', max\_depth=4, min\_samples\_leaf=3, random\_state=1)

Here we defined the parameter grid with a set of hyperparameters and a list of possible values. The GridSearchCV evaluates the different hyperparameter combinations for the DecissionTree Classifier and selects the best combination of hyperparameters based on the performance across all k folds.

## Visualizing the Decision Tree Classifier

Decision Tree visualization is used to interpret and comprehend model's choices. We'll plot feature importance obtained from the Decision Tree model to see which features have the greatest predictive power. Here we fetch the best estimator obtained from the gridsearchcv as the decision tree classifier.

Python`
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

tree_clf = grid_search.best_estimator_
plt.figure(figsize=(18, 15))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names)
plt.show()
`

**Output:**

![iris_decision_tree](https://media.geeksforgeeks.org/wp-content/uploads/20240702085105/iris_decision_tree.jpg)DecisionTree Visualization

We can see that it start from the root node (depth 0, at the top).

- The root node checks whether the flower petal width is less than or equal to 0.75. If it is then we move to the root's left child node (depth1, left). Here the left node doesn't have any child nodes so the classifier will predict the class for that node as setosa.
- If the petal width is greater than 0.75 then we must move down to the root's right child node (depth 1, right). Here the right node is not a leaf node, so node check for the condition until it reaches the leaf node.

## Decision Tree Classifier With Spam Email Detection Dataset

Spam email detection dataset is trained on decision trees to predict e-mails as spam or safe to open(ham). As scikit-learn library is used for this implementation.

Let's load the spam email dataset and plot the count of spam and ham emails using matplotlib. The code is as follows:

Python`
import pandas as pd
import matplotlib.pyplot as plt
dataset_link = 'https://media.geeksforgeeks.org/wp-content/uploads/20240620175612/spam_email.csv'
df = pd.read_csv(dataset_link)
df['Category'].value_counts().plot.bar(color = ["g","r"])
plt.title('Total number of ham and spam in the dataset')
plt.show()
`

**Output:**

![spam_email_plot](https://media.geeksforgeeks.org/wp-content/uploads/20240620191635/spam_email_plot.png)Spam and Ham Email Count

As a next step, let's prepare the data for decision tree classifier. The code is as follows:

Python`
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def clean_str(string):
    reg = RegexpTokenizer(r'[a-z]+')
    string = string.lower()
    tokens = reg.tokenize(string)
    return " ".join(tokens)
df['Category'] = df['Category'].map({'ham' : 0,'spam' : 1 })
df['text_clean'] = df['Message'].apply(
lambda string: clean_str(string))
cv = CountVectorizer()
X = cv.fit_transform(df.text_clean)
y = df.Category
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)
`

As part of data preparation the category string is replaced with a numeric attribute, RegexpTokenizer is used for message cleaning and CountVectorizer() is used to convert text documents to a matrix of tokens. Finally, the dataset is separated into training and test sets.

Now we can use the prepared data to train a DecisionTreeClassifier. The code is as follows:

Python`
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(classification_report(y_test, pred))
`

**Output:**

> precision recall f1-score support
>
> 0 0.98 0.98 0.98 966
>
> 1 0.89 0.89 0.89 149
>
> accuracy 0.97 1115
>
> macro avg 0.94 0.93 0.94 1115
>
> weighted avg 0.97 0.97 0.97 1115

Here we used DecisionTreeClassifier from sklearn to train our model, and the classicfication\_metrics() is used for evaluating the predictions. Let's check the confusion matrix for the decision tree classifier.

Python`
import seaborn as sns
cmat = confusion_matrix(y_test, pred)
sns.heatmap(cmat, annot=True, cmap='Paired',
            cbar=False, fmt="d", xticklabels=[\
            'Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
`

Output:

![heatmap_spam_email](https://media.geeksforgeeks.org/wp-content/uploads/20240620193402/heatmap_spam_email.png)Heatmap

## Advantages and Disadvantages of Decision Tree Classifier

### Advantages of Decision Tree Classifier

- **Interpretability**: Decision trees are accessible for comprehending the decisions made by the model because they are simple to grasp and display.
- **Takes Care of Non-linearity:** They don't need feature scaling in order to capture non-linear correlations between features and target variables.
- **Handles Mixed Data Types**: Without the necessity for one-hot encoding, decision trees are able to handle both numerical and categorical data.
- **Feature Selection**: By choosing significant features further up the tree, they obliquely carry out feature selection.
- **No Assumptions about Data Distribution**: Unlike parametric models, decision trees do not make any assumptions on the distribution of data.
- **Robust to Outliers**: Because of the way dividing nodes works they are resistant to data outliers.

### Drawbacks of Decision Tree Classifier

- **Overfitting**: If training data isn't restricted or pruned, decision trees have a tendency to overfit and capture noise and anomalies.
- **Instability:** Minor changes in the data might result in entirely different tree structures, which is what makes them unstable.
- **Difficulty in Capturing Complex interactions:** Deeper trees may be necessary in order to capture complex interactions such as XOR.
- **Bias** towards Dominant Classes: Decision Trees may exhibit bias towards dominant classes in datasets with an uneven distribution of classes.

Decision Tree Classifiers are a powerful and interpretable tool in machine learning and we can implement them using Scikit learn python library. By using hyperparameter tuning methods like GridSearchCV we can optimize their performance.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/least-mean-squares-filter-in-signal-processing/)

[Least Mean Squares Filter in Signal Processing](https://www.geeksforgeeks.org/least-mean-squares-filter-in-signal-processing/)

[![author](https://media.geeksforgeeks.org/auth/profile/gzpybsujg7wgz5njwnle)](https://www.geeksforgeeks.org/user/batraharshita12/)

[batraharshita12](https://www.geeksforgeeks.org/user/batraharshita12/)

Follow

3

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Building a Custom Estimator for Scikit-learn: A Comprehensive Guide\\
\\
\\
Scikit-learn is a powerful machine learning library in Python that offers a wide range of tools for data analysis and modeling. One of its best features is the ease with which you can create custom estimators, allowing you to meet specific needs. In this article, we will walk through the process of\\
\\
5 min read](https://www.geeksforgeeks.org/building-a-custom-estimator-for-scikit-learn-a-comprehensive-guide/)
[Comprehensive Guide to Classification Models in Scikit-Learn\\
\\
\\
Scikit-Learn, a powerful and user-friendly machine learning library in Python, has become a staple for data scientists and machine learning practitioners. It offers a wide array of tools for data mining and data analysis, making it accessible and reusable in various contexts. This article delves int\\
\\
12 min read](https://www.geeksforgeeks.org/comprehensive-guide-to-classification-models-in-scikit-learn/)
[Determining Feature Importance in SVM Classifiers with Scikit-Learn\\
\\
\\
Support Vector Machines (SVM) are a powerful tool in the machine learning toolbox, renowned for their ability to handle high-dimensional data and perform both linear and non-linear classification tasks. However, one of the challenges with SVMs is interpreting the model, particularly when it comes to\\
\\
5 min read](https://www.geeksforgeeks.org/determining-feature-importance-in-svm-classifiers-with-scikit-learn/)
[Classifier Comparison in Scikit Learn\\
\\
\\
In scikit-learn, a classifier is an estimator that is used to predict the label or class of an input sample. There are many different types of classifiers that can be used in scikit-learn, each with its own strengths and weaknesses. Let's load the iris datasets from the sklearn.datasets and then tra\\
\\
3 min read](https://www.geeksforgeeks.org/classifier-comparison-in-scikit-learn/)
[How to implement cost-sensitive learning in decision trees?\\
\\
\\
Decision trees are tools for classification, but they can struggle with imbalanced datasets where one class significantly outnumbers the other. Cost-sensitive learning is a technique that addresses this issue by assigning different costs to misclassification errors, making the decision tree more sen\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-implement-cost-sensitive-learning-in-decision-trees/)
[Machine Learning Packages and IDEs: A Comprehensive Guide\\
\\
\\
Machine learning (ML) has revolutionized various industries by enabling systems to learn from data and make intelligent decisions. To harness the power of machine learning, developers and data scientists rely on a plethora of packages and Integrated Development Environments (IDEs). This article delv\\
\\
12 min read](https://www.geeksforgeeks.org/machine-learning-packages-and-ides-a-comprehensive-guide/)
[Decision Tree Classifiers in R Programming\\
\\
\\
Classification is the task in which objects of several categories are categorized into their respective classes using the properties of classes. A classification model is typically used to, Predict the class label for a new unlabeled data objectProvide a descriptive model explaining what features ch\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-classifiers-in-r-programming/)
[How to Identify the Most Informative Features for scikit-learn Classifiers\\
\\
\\
Feature selection is an important step in the machine learning pipeline. By identifying the most informative features, you can enhance model performance, reduce overfitting, and improve computational efficiency. In this article, we will demonstrate how to use scikit-learn to determine feature import\\
\\
7 min read](https://www.geeksforgeeks.org/how-to-identify-the-most-informative-features-for-scikit-learn-classifiers/)
[Comparing Support Vector Machines and Decision Trees for Text Classification\\
\\
\\
Support Vector Machines (SVMs) and Decision Trees are both popular algorithms for text classification, but they have different characteristics and are suitable for different types of problems. Why is model selection important in Text Classification?Selecting the ideal model for text classification r\\
\\
8 min read](https://www.geeksforgeeks.org/comparing-support-vector-machines-and-decision-trees-for-text-classification/)
[Building Naive Bayesian classifier with WEKA\\
\\
\\
The use of the Naive Bayesian classifier in Weka is demonstrated in this article. The â€œweather-nominalâ€ data set used in this experiment is available in ARFF format. This paper assumes that the data has been properly preprocessed. The Bayes' Theorem is used to build a set of classification algorithm\\
\\
3 min read](https://www.geeksforgeeks.org/building-naive-bayesian-classifier-with-weka/)

Like3

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=807695563.1745055872&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1789833883)

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