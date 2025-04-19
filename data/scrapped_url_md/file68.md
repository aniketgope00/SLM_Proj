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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/?type%3Darticle%26id%3D1167365&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Top Machine Learning Projects for Healthcare\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/machine-learning-projects-for-healthcare/)

# Random Forest Algorithm in Machine Learning

Last Updated : 16 Jan, 2025

Comments

Improve

Suggest changes

9 Likes

Like

Report

A Random Forest is a collection of decision trees that work together to make predictions. In this article, we'll explain how the Random Forest algorithm works and how to use it.

## Understanding Intuition for Random Forest Algorithm

Random Forest algorithm is a powerful tree learning technique in Machine Learning to make predictions and **then we do voting of all the tress to make prediction**. They are widely used for classification and regression task.

- It is a type of classifier that uses many decision trees to make predictions.
- It takes different random parts of the dataset to train each tree and then it combines the results by averaging them. This approach helps improve the accuracy of predictions. Random Forest is based on [ensemble learning](https://www.geeksforgeeks.org/a-comprehensive-guide-to-ensemble-learning/).

Imagine asking a group of friends for advice on where to go for vacation. Each friend gives their recommendation based on their unique perspective and preferences (decision trees trained on different subsets of data). You then make your final decision by considering the majority opinion or averaging their suggestions (ensemble prediction).

![Lightbox](https://media.geeksforgeeks.org/wp-content/uploads/20240701170624/Random-Forest-Algorithm.webp)

**As explained in image:** Process starts with a dataset with rows and their corresponding class labels (columns).

- Then - Multiple Decision Trees are created from the training data. Each tree is trained on a random subset of the data (with replacement) and a random subset of features. This process is known as **bagging** or **bootstrap aggregating**.
- Each Decision Tree in the ensemble learns to make predictions independently.
- When presented with a new, unseen instance, each Decision Tree in the ensemble makes a prediction.

The final prediction is made by combining the predictions of all the Decision Trees. This is typically done through a majority vote (for classification) or averaging (for regression).

## **Key Features of Random Forest**

- **Handles Missing Data**: Automatically handles missing values during training, eliminating the need for manual imputation.
- Algorithm ranks **features based on their importance in making predictions** offering valuable insights for feature selection and interpretability.
- **Scales Well with Large and Complex Data** without significant performance degradation.
- Algorithm is versatile and can be applied to both classification tasks (e.g., predicting categories) and regression tasks (e.g., predicting continuous values).

## How Random Forest Algorithm Works?

The random Forest algorithm works in several steps:

- Random Forest builds **multiple decision trees using random samples of the data**. **Each tree is trained on a different subset of the data which makes each tree unique**.
- When creating each tree the **algorithm randomly selects a subset of features or variables to split the data rather than using all available features at a time. This adds diversity to the trees.**
- Each decision tree in the forest **makes a prediction based on the data it was trained on. When making final prediction random forest combines the results from all the trees.**
  - For classification tasks the final prediction is decided by a majority vote. This means that the category predicted by most trees is the final prediction.
  - For regression tasks the final prediction is the average of the predictions from all the trees.
- The **randomness in data samples and feature selection helps to prevent the model from overfitting making the predictions more accurate and reliable.**

## Assumptions of Random Forest

- **Each tree makes its own decisions**: Every tree in the forest makes its own predictions without relying on others.
- **Random parts of the data are used**: Each tree is built using random samples and features to reduce mistakes.
- **Enough data is needed**: Sufficient data ensures the trees are different and learn unique patterns and variety.
- **Different predictions improve accuracy**: Combining the predictions from different trees leads to a more accurate final results.

Now, as we've understood the concept behind the algorithm we'll try implementing:

## Implementing Random Forest for Classification Tasks

Python`
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
# Corrected URL for the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(url)
# Drop rows with missing 'Survived' values
titanic_data = titanic_data.dropna(subset=['Survived'])
# Features and target variable
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']
# Encode 'Sex' column
X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})
# Fill missing 'Age' values with the median
X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred = rf_classifier.predict(X_test)
# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
# Sample prediction
sample = X_test.iloc[0:1]  # Keep as DataFrame to match model input format
prediction = rf_classifier.predict(sample)
# Retrieve and display the sample
sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
`

**Output:**

```
Accuracy: 0.80

Classification Report:
               precision    recall  f1-score   support

           0       0.82      0.85      0.83       105
           1       0.77      0.73      0.75        74

    accuracy                           0.80       179
   macro avg       0.79      0.79      0.79       179
weighted avg       0.80      0.80      0.80       179

Sample Passenger: {'Pclass': 3, 'Sex': 1, 'Age': 28.0, 'SibSp': 1, 'Parch': 1, 'Fare': 15.2458}
Predicted Survival: Did Not Survive
```

In the above code we use a Random Forest Classifier to analyze the Titanic dataset. The Random Forest Classifier learns from the training data and is tested on the test set and we evaluate the model's performance using a classification report to see how well it predicts the outcomes and used a random sample to check model prediction.

## Implementing Random Forest for Regression Tasks

Python`
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Load the California housing dataset
california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target
# Features and target variable
X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the regressor
rf_regressor.fit(X_train, y_train)
# Make predictions
y_pred = rf_regressor.predict(X_test)
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Sample Prediction
single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
# Print results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
`

**Output:**

```
Predicted Value: 0.51
Actual Value: 0.48
Mean Squared Error: 0.26
R-squared Score: 0.80
```

Random Forest learns from the training data like a real estate expert. After training it predicts house prices on the test set. We evaluate the model's performance using [Mean Squared Error and R-squared Score](https://www.geeksforgeeks.org/ml-r-squared-in-regression-analysis/) which show how accurate the predictions are and used a random sample to check model prediction.

## Advantages of Random Forest

- Random Forest provides very accurate predictions even with large datasets.
- Random Forest can handle missing data well without compromising with accuracy.
- It doesn’t require normalization or standardization on dataset.
- When we combine multiple decision trees it reduces the risk of overfitting of the model.

## Limitations of Random Forest

- It can be computationally expensive especially with a large number of trees.
- It’s harder to interpret the model compared to simpler models like decision trees.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/machine-learning-projects-for-healthcare/)

[Top Machine Learning Projects for Healthcare](https://www.geeksforgeeks.org/machine-learning-projects-for-healthcare/)

[![author](https://media.geeksforgeeks.org/auth/profile/cor53rinikmzcmzuvt63)](https://www.geeksforgeeks.org/user/susmit_sekhar_bhakta/)

[susmit\_sekhar\_bhakta](https://www.geeksforgeeks.org/user/susmit_sekhar_bhakta/)

Follow

9

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[First-Order algorithms in machine learning\\
\\
\\
First-order algorithms are a cornerstone of optimization in machine learning, particularly for training models and minimizing loss functions. These algorithms are essential for adjusting model parameters to improve performance and accuracy. This article delves into the technical aspects of first-ord\\
\\
7 min read](https://www.geeksforgeeks.org/first-order-algorithms-in-machine-learning/?ref=ml_lbp)
[How to Choose Right Machine Learning Algorithm?\\
\\
\\
Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed. ML is one of the most exciting technologies that one would have ever come across. A machine-learning algorithm is a program with a particular manner of altering its own parameters\\
\\
4 min read](https://www.geeksforgeeks.org/choosing-a-suitable-machine-learning-algorithm/?ref=ml_lbp)
[Tree Based Machine Learning Algorithms\\
\\
\\
Tree-based algorithms are a fundamental component of machine learning, offering intuitive decision-making processes akin to human reasoning. These algorithms construct decision trees, where each branch represents a decision based on features, ultimately leading to a prediction or classification. By\\
\\
14 min read](https://www.geeksforgeeks.org/tree-based-machine-learning-algorithms/?ref=ml_lbp)
[Machine Learning Algorithms Cheat Sheet\\
\\
\\
This guide is a go to guide for machine learning algorithms for your interview and test preparation. This guide briefly describes key points and applications for each algorithm. This article provides an overview of key algorithms in each category, their purposes, and best use-cases. Types of Machine\\
\\
6 min read](https://www.geeksforgeeks.org/machine-learning-algorithms-cheat-sheet/?ref=ml_lbp)
[Top 6 Machine Learning Classification Algorithms\\
\\
\\
Are you navigating the complex world of machine learning and looking for the most efficient algorithms for classification tasks? Look no further. Understanding the intricacies of Machine Learning Classification Algorithms is essential for professionals aiming to find effective solutions across diver\\
\\
13 min read](https://www.geeksforgeeks.org/top-6-machine-learning-algorithms-for-classification/?ref=ml_lbp)
[Rand-Index in Machine Learning\\
\\
\\
Cluster analysis, also known as clustering, is a method used in unsupervised learning to group similar objects or data points into clusters. It's a fundamental technique in data mining, machine learning, pattern recognition, and exploratory data analysis. To assess the quality of the clustering resu\\
\\
8 min read](https://www.geeksforgeeks.org/rand-index-in-machine-learning/?ref=ml_lbp)
[Multiclass vs Multioutput Algorithms in Machine Learning\\
\\
\\
This article will explore the realm of multiclass classification and multioutput regression algorithms in sklearn (scikit learn). We will delve into the fundamentals of classification and examine algorithms provided by sklearn, for these tasks, and gain insight, into effectively managing imbalanced\\
\\
6 min read](https://www.geeksforgeeks.org/multiclass-vs-multioutput-algorithms-in-machine-learning/?ref=ml_lbp)
[Implement Machine Learning With Caret In R\\
\\
\\
In today's society, technological answers to human issues are knocking on the doors of practically all fields of knowledge. Every aspect of this universe's daily operations generates data, and technology solutions base their decisions on these data-driven intuitions. In order to create a machine tha\\
\\
8 min read](https://www.geeksforgeeks.org/implement-machine-learning-with-caret-in-r/?ref=ml_lbp)
[Flowchart for basic Machine Learning models\\
\\
\\
Machine learning tasks have been divided into three categories, depending upon the feedback available: Supervised Learning: These are human builds models based on input and output.Unsupervised Learning: These are models that depend on human input. No labels are given to the learning algorithm, the m\\
\\
2 min read](https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/?ref=ml_lbp)
[Introduction to Machine Learning in R\\
\\
\\
The word Machine Learning was first coined by Arthur Samuel in 1959. The definition of machine learning can be defined as that machine learning gives computers the ability to learn without being explicitly programmed. Also in 1997, Tom Mitchell defined machine learning that â€œA computer program is sa\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-to-machine-learning-in-r/?ref=ml_lbp)

Like9

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=630497429.1745056393&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=663201796)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056393129&cv=11&fst=1745056393129&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Frandom-forest-algorithm-in-machine-learning%2F&hn=www.googleadservices.com&frm=0&tiba=Random%20Forest%20Algorithm%20in%20Machine%20Learning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=2099457933.1745056393&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)