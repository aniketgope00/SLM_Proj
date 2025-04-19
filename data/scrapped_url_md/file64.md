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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/?type%3Darticle%26id%3D1108249&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Gaussian Process Regression (GPR) with Noise-Level Estimation\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/gaussian-process-regression-gpr-with-noise-level-estimation/)

# Gaussian Naive Bayes using Sklearn

Last Updated : 17 Dec, 2023

Comments

Improve

Suggest changes

Like Article

Like

Report

In the world of machine learning, Gaussian Naive Bayes is a simple yet powerful algorithm used for classification tasks. It belongs to the Naive Bayes algorithm family, which uses Bayes' Theorem as its foundation. The goal of this post is to explain the Gaussian Naive Bayes classifier and offer a detailed implementation tutorial for [Python](https://www.geeksforgeeks.org/introduction-to-python/) users utilizing the Sklearn module.

A family of algorithms known as " [naive Bayes classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/)" use the Bayes Theorem with the strong (naive) presumption that every feature in the dataset is unrelated to every other feature. Naive Bayes classifiers perform very well in a variety of real-world situations despite this simplicity. The Naive Bayes classifier is a probabilistic algorithm based on Bayes' theorem. It assumes that features are conditionally independent, given the class label. Despite its 'naive' assumption, Naive Bayes often performs well in various real-world scenarios.

## Gaussian Naive Bayes

The probabilistic classification algorithm [Gaussian Naive Bayes](https://www.geeksforgeeks.org/gaussian-naive-bayes/) (GNB) is founded on the Bayes theorem. Given the class label, it is assumed that features follow a Gaussian distribution and are conditionally independent. For continuous data, GNB is especially helpful. The algorithm calculates the variance and mean of each feature for every class during training. During the prediction stage, it determines which class an instance is most likely to belong to by calculating the probability of each class. Text classification and spam filtering are just two of the many applications that can benefit from GNB's computational efficiency and ability to handle high-dimensional datasets.

### Bayes’ Theorem

The [Bayes Theorem](https://www.geeksforgeeks.org/bayes-theorem/) allows us to calculate the probability of an event based on the likelihood of a previous occurrence. The theorem is expressed mathematically as:

```
P(A∣B)=P(B∣A)⋅P(A)P(B)P(A∣B)=\frac{P(B∣A)⋅P(A)}{P(B)}P(A∣B)=P(B)P(B∣A)⋅P(A)​
```

Where:

- ( P(A\|B) ) is the probability of event A given that B is true.
- ( P(B\|A) ) is the probability of event B given that A is true.
- ( P(A) ) and ( P(B) ) are the probabilities of observing A and B independently of each other.

The Gaussian Naive Bayes classifier is one of several algorithms available in machine learning that may be used to tackle a wide range of issues. This article uses the well-known Scikit-Learn package (Sklearn) to walk readers who are new to data science and machine learning through the basic ideas of Gaussian Naive Bayes. We will go over the fundamental ideas, important vocabulary, and useful examples to help you grasp.

### Representation for Gaussian Naïve Bayes

Gaussian Naive Bayes (GNB) uses Gaussian (normal) distributions to represent the probability distribution of features within each class. Estimating the mean (μ) and variance (σ2 ) for every feature in every class is part of the representation for a dataset with m features and n classes.

Mathematically, the Gaussian distribution for a feature Xi​ in class ⁬ Cj​ is represented as follows:

P(Xi∣Ci)=12πσ2e−(x−μc)22σc2P(X\_{i}\|C\_{i}) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^\\frac{-(x-\\mu\_{c})^2}{2\\sigma\_c^2}

P(Xi​∣Ci​)=2πσ2​1​e2σc2​−(x−μc​)2​

Where,

- μc\\mu\_{c}

μc​ is the mean feature X in class c.
- σc2\\sigma^2\_c

σc2​ is the variance in class c.

### Implementation of Gaussian Naive Bayes using Synthetic Dataset

#### Generating a Synthetic Dataset

We’ll start by creating a synthetic dataset suitable for classification. The make\_classification function in Sklearn will be used to create a dataset with two features.

Python3`
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
# Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)
# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('Synthetic Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
`

**Output:**

![download-(3)-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20231205173711/download-(3).png)

- **The reason for our actions:** To have a controlled environment where we know the precise attributes of the data, we create a synthetic dataset. This facilitates comprehension of the algorithm's behavior.
- **How it works:** To generate a dataset, we may define the amount of samples, features, and other parameters using the make\_classification function.
- **Gained outcome:** The distribution of the synthetic dataset is displayed by a scatter plot, which uses various colors to represent the two classes of data.

#### Training the Gaussian Naive Bayes Model

Now, we’ll train the Gaussian Naive Bayes model using the synthetic dataset.

Python3`
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
# Train the model
gnb.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = gnb.predict(X_test)
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
`

**Output:**

```
Accuracy: 0.9666666666666667
```

- **The reason for our actions:** Training the model allows us to learn the parameters that best fit our data.
- **How it works:** We use the [train\_test\_split](https://www.geeksforgeeks.org/how-to-split-a-dataset-into-train-and-test-sets-using-python/) function to divide our data into training and testing sets. The GaussianNB class is used to initialize and train the model.
- **Gained outcome:** The accuracy score tells us how well our model performs on unseen data.

### Implementation of Gaussian Naive Bayes on Census Income Dataset

#### Importing Libraries

Python3`
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
`

The code performs Naive Bayes classification using scikit-learn and handles data using [pandas](https://www.geeksforgeeks.org/python-pandas-dataframe/). Labels are encoded, data is divided into training and testing sets, a Gaussian Naive Bayes classifier is trained, and the accuracy of the classifier is assessed.

#### Loading the Census Income Dataset

We’ll start by loading the Census Income dataset from the UCI Machine Learning Repository.

Python3`
# Load the Census Income dataset
url = &quot;https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data&quot;
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',\
                'marital-status', 'occupation','relationship', 'race',\
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week',\
                'native-country', 'income']
census_data = pd.read_csv(url, names=column_names)
# Display the first few rows of the dataset
print(census_data.head())
`

**Output:**

```
   age          workclass  fnlwgt   education  education-num  \
0   39          State-gov   77516   Bachelors             13
1   50   Self-emp-not-inc   83311   Bachelors             13
2   38            Private  215646     HS-grad              9
3   53            Private  234721        11th              7
4   28            Private  338409   Bachelors             13
        marital-status          occupation    relationship    race      sex  \
0        Never-married        Adm-clerical   Not-in-family   White     Male
1   Married-civ-spouse     Exec-managerial         Husband   White     Male
2             Divorced   Handlers-cleaners   Not-in-family   White     Male
3   Married-civ-spouse   Handlers-cleaners         Husband   Black     Male
4   Married-civ-spouse      Prof-specialty            Wife   Black   Female
   capital-gain  capital-loss  hours-per-week  native-country  income
0          2174             0              40   United-States   <=50K
1             0             0              13   United-States   <=50K
2             0             0              40   United-States   <=50K
3             0             0              40   United-States   <=50K
4             0             0              40            Cuba   <=50K

```

- **The reason for our actions:** The Census Income dataset contains a mix of continuous and categorical data, making it a good fit for Gaussian Naive Bayes after appropriate preprocessing.
- **How it works:** We use the pandas library to load the dataset from the URL into a DataFrame.
- **Gained outcome:** The first few rows of the dataset are displayed to give us an idea of the data structure.

#### Preprocessing the Data

Before we can train our model, we need to preprocess the data. This includes converting categorical variables into numerical values and normalizing the continuous variables.

Python3`
from sklearn.preprocessing import LabelEncoder
# Convert categorical variables to numerical values
le = LabelEncoder()
categorical_features = ['workclass', 'education', 'marital-status',\
                        'occupation', 'relationship', 'race', 'sex',\
                        'native-country', 'income']
for feature in categorical_features:
    census_data[feature] = le.fit_transform(census_data[feature])
# Normalize continuous variables
census_data[\
['age', 'fnlwgt', 'education-num', 'capital-gain',\
             'capital-loss', 'hours-per-week']] = census_data[\
['age', 'fnlwgt','education-num', 'capital-gain', 'capital-loss',\
'hours-per-week']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# Display the preprocessed data
print(census_data.head())
`

**Output:**

```
        age  workclass    fnlwgt  education  education-num  marital-status  \
0  0.301370          7  0.044302          9       0.800000               4
1  0.452055          6  0.048238          9       0.800000               2
2  0.287671          4  0.138113         11       0.533333               0
3  0.493151          4  0.151068          1       0.400000               2
4  0.150685          4  0.221488          9       0.800000               2
   occupation  relationship  race  sex  capital-gain  capital-loss  \
0           1             1     4    1       0.02174           0.0
1           4             0     4    1       0.00000           0.0
2           6             1     4    1       0.00000           0.0
3           6             0     2    1       0.00000           0.0
4          10             5     2    0       0.00000           0.0
   hours-per-week  native-country  income
0        0.397959              39       0
1        0.122449              39       0
2        0.397959              39       0
3        0.397959              39       0
4        0.397959               5       0

```

- **The reason for our actions:** Preprocessing is essential to ensure that the model receives data in a format it can work with effectively.
- **How it works:** We use [LabelEncoder](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/) to encode categorical features and normalization to scale continuous features.
- **Gained outcome:** The preprocessed data is now ready for training.

#### Training the Gaussian Naive Bayes Model

With our data preprocessed, we can now train the Gaussian Naive Bayes model.

Python3`
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Extract features and labels
X = census_data.drop('income', axis=1)
y = census_data['income']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
# Train the model
gnb.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = gnb.predict(X_test)
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
`

**Output:**

```
Accuracy: 0.8086805200122837

```

- **The reason for our actions:** Training the model allows us to learn the parameters that best fit our data.
- **How it works:** We use the train\_test\_split function to divide our data into training and testing sets. The GaussianNB class is used to initialize and train the model.
- **Gained outcome:** The accuracy score tells us how well our model performs on unseen data.

This example shows how to use the Census Income dataset to apply [Gaussian Naive Bayes](https://www.geeksforgeeks.org/gaussian-naive-bayes/). You may use this approach to forecast income levels based on employment and demographic characteristics by following these steps.

### Conclusion

In this article, we've introduced the Gaussian Naive Bayes classifier and demonstrated its implementation using Scikit-Learn. Understanding the basics of this algorithm, key terminologies, and following the provided steps will empower you to apply Gaussian Naive Bayes to your own projects. As you continue your journey into machine learning, this knowledge will serve as a valuable foundation for more advanced concepts and techniques.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/gaussian-process-regression-gpr-with-noise-level-estimation/)

[Gaussian Process Regression (GPR) with Noise-Level Estimation](https://www.geeksforgeeks.org/gaussian-process-regression-gpr-with-noise-level-estimation/)

[A](https://www.geeksforgeeks.org/user/abhijat_sarari/)

[abhijat\_sarari](https://www.geeksforgeeks.org/user/abhijat_sarari/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Geeks Premier League](https://www.geeksforgeeks.org/category/geeksforgeeks-initiatives/geeks-premier-league/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Python scikit-module](https://www.geeksforgeeks.org/tag/python-scikit-module/)
- [Geeks Premier League 2023](https://www.geeksforgeeks.org/tag/geeks-premier-league-2023/)

+1 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Gaussian Naive Bayes\\
\\
\\
Gaussian Naive Bayes is a type of Naive Bayes method working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset. Before diving deep into this topic we must gain a basic understanding of principles on which Gaussian Naive Bayes work. Here are some\\
\\
5 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes/)
[Classification Metrics using Sklearn\\
\\
\\
Machine learning classification is a powerful tool that helps us make predictions and decisions based on data. Whether it's determining whether an email is spam or not, diagnosing diseases from medical images, or predicting customer churn, classification algorithms are at the heart of many real-worl\\
\\
14 min read](https://www.geeksforgeeks.org/sklearn-classification-metrics/)
[Naive Bayes vs. SVM for Text Classification\\
\\
\\
Text classification is a fundamental task in natural language processing (NLP), with applications ranging from spam detection to sentiment analysis and document categorization. Two popular machine learning algorithms for text classification are Naive Bayes classifier (NB) and Support Vector Machines\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-vs-svm-for-text-classification/)
[Python \| Linear Regression using sklearn\\
\\
\\
Prerequisite: Linear Regression Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecast\\
\\
3 min read](https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/)
[Dummy Classifiers using Sklearn - ML\\
\\
\\
Dummy classifier is a classifier that classifies data with basic rules without producing any insight from the training data. It entirely disregards data trends and outputs the class label based on pre-specified strategies. A dummy classifier is designed to act as a baseline, with which more sophisti\\
\\
3 min read](https://www.geeksforgeeks.org/ml-dummy-classifiers-using-sklearn/)
[ML \| Voting Classifier using Sklearn\\
\\
\\
A Voting Classifier is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output. It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output cl\\
\\
3 min read](https://www.geeksforgeeks.org/ml-voting-classifier-using-sklearn/)
[Decision Tree vs. Naive Bayes Classifier\\
\\
\\
Decision Tree and Naive Bayes are two popular classification algorithms. Both are widely used in various applications such as spam filtering, fraud detection, and medical diagnosis. However, they are based on different theoretical foundations, and their performance varies depending on the nature of\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-vs-naive-bayes-classifier/)
[Bayesian Inference for the Gaussian\\
\\
\\
Bayesian inference is a strong statistical tool for revising beliefs regarding an unknown parameter given newly released data. For Gaussian (Normal) distributed data, Bayesian inference enables us to make inferences of the mean and variance of the underlying normal distribution in a principled manne\\
\\
6 min read](https://www.geeksforgeeks.org/bayesian-inference-for-the-gaussian/)
[Spam Classification using OpenAI\\
\\
\\
The majority of people in today's society own a mobile phone, and they all frequently get communications (SMS/email) on their phones. But the key point is that some of the messages you get may be spam, with very few being genuine or important interactions. You may be tricked into providing your pers\\
\\
6 min read](https://www.geeksforgeeks.org/spam-classification-using-openai/)
[ML \| Naive Bayes Scratch Implementation using Python\\
\\
\\
Naive Bayes is a probabilistic machine learning algorithms based on the Bayes Theorem. It is a simple yet powerful algorithm because of its understanding, simplicity and ease of implementation. It is popular method for classification applications such as spam filtering and text classification. In th\\
\\
7 min read](https://www.geeksforgeeks.org/ml-naive-bayes-scratch-implementation-using-python/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/)

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