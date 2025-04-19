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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/gaussian-naive-bayes/?type%3Darticle%26id%3D1093734&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Machine Learning for Time Series Data in R\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/machine-learning-for-time-series-data-in-r/)

# Gaussian Naive Bayes

Last Updated : 29 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Gaussian Naive Bayes is a type of [Naive Bayes](https://www.geeksforgeeks.org/naive-bayes-classifiers/) method **working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset.** Before diving deep into this topic we must gain a basic understanding of principles on which Gaussian Naive Bayes work. Here are some terminologies that can help us gain knowledge for further study.

## Mathematics Behind Gaussian Naive Bayes

Gaussian Naive Bayes assumes that the likelihood (P(xi∣yx\_i\|yxi​∣y)) follows the Gaussian Distribution for each xix\_ixi​ within yky\_kyk​. Therefore,

P(xi∣y)=1σ2πe−(x−μ)22σ2P(x\_i\|y) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{(x - \\mu)^2}{2\\sigma^2}}P(xi​∣y)=σ2π​1​e−2σ2(x−μ)2​

Where:

- xix\_ixi​ is the feature value,
- μ is the mean of the feature values for a given class yky\_kyk​,
- σ is the standard deviation of the feature values for that class,
- π is a constant (approximately 3.14159),
- eee is the base of the natural logarithm.

To classify each new data point x the algorithm finds out the maximum value of the posterior probability of each class and assigns the data point to that class.

### Gaussian Distribution

The Gaussian distribution is also known as Normal distribution and is a continuous probability distribution defined by its mean μ and standard deviation σ. It is symmetric around the mean and is described by the bell-shaped curve. The formula for the probability density function (PDF) of a Gaussian distribution is:

P(x)=1σ2πexp⁡(−(x−μ)22σ2)P(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} \\exp\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)P(x)=σ2π​1​exp(−2σ2(x−μ)2​)

This distribution is crucial in Gaussian Naive Bayes because the algorithm models each feature xix\_ixi​ as a normal distribution within each class. By assuming that the data for each feature is normally distributed, we can calculate the likelihood of a data point belonging to a particular class.

### Why Gaussian Naive Bayes Works Well for Continuous Data?

Gaussian Naive Bayes is effective for continuous data because it assumes each feature follows a Gaussian (normal) distribution. When this assumption holds true the algorithm performs well. For example in tasks like spam detection, medical diagnosis or predicting house prices where features such as age, income or height fit a normal distribution there Gaussian Naive Bayes can make accurate predictions.

## Python Implementation of Gaussian Naive Bayes

Here we will be applying Gaussian Naive Bayes to the Iris Dataset, this dataset consists of four features namely Sepal Length in cm, Sepal Width in cm, Petal Length in cm, Petal Width in cm and from these features we have to identify which feature set belongs to which specie class. The iris flower dataset is available in Sklearn library of python.

Now we will be using Gaussian Naive Bayes in predicting the correct specie of Iris flower.

Lets break down the above code step by step:

First we will be importing the required libraries:

- **pandas:** for data manipulation
- **load\_iris:** to load dataset
- **train\_test\_split:** to split the data into training and testing sets
- **GaussianNB:** for the Gaussian Naive Bayes classifier
- **accuracy\_score:** to evaluate the model
- **LabelEncoder:** to encode the categorical target variable.

Python`
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
`

- After that we will load the Iris dataset from a CSV file named "Iris.csv" into a pandas DataFrame.
- Then we will separate the features (X) and the target variable (y) from the dataset. Features are obtained by dropping the "Species" column, and the target variable is set to the "Species" column which we will be predicting.

Python`
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target
X = data.drop("Species", axis=1)
y = data['Species']
`

- Since the target variable "Species" is categorical, we will be using LabelEncoder to convert it into numerical form. This is necessary for the Gaussian Naive Bayes classifier, as it requires numerical inputs.
- We will be splitting the dataset into training and testing sets using the train\_test\_split function. 70% of the data is used for training, and 30% is used for testing. The random\_state parameter ensures reproducibility of the same data.

Python`
# Encoding the Species column to get numerical class
le = LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
`

- We will be creating a Gaussian Naive Bayes Classifier(gnb) and then training it on the training data using the fit method.

Python`
# Gaussian Naive Bayes classifier
gnb = GaussianNB()
# Train the classifier on the training data
gnb.fit(X_train, y_train)
`

- At last we will be using the trained model to make predictions on the testing data.

Python`
# Make predictions on the testing data
y_pred = gnb.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f&quot;The Accuracy of Prediction on Iris Flower is: {accuracy}&quot;)
`

**Output:**

> The Accuracy of Prediction on Iris Flower is: 0.9777777777777777

High accuracy suggests that the model has effectively learned to distinguish between the three different species of Iris based on the given features (sepal length, sepal width, petal length and petal width).

Gaussian Naive Bayes classifier provides an effective and efficient approach for classification tasks particularly when the data follows a Gaussian distribution. In the next article, we will explore other variations of Naive Bayes classifier that are [Multinomial Naive Bayes](https://www.geeksforgeeks.org/multinomial-naive-bayes/) and [Bernoulli Naive Bayes](https://www.geeksforgeeks.org/bernoulli-naive-bayes/) which offer different advantages based on type of data.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/machine-learning-for-time-series-data-in-r/)

[Machine Learning for Time Series Data in R](https://www.geeksforgeeks.org/machine-learning-for-time-series-data-in-r/)

[S](https://www.geeksforgeeks.org/user/surajjoshdylh/)

[surajjoshdylh](https://www.geeksforgeeks.org/user/surajjoshdylh/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Geeks Premier League](https://www.geeksforgeeks.org/category/geeksforgeeks-initiatives/geeks-premier-league/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)
- [Machine Learning](https://www.geeksforgeeks.org/tag/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Gaussian Naive Bayes using Sklearn\\
\\
\\
In the world of machine learning, Gaussian Naive Bayes is a simple yet powerful algorithm used for classification tasks. It belongs to the Naive Bayes algorithm family, which uses Bayes' Theorem as its foundation. The goal of this post is to explain the Gaussian Naive Bayes classifier and offer a de\\
\\
8 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/)
[Naive Bayes Classifiers\\
\\
\\
Naive Bayes classifiers are supervised machine learning algorithms used for classification tasks, based on Bayes' Theorem to find probabilities. This article will give you an overview as well as more advanced use and implementation of Naive Bayes in machine learning. Key Features of Naive Bayes Clas\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
[Multinomial Naive Bayes\\
\\
\\
Multinomial Naive Bayes is one of the variation of Naive Bayes algorithm. A classification algorithm based on Bayes' Theorem ideal for discrete data and is typically used in text classification problems. It models the frequency of words as counts and assumes each feature or word is multinomially dis\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes/)
[Bernoulli Naive Bayes\\
\\
\\
Bernoulli Naive Bayes is a subcategory of the Naive Bayes Algorithm. It is typically used when the data is binary and it models the occurrence of features using Bernoulli distribution. It is used for the classification of binary features such as 'Yes' or 'No', '1' or '0', 'True' or 'False' etc. Here\\
\\
5 min read](https://www.geeksforgeeks.org/bernoulli-naive-bayes/)
[Bayesian Inference for the Gaussian\\
\\
\\
Bayesian inference is a strong statistical tool for revising beliefs regarding an unknown parameter given newly released data. For Gaussian (Normal) distributed data, Bayesian inference enables us to make inferences of the mean and variance of the underlying normal distribution in a principled manne\\
\\
6 min read](https://www.geeksforgeeks.org/bayesian-inference-for-the-gaussian/)
[Bayesian Causal Networks\\
\\
\\
A Bayesian Causal Network (BCN) is a probabilistic graphical model that represents the causal relationships between variables using Bayesian inference. It combines Bayesian networks (BN) with causality, allowing us to model dependencies and make predictions even in the presence of uncertainty. The k\\
\\
7 min read](https://www.geeksforgeeks.org/bayesian-causal-networks/)
[Multinomial Naive Bayes Classifier in R\\
\\
\\
The Multinomial Naive Bayes (MNB) classifier is a popular machine learning algorithm, especially useful for text classification tasks such as spam detection, sentiment analysis, and document categorization. In this article, we discuss about the basics of the MNB classifier and how to implement it in\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes-classifier-in-r/)
[Decision Tree vs. Naive Bayes Classifier\\
\\
\\
Decision Tree and Naive Bayes are two popular classification algorithms. Both are widely used in various applications such as spam filtering, fraud detection, and medical diagnosis. However, they are based on different theoretical foundations, and their performance varies depending on the nature of\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-vs-naive-bayes-classifier/)
[Naive Bayes vs. SVM for Text Classification\\
\\
\\
Text classification is a fundamental task in natural language processing (NLP), with applications ranging from spam detection to sentiment analysis and document categorization. Two popular machine learning algorithms for text classification are Naive Bayes classifier (NB) and Support Vector Machines\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-vs-svm-for-text-classification/)
[Building Naive Bayesian classifier with WEKA\\
\\
\\
The use of the Naive Bayesian classifier in Weka is demonstrated in this article. The â€œweather-nominalâ€ data set used in this experiment is available in ARFF format. This paper assumes that the data has been properly preprocessed. The Bayes' Theorem is used to build a set of classification algorithm\\
\\
3 min read](https://www.geeksforgeeks.org/building-naive-bayesian-classifier-with-weka/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/gaussian-naive-bayes/)

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