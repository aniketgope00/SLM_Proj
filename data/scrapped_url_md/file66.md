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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/bernoulli-naive-bayes/?type%3Darticle%26id%3D1077511&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Sentiment Analysis of YouTube Comments\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/sentiment-analysis-of-youtube-comments/)

# Bernoulli Naive Bayes

Last Updated : 29 Jan, 2025

Comments

Improve

Suggest changes

2 Likes

Like

Report

Bernoulli Naive Bayes is a subcategory of the [Naive Bayes Algorithm](https://www.geeksforgeeks.org/naive-bayes-classifiers/). **It is typically used when the data is binary and it models the occurrence of features using Bernoulli distribution.** **It is used for the classification of binary features such as 'Yes' or 'No', '1' or '0', 'True' or 'False' etc.** Here it is to be noted that the features are independent of one another. In this article we will be discussing more about it.

## Mathematics behind Bernoulli Naive Bayes

The core of Bernoulli Naive Bayes is based on [Bayes' Theorem](https://www.geeksforgeeks.org/bayes-theorem/) which helps in calculating the conditional probability of a given class yyy given some data x=(x1,x2,...,xn)x = (x\_1, x\_2, ..., x\_n)x=(x1​,x2​,...,xn​). Now in the Bernoulli Naive Bayes model we assume that each feature is conditionally independent given the class yyy. This means that we can calculate the likelihood of each feature occurring as:

p(xi∣y)=p(i∣y)xi+(1−p(i∣y))(1−xi)p(x\_i\|y)=p(i\|y)x\_i+(1-p(i\|y))(1-x\_i)p(xi​∣y)=p(i∣y)xi​+(1−p(i∣y))(1−xi​)

- Here, p(xix\_ixi​ \|y) is the conditional probability of xi occurring provided y has occurred.
- i is the event
- xix\_ixi​ holds binary value either 0 or 1

Now we will learn Bernoulli distribution as Bernoulli Naive Bayes works on that.

### Bernoulli distribution

[Bernoulli distribution](https://www.geeksforgeeks.org/python-bernoulli-distribution-in-statistics/) is used for discrete probability calculation. It either calculates success or failure. Here the random variable is either 1 or 0 whose chance of occurring is either denoted by p or (1-p) respectively.

The mathematical formula is given

f(x)={px∗(1−p)1−xif x=0,10otherwisef(x)=\\begin{cases} p^x\*(1-p)^{1-x} & \\text{if x=0,1} \\\ 0 \\; otherwise\\\ \\end{cases} f(x)={px∗(1−p)1−x0otherwise​if x=0,1

Now in the above function if we put x=1 then the value of f(x) is p and if we put x=0 then the value of f(x) is 1-p. Here p denotes the success of an event.

## Implementing Bernoulli Naive Bayes

For performing classification using Bernoulli Naive Bayes we have considered an email dataset.

The email dataset comprises of four columns named Unnamed: 0, label, label\_num and text. The category of label is either ham or spam. For ham the number assigned is 0 and for spam 1 is assigned. Text comprises the body of the mail. The length of the dataset is 5171. The dataset can be downloaded from [here](https://media.geeksforgeeks.org/wp-content/uploads/20250127144046966510/spam_ham_dataset.csv).

Python`
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
`

In the above code we have imported necessary libraries like pandas, numpy and sklearn. Bernoulli Naive Bayes is a part of sklearn package.

Python`
df=pd.read_csv("spam_ham_dataset.csv")
print(df.shape)
print(df.columns)
df= df.drop(['Unnamed: 0'], axis=1)
`

**Output:**

> (5171, 4) Index(\['Unnamed: 0', 'label', 'text', 'label\_num'\], dtype='object')

In this above code we have performed a quick data analysis that includes reading the data, dropping unnecessary columns, printing shape of data, information about dataset etc.

Python`
x = df["text"].values
y = df["label_num"].values
cv = CountVectorizer()
x = cv.fit_transform(x)
`

In the above code since text data is used to train our classifier we convert the text into a matrix comprising numbers using Count Vectorizer so that the model can perform well.

Python`
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
bnb = BernoulliNB(binarize=0.0)
model = bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
`

**Output:**

> precision recall f1-score support
>
> 0 0.84 0.98 0.91 732
>
> 1 0.92 0.56 0.70 303
>
> accuracy 0.86 1035
>
> macro avg 0.88 0.77 0.80 1035
>
> weighted avg 0.87 0.86 0.84 1035

The classification report shows that for class 0 (not spam) precision, recall and F1 score are 0.84, 0.98 and 0.91 respectively. For class 1 (spam) they are 0.92, 0.56 and 0.70. The recall for class 1 drops due to the 13% spam data. The overall accuracy of the model is 86%, which is good.

Bernoulli Naive Bayes is used for spam detection, text classification, Sentiment Analysis and used to determine whether a certain word is present in a document or not.

## Difference Between Different Naive Bayes Model

| Aspect | Gaussian Naive Bayes | Multinomial Naive Bayes | Bernoulli Naive Bayes |
| --- | --- | --- | --- |
| **Feature Type** | Continuous (real-valued features) | Discrete (count data or frequency-based features) | Binary (presence or absence of features) |
| **Assumption** | Assumes data follows a Gaussian (normal) distribution | Assumes data follows a multinomial distribution | Assumes data follows a Bernoulli (binary) distribution |
| **Common Use Case** | Suitable for continuous features like height, weight, etc. | Suitable for text classification (word counts) | Suitable for binary classification tasks (e.g., spam detection) |
| **Data Representation** | Features are treated as continuous variables | Features are treated as discrete counts or frequencies | Features are treated as binary (0 or 1) values |
| **Mathematical Model** | Uses Gaussian distribution (mean and variance) for each feature | Uses the multinomial distribution for word counts in text classification | Uses Bernoulli distribution (probability of a feature being present) |
| **Example** | Predicting whether an email is spam based on numeric features | Predicting whether a document is spam based on word counts | Classifying a document as spam or not based on word presence |

Here is the quick comparison between types of Naive Bayes that are [Gaussian Naive Bayes](https://www.geeksforgeeks.org/gaussian-naive-bayes/), [Multinomial Naive Bayes](https://www.geeksforgeeks.org/multinomial-naive-bayes/) and Bernoulli Naive Bayes

Bernoulli Naive Bayes is a simple yet effective for binary classification tasks. Its efficiency in handling binary data makes it suitable for applications like spam detection, sentiment analysis and many more. Its simplicity and speed makes it suitable for real-time classification problems.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/sentiment-analysis-of-youtube-comments/)

[Sentiment Analysis of YouTube Comments](https://www.geeksforgeeks.org/sentiment-analysis-of-youtube-comments/)

[![author](https://media.geeksforgeeks.org/auth/profile/62wp4qjhm5d0hcpwuls9)](https://www.geeksforgeeks.org/user/jhimlic1/)

[jhimlic1](https://www.geeksforgeeks.org/user/jhimlic1/)

Follow

2

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Geeks Premier League](https://www.geeksforgeeks.org/category/geeksforgeeks-initiatives/geeks-premier-league/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Python scikit-module](https://www.geeksforgeeks.org/tag/python-scikit-module/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)
- [Geeks Premier League 2023](https://www.geeksforgeeks.org/tag/geeks-premier-league-2023/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

+3 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Multinomial Naive Bayes\\
\\
\\
Multinomial Naive Bayes is one of the variation of Naive Bayes algorithm. A classification algorithm based on Bayes' Theorem ideal for discrete data and is typically used in text classification problems. It models the frequency of words as counts and assumes each feature or word is multinomially dis\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes/)
[Gaussian Naive Bayes\\
\\
\\
Gaussian Naive Bayes is a type of Naive Bayes method working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset. Before diving deep into this topic we must gain a basic understanding of principles on which Gaussian Naive Bayes work. Here are some\\
\\
5 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes/)
[Naive Bayes Classifiers\\
\\
\\
Naive Bayes classifiers are supervised machine learning algorithms used for classification tasks, based on Bayes' Theorem to find probabilities. This article will give you an overview as well as more advanced use and implementation of Naive Bayes in machine learning. Key Features of Naive Bayes Clas\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
[Bernoulli's Principle\\
\\
\\
Bernoulli's Principle is a very important concept in Fluid Mechanics which is the study of fluids (like air and water) and their interaction with other fluids. Bernoulli's principle is also referred to as Bernoulli's Equation or Bernoulli Theorem. This principle was first stated by Daniel Bernoulli\\
\\
15+ min read](https://www.geeksforgeeks.org/bernoullis-principle/)
[Multinomial Naive Bayes Classifier in R\\
\\
\\
The Multinomial Naive Bayes (MNB) classifier is a popular machine learning algorithm, especially useful for text classification tasks such as spam detection, sentiment analysis, and document categorization. In this article, we discuss about the basics of the MNB classifier and how to implement it in\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes-classifier-in-r/)
[Complement Naive Bayes (CNB) Algorithm\\
\\
\\
Naive Bayes algorithms are a group of very popular and commonly used Machine Learning algorithms used for classification. There are many different ways the Naive Bayes algorithm is implemented like Gaussian Naive Bayes, Multinomial Naive Bayes, etc. To learn more about the basics of Naive Bayes, you\\
\\
7 min read](https://www.geeksforgeeks.org/complement-naive-bayes-cnb-algorithm/)
[Bayes' Theorem\\
\\
\\
Bayes' Theorem is a mathematical formula that helps determine the conditional probability of an event based on prior knowledge and new evidence. It adjusts probabilities when new information comes in and helps make better decisions in uncertain situations. Bayes' Theorem helps us update probabilitie\\
\\
12 min read](https://www.geeksforgeeks.org/bayes-theorem/)
[Building Naive Bayesian classifier with WEKA\\
\\
\\
The use of the Naive Bayesian classifier in Weka is demonstrated in this article. The â€œweather-nominalâ€ data set used in this experiment is available in ARFF format. This paper assumes that the data has been properly preprocessed. The Bayes' Theorem is used to build a set of classification algorithm\\
\\
3 min read](https://www.geeksforgeeks.org/building-naive-bayesian-classifier-with-weka/)
[Bayes Theorem in Machine learning\\
\\
\\
Bayes' theorem is fundamental in machine learning, especially in the context of Bayesian inference. It provides a way to update our beliefs about a hypothesis based on new evidence. What is Bayes theorem?Bayes' theorem is a fundamental concept in probability theory that plays a crucial role in vario\\
\\
5 min read](https://www.geeksforgeeks.org/bayes-theorem-in-machine-learning/)
[Gaussian Naive Bayes using Sklearn\\
\\
\\
In the world of machine learning, Gaussian Naive Bayes is a simple yet powerful algorithm used for classification tasks. It belongs to the Naive Bayes algorithm family, which uses Bayes' Theorem as its foundation. The goal of this post is to explain the Gaussian Naive Bayes classifier and offer a de\\
\\
8 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/bernoulli-naive-bayes/)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=bsfbogt884sz)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=swpct3f02xml)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=invisible&cb=q7l5xjee8xg4)