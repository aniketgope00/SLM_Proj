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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/multinomial-naive-bayes/?type%3Darticle%26id%3D1093537&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
BSc Data Science Course : Updated Syllabus 2024, Top Colleges, Jobs, Salary & Scope\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/bsc-data-science/)

# Multinomial Naive Bayes

Last Updated : 29 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Multinomial Naive Bayesis one of the variation of [Naive Bayes algorithm](https://www.geeksforgeeks.org/naive-bayes-classifiers/). A classification algorithm based on Bayes' Theorem **ideal for discrete data and is typically used in text classification problems.** _**It models the frequency of words as counts and assumes each feature or word is multinomially distributed.**_ MNB is widely used for tasks like classifying documents based on word frequencies like in spam email detection.

## How Does Multinomial Naive Bayes Work?

In Multinomial Naive bayes - **Naive** means that the method **assumes all features like words in a sentence are independent from each other and Multinomial refers to how many times a word appears or how often a category occurs.** It works by **using word counts to classify text.** The main idea is that it assumes each word in a message or feature is independent of each others. This means the presence of one word doesn't affect the presence of another word which makes the model easy to use.

The model looks at how many times each word appears in messages from different categories (like "spam" or "not spam"). For example if the word "free" appears often in spam messages that will be used to help predict whether a new message is spam or not.

To calculate the probability of a message belonging to a certain category Multinomial Naive Bayes uses the **multinomial distribution**:

P(X) = \\frac{n!}{n\_1! n\_2! \\ldots n\_m!} p\_1^{n\_1} p\_2^{n\_2} \\ldots p\_m^{n\_m}

Where:

- n is the total number of trials.
- ni is the count of occurrences for outcome i.
- pi is the probability of outcome i.

To estimate how likely each word is in a particular class like "spam" or "not spam" we use a method called [**Maximum Likelihood Estimation (MLE)**](https://www.geeksforgeeks.org/probability-density-estimation-maximum-likelihood-estimation/) **.** This helps finding probabilities based on actual counts from our data. The formula is:

\\quad \\theta\_{c,i} = \\frac{\\text{count}(w\_i, c) + 1}{N + v}

Where:

- count(wi,c) is the number of times word wi appears in documents of class c.
- \\Nu is the total number of words in documents of class c _c_.
- v is the vocabulary size.

## Implementation of Multinomial Naive Bayes

Let's understand it with a example of spam email detection. We'll classify emails into two categories: **spam** and **not spam**.

1\. **Importing Libraries**:

- **`pandas`**: Used for handling data in DataFrame format.
- **`CountVectorizer`**: Converts a collection of text documents into a matrix of token counts.
- **`train_test_split`**: Splits the data into training and test sets for model evaluation.
- **`MultinomialNB`**: A Naive Bayes classifier suited for classification tasks with discrete features (such as word counts).
- **`accuracy_score`**: Computes the accuracy of the model's predictions.

Python`
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
`

2\. **Creating the Dataset**

Python`
data = {
    'text': [\
        'Free money now',\
        'Call now to claim your prize',\
        'Meet me at the park',\
        'Let’s catch up later',\
        'Win a new car today!',\
        'Lunch plans?',\
        'Congratulations! You won a lottery',\
        'Can you send me the report?',\
        'Exclusive offer for you',\
        'Are you coming to the meeting?'\
    ],
    'label': ['spam', 'spam', 'not spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}
df = pd.DataFrame(data)
`

- A simple dataset is created with text messages labeled as either **spam** or **not spam**.
- This data is then converted into a DataFrame for easy handling.

3\. **Mapping Labels to Numerical Values**

Python`
df['label'] = df['label'].map({'spam': 1, 'not spam': 0})
`

- The labels ( `spam` and `not spam`) are mapped to numerical values where `spam` becomes `1` and `not spam` becomes `0`. This is necessary for the classifier, as it works with numerical data.

4\. **Splitting the Data**

Python`
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
`

- `X` contains the text messages (features), and `y` contains the labels (target).
- The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.

5\. **Vectorizing the Text Data**

Python`
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
`

- **`CountVectorizer`** is used to convert text data into numerical vectors. It counts the occurrences of each word in the corpus.
- `fit_transform()` is applied to the training data to learn the vocabulary and transform it into a feature matrix.
- `transform()` is applied to the test data to convert it into the same feature space.

6\. **Training the Naive Bayes Model**

Python`
model = MultinomialNB()
model.fit(X_train_vectors, y_train)
`

- A **Multinomial Naive Bayes** classifier is created and trained using the vectorized training data ( `X_train_vectors`) and corresponding labels ( `y_train`).

7\. **Making Predictions and Evaluating Accuracy**

Python`
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")
`

8\. **Predicting for a Custom Message**

Python`
custom_message = ["Congratulations, you've won a free vacation"]
custom_vector = vectorizer.transform(custom_message)
prediction = model.predict(custom_vector)
print("Prediction for custom message:", "Spam" if prediction[0] == 1 else "Not Spam")
`

**Output:**

> Accuracy: 66.67%
>
> Prediction for custom message: Spam

In the above code we did spam detection for given set of messages and evaluated model accuracy for the output it gave.

## How Multinomial Naive Bayes differs from Gaussian Naive Bayes?

The Multinomial naive bayes and [Gaussian naive bayes](https://www.geeksforgeeks.org/gaussian-naive-bayes/) both are the variants of same algorithm. However they have several number of differences which are discussed below:

| Multinomial Naive Bayes | Gaussian Naive Bayes |
| --- | --- |
| It specially designed for discrete data particularly text data. | It is suitable for continuous data where features follow a Gaussian distribution. |
| It assumes features and represent its counts like word counts. | It assumes a Gaussian distribution for the likelihood. |
| It is commonly used in NLP for document classification tasks. | It is commonly used in tasks involving continuous data such as medical diagnosis, fraud detection and weather prediction. |
| The likelihood of each feature is calculated using the multinomial distribution. | The likelihood of each feature is modeled using the Gaussian distribution. |
| It is more efficient when the number of features is very high like in text datasets with thousands of words. | It can handle continuous data but if the data is sparse or contains many outliers it struggle with accuracy |

Multinomial Naive Bayes efficiency combined with its ability to handle large datasets makes it useful for applications like document categorization and email filtering. In the next article we will explore the [Bernoulli Naive Bayes](https://www.geeksforgeeks.org/bernoulli-naive-bayes/) model which is another variant of Naive Bayes.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/bsc-data-science/)

[BSc Data Science Course : Updated Syllabus 2024, Top Colleges, Jobs, Salary & Scope](https://www.geeksforgeeks.org/bsc-data-science/)

[![author](https://media.geeksforgeeks.org/auth/profile/cor53rinikmzcmzuvt63)](https://www.geeksforgeeks.org/user/susmit_sekhar_bhakta/)

[susmit\_sekhar\_bhakta](https://www.geeksforgeeks.org/user/susmit_sekhar_bhakta/)

Follow

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

[Multinomial Naive Bayes Classifier in R\\
\\
\\
The Multinomial Naive Bayes (MNB) classifier is a popular machine learning algorithm, especially useful for text classification tasks such as spam detection, sentiment analysis, and document categorization. In this article, we discuss about the basics of the MNB classifier and how to implement it in\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes-classifier-in-r/)
[Gaussian Naive Bayes\\
\\
\\
Gaussian Naive Bayes is a type of Naive Bayes method working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset. Before diving deep into this topic we must gain a basic understanding of principles on which Gaussian Naive Bayes work. Here are some\\
\\
5 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes/)
[Bernoulli Naive Bayes\\
\\
\\
Bernoulli Naive Bayes is a subcategory of the Naive Bayes Algorithm. It is typically used when the data is binary and it models the occurrence of features using Bernoulli distribution. It is used for the classification of binary features such as 'Yes' or 'No', '1' or '0', 'True' or 'False' etc. Here\\
\\
5 min read](https://www.geeksforgeeks.org/bernoulli-naive-bayes/)
[Applying Multinomial Naive Bayes to NLP Problems\\
\\
\\
Multinomial Naive Bayes (MNB) is a popular machine learning algorithm for text classification problems in Natural Language Processing (NLP). It is particularly useful for problems that involve text data with discrete features such as word frequency counts. MNB works on the principle of Bayes theorem\\
\\
7 min read](https://www.geeksforgeeks.org/applying-multinomial-naive-bayes-to-nlp-problems/)
[Naive Bayes Classifiers\\
\\
\\
Naive Bayes classifiers are supervised machine learning algorithms used for classification tasks, based on Bayes' Theorem to find probabilities. This article will give you an overview as well as more advanced use and implementation of Naive Bayes in machine learning. Key Features of Naive Bayes Clas\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-classifiers/)
[Complement Naive Bayes (CNB) Algorithm\\
\\
\\
Naive Bayes algorithms are a group of very popular and commonly used Machine Learning algorithms used for classification. There are many different ways the Naive Bayes algorithm is implemented like Gaussian Naive Bayes, Multinomial Naive Bayes, etc. To learn more about the basics of Naive Bayes, you\\
\\
7 min read](https://www.geeksforgeeks.org/complement-naive-bayes-cnb-algorithm/)
[Gaussian Naive Bayes using Sklearn\\
\\
\\
In the world of machine learning, Gaussian Naive Bayes is a simple yet powerful algorithm used for classification tasks. It belongs to the Naive Bayes algorithm family, which uses Bayes' Theorem as its foundation. The goal of this post is to explain the Gaussian Naive Bayes classifier and offer a de\\
\\
8 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes-using-sklearn/)
[Important Naive Bayes Interview Questions\\
\\
\\
Naive Bayes is a foundational algorithm in machine learning based on Bayes' Theorem - which is a way to calculate the probability of an event occurring given some prior knowledge. For classifying, it helps predict the class (or category) of a new data point based on its features. Naive Bayes is a co\\
\\
7 min read](https://www.geeksforgeeks.org/important-naive-bayes-interview-questions/)
[ML \| Naive Bayes Scratch Implementation using Python\\
\\
\\
Naive Bayes is a probabilistic machine learning algorithms based on the Bayes Theorem. It is a simple yet powerful algorithm because of its understanding, simplicity and ease of implementation. It is popular method for classification applications such as spam filtering and text classification. In th\\
\\
7 min read](https://www.geeksforgeeks.org/ml-naive-bayes-scratch-implementation-using-python/)
[Naive Bayes vs. SVM for Text Classification\\
\\
\\
Text classification is a fundamental task in natural language processing (NLP), with applications ranging from spam detection to sentiment analysis and document categorization. Two popular machine learning algorithms for text classification are Naive Bayes classifier (NB) and Support Vector Machines\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-vs-svm-for-text-classification/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/multinomial-naive-bayes/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1516086674.1745056381&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1356199245)