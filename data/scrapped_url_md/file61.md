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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/naive-bayes-classifiers/?type%3Darticle%26id%3D141142&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Linear Regression (Python Implementation)\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

# Naive Bayes Classifiers

Last Updated : 02 Apr, 2025

Comments

Improve

Suggest changes

191 Likes

Like

Report

Naive Bayes classifiers are supervised machine learning algorithms used for classification tasks, based on [Bayes’ Theorem](https://www.geeksforgeeks.org/bayes-theorem/) to find probabilities. This article will give you an overview as well as more advanced use and implementation of Naive Bayes in machine learning.

## Key Features of Naive Bayes Classifiers

The main idea behind the Naive Bayes classifier is to use **Bayes’ Theorem** to classify data based on the probabilities of different classes given the features of the data. It is used mostly in high-dimensional text classification

- The Naive Bayes Classifier is a simple probabilistic classifier and it has very few number of parameters which are used to build the ML models that can predict at a faster speed than other classification algorithms.
- It is a probabilistic classifier because it assumes that one feature in the model is independent of existence of another feature. In other words, each feature contributes to the predictions with no relation between each other.
- Naïve Bayes Algorithm is used in spam filtration, Sentimental analysis, classifying articles and many more.

## Why it is Called Naive Bayes?

It is named as “Naive” because it assumes the presence of one feature does not affect other features.

The “Bayes” part of the name refers to  for the basis in Bayes’ Theorem.

Consider a fictional dataset that describes the weather conditions for playing a game of golf. Given the weather conditions, each tuple classifies the conditions as fit(“Yes”) or unfit(“No”) for playing golf. Here is a tabular representation of our dataset.

|  | Outlook | Temperature | Humidity | Windy | Play Golf |
| --- | --- | --- | --- | --- | --- |
| 0 | Rainy | Hot | High | False | No |
| 1 | Rainy | Hot | High | True | No |
| 2 | Overcast | Hot | High | False | Yes |
| 3 | Sunny | Mild | High | False | Yes |
| 4 | Sunny | Cool | Normal | False | Yes |
| 5 | Sunny | Cool | Normal | True | No |
| 6 | Overcast | Cool | Normal | True | Yes |
| 7 | Rainy | Mild | High | False | No |
| 8 | Rainy | Cool | Normal | False | Yes |
| 9 | Sunny | Mild | Normal | False | Yes |
| 10 | Rainy | Mild | Normal | True | Yes |
| 11 | Overcast | Mild | High | True | Yes |
| 12 | Overcast | Hot | Normal | False | Yes |
| 13 | Sunny | Mild | High | True | No |

The dataset is divided into two parts, namely, **feature matrix** and the **response vector**.

- Feature matrix contains all the vectors(rows) of dataset in which each vector consists of the value of **dependent features**. In above dataset, features are ‘Outlook’, ‘Temperature’, ‘Humidity’ and ‘Windy’.
- Response vector contains the value of **class variable**(prediction or output) for each row of feature matrix. In above dataset, the class variable name is ‘Play golf’.

## Assumption of Naive Bayes

The fundamental Naive Bayes assumption is that each feature makes an:

- **Feature independence:** This means that when we are trying to classify something, we assume that each feature (or piece of information) in the data does not affect any other feature.
- **Continuous features are normally distributed:** If a feature is continuous, then it is assumed to be normally distributed within each class.
- **Discrete features have multinomial distributions:** If a feature is discrete, then it is assumed to have a multinomial distribution within each class.
- **Features are equally important:** All features are assumed to contribute equally to the prediction of the class label.
- **No missing data:** The data should not contain any missing values.

> The assumptions made by Naive Bayes are not generally correct in real-world situations. In-fact, the independence assumption is never correct but often works well in practice. Now, before moving to the formula for Naive Bayes, it is important to know about Bayes’ theorem.

## **Understanding Bayes’ Theorem for naive bayes**

[Bayes’ Theorem](https://www.geeksforgeeks.org/bayes-theorem/) finds the probability of an event occurring given the probability of another event that has already occurred. Bayes’ theorem is stated mathematically as the following equation:

P(y∣X)=P(X∣y)P(y)P(X) P(y\|X) = \\frac{P(X\|y) P(y)}{P(X)}

P(y∣X)=P(X)P(X∣y)P(y)​

where A and B are events and P(B) ≠ 0

**Where,**

**P(A\|B) is Posterior probability**: Probability of hypothesis A on the observed event B.

**P(B\|A) is Likelihood probability**: Probability of the evidence given that the probability of a hypothesis is true.X=(x1,x2,x3,…..,xn) X = (x\_1,x\_2,x\_3,…..,x\_n) X=(x1​,x2​,x3​,…..,xn​)

Now, with regards to our dataset, we can apply Bayes’ theorem in following way:

![NaiveBayesExample](https://media.geeksforgeeks.org/wp-content/uploads/20250320105352155839/NaiveBayesExample.png)

Example Tables for Naive Bayes

where, y is class variable and X is a dependent feature vector (of size _n_) where:

P(No∣today)=P(SunnyOutlook∣No)P(HotTemperature∣No)P(NormalHumidity∣No)P(NoWind∣No)P(No)P(today) P(No \| today) = \\frac{P(Sunny Outlook\|No)P(Hot Temperature\|No)P(Normal Humidity\|No)P(No Wind\|No)P(No)}{P(today)}

P(No∣today)=P(today)P(SunnyOutlook∣No)P(HotTemperature∣No)P(NormalHumidity∣No)P(NoWind∣No)P(No)​

Just to clear, an example of a feature vector and corresponding class variable can be: (refer 1st row of dataset)

```
X = (Rainy, Hot, High, False)
y = No
```

So basically, P(y∣X)P(y\|X)P(y∣X)here means, the probability of “Not playing golf” given that the weather conditions are “Rainy outlook”, “Temperature is hot”, “high humidity” and “no wind”.

With relation to our dataset, this concept can be understood as:

- We assume that no pair of features are dependent. For example, the temperature being ‘Hot’ has nothing to do with the humidity or the outlook being ‘Rainy’ has no effect on the winds. Hence, the features are assumed to be **independent**.
- Secondly, each feature is given the same weight(or importance). For example, knowing only temperature and humidity alone can’t predict the outcome accurately. None of the attributes is irrelevant and assumed to be contributing **equally** to the outcome.

Now, its time to put a naive assumption to the Bayes’ theorem, which is, **independence** among the features. So now, we split **evidence** into the independent parts.

Now, if any two events A and B are independent, then,

```
P(A,B) = P(A)P(B)

```

Hence, we reach to the result:

P(y∣x1,…,xn)=P(x1∣y)P(x2∣y)…P(xn∣y)P(y)P(x1)P(x2)…P(xn) P(y\|x\_1,…,x\_n) = \\frac{ P(x\_1\|y)P(x\_2\|y)…P(x\_n\|y)P(y)}{P(x\_1)P(x\_2)…P(x\_n)} P(y∣x1​,…,xn​)=P(x1​)P(x2​)…P(xn​)P(x1​∣y)P(x2​∣y)…P(xn​∣y)P(y)​

which can be expressed as:

P(y∣x1,…,xn)=P(y)∏i=1nP(xi∣y)P(x1)P(x2)…P(xn) P(y\|x\_1,…,x\_n) = \\frac{P(y)\\prod\_{i=1}^{n}P(x\_i\|y)}{P(x\_1)P(x\_2)…P(x\_n)} P(y∣x1​,…,xn​)=P(x1​)P(x2​)…P(xn​)P(y)∏i=1n​P(xi​∣y)​

Now, as the denominator remains constant for a given input, we can remove that term:

P(y∣x1,…,xn)∝P(y)∏i=1nP(xi∣y) P(y\|x\_1,…,x\_n)\\propto P(y)\\prod\_{i=1}^{n}P(x\_i\|y) P(y∣x1​,…,xn​)∝P(y)∏i=1n​P(xi​∣y)

Now, we need to create a classifier model. For this, we find the probability of given set of inputs for all possible values of the class variable _y_ and pick up the output with maximum probability. This can be expressed mathematically as:

y=argmaxyP(y)∏i=1nP(xi∣y)y = argmax\_{y} P(y)\\prod\_{i=1}^{n}P(x\_i\|y) y=argmaxy​P(y)∏i=1n​P(xi​∣y)

So, finally, we are left with the task of calculating P(y) P(y) P(y)and P(xi∣y)P(x\_i \| y)P(xi​∣y).

Please note that P(y)P(y)P(y) is also called class probability and P(xi∣y)P(x\_i \| y)P(xi​∣y) is called conditional probability.

The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi∣y).P(x\_i \| y).P(xi​∣y).

Let us try to apply the above formula manually on our weather dataset. For this, we need to do some precomputations on our dataset.

We need to findP(xi∣yj) P(x\_i \| y\_j) P(xi​∣yj​)for each xix\_ixi​ in X andyjy\_jyj​ in y. All these calculations have been demonstrated in the tables below:

P(Yes∣today)∝39.29.69.69.914≈0.02116 P(Yes \| today) \\propto \\frac{3}{9}.\\frac{2}{9}.\\frac{6}{9}.\\frac{6}{9}.\\frac{9}{14} \\approx 0.02116

P(Yes∣today)∝93​.92​.96​.96​.149​≈0.02116

So, in the figure above, we have calculated P(xi∣yj)P(x\_i \| y\_j)P(xi​ ∣yj​) for each xix\_ixi​ in X and yjy\_jyj​ in y manually in the tables 1-4. For example, probability of playing golf given that the temperature is cool, i.e P(temp. = cool \| play golf = Yes) = 3/9.

Also, we need to find class probabilities P(y)P(y)P(y) which has been calculated in the table 5. For example, P(play golf = Yes) = 9/14.

So now, we are done with our pre-computations and the classifier is ready!

Let us test it on a new set of features (let us call it today):

```
today = (Sunny, Hot, Normal, False)
```

P(Yes∣today)=P(SunnyOutlook∣Yes)P(HotTemperature∣Yes)P(NormalHumidity∣Yes)P(NoWind∣Yes)P(Yes)P(today) P(Yes \| today) = \\frac{P(Sunny Outlook\|Yes)P(Hot Temperature\|Yes)P(Normal Humidity\|Yes)P(No Wind\|Yes)P(Yes)}{P(today)} P(Yes∣today)=P(today)P(SunnyOutlook∣Yes)P(HotTemperature∣Yes)P(NormalHumidity∣Yes)P(NoWind∣Yes)P(Yes)​

and probability to not play golf is given by:

P(No∣today)∝35.25.15.25.514≈0.0068 P(No \| today) \\propto \\frac{3}{5}.\\frac{2}{5}.\\frac{1}{5}.\\frac{2}{5}.\\frac{5}{14} \\approx 0.0068

P(No∣today)∝53​.52​.51​.52​.145​≈0.0068

Since, P(today) is common in both probabilities, we can ignore P(today) and find proportional probabilities as:

P(Yes∣today)+P(No∣today)=1 P(Yes \| today) + P(No \| today) = 1

P(Yes∣today)+P(No∣today)=1

and

P(Yes∣today)=0.021160.02116+0.0068≈0.75 P(Yes \| today) = \\frac{0.02116}{0.02116 + 0.0068} \\approx 0.75

P(Yes∣today)=0.02116+0.00680.02116​≈0.75

Now, since

P(No∣today)=0.00680.0141+0.0068≈0.32 P(No \| today) = \\frac{0.0068}{0.0141 + 0.0068} \\approx 0.32

P(No∣today)=0.0141+0.00680.0068​≈0.32

These numbers can be converted into a probability by making the sum equal to 1 (normalization):

P(Yes∣today)>P(No∣today) P(Yes \| today) > P(No \| today)

P(Yes∣today)>P(No∣today)

and

P(xi∣y)=12πσy2exp(−(xi−μy)22σy2) P(x\_i \| y) = \\frac{1}{\\sqrt{2\\pi\\sigma \_{y}^{2} }} exp \\left (-\\frac{(x\_i-\\mu \_{y})^2}{2\\sigma \_{y}^{2}} \\right )

P(xi​∣y)=2πσy2​​1​exp(−2σy2​(xi​−μy​)2​)

Since

So, prediction that golf would be played is ‘ **Yes**’.

The method that we discussed above is applicable for discrete data. In case of continuous data, we need to make some assumptions regarding the distribution of values of each feature. The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of P(xi∣y).P(x\_i \| y).P(xi​∣y).

## Types of Naive Bayes Model

There are three types of Naive Bayes Model :

### **Gaussian Naive Bayes**

In [**Gaussian Naive Bayes**](https://www.geeksforgeeks.org/gaussian-naive-bayes/), continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. A Gaussian distribution is also called [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) When plotted, it gives a bell shaped curve which is symmetric about the mean of the feature values as shown below:

### **Multinomial Naive Bayes**

[**Multinomial Naive Bayes**](https://www.geeksforgeeks.org/multinomial-naive-bayes/) is used when features represent the frequency of terms (such as word counts) in a document. It is commonly applied in text classification, where term frequencies are important.

### **Bernoulli Naive Bayes**

[**Bernoulli Naive Bayes**](https://www.geeksforgeeks.org/bernoulli-naive-bayes/) deals with binary features, where each feature indicates whether a word appears or not in a document. It is suited for scenarios where the presence or absence of terms is more relevant than their frequency. Both models are widely used in document classification tasks

## Advantages of Naive Bayes Classifier

- Easy to implement and computationally efficient.
- Effective in cases with a large number of features.
- Performs well even with limited training data.
- It performs well in the presence of categorical features.
- For numerical features data is assumed to come from normal distributions

## Disadvantages of Naive Bayes Classifier

- Assumes that features are independent, which may not always hold in real-world data.
- Can be influenced by irrelevant attributes.
- May assign zero probability to unseen events, leading to poor generalization.

## Applications of Naive Bayes Classifier

- **Spam Email Filtering**: Classifies emails as spam or non-spam based on features.
- **Text Classification**: Used in sentiment analysis, document categorization, and topic classification.
- **Medical Diagnosis:** Helps in predicting the likelihood of a disease based on symptoms.
- **Credit Scoring:** Evaluates creditworthiness of individuals for loan approval.
- **Weather Prediction**: Classifies weather conditions based on various factors.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

[Linear Regression (Python Implementation)](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

[![author](https://media.geeksforgeeks.org/auth/profile/sb7ciorr5k5t22woqkes)](https://www.geeksforgeeks.org/user/kartik/)

[kartik](https://www.geeksforgeeks.org/user/kartik/)

Follow

191

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Machine Learning](https://www.geeksforgeeks.org/tag/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Passive Aggressive Classifiers\\
\\
\\
The Passive-Aggressive algorithms are a family of Machine learning algorithms that are not very well known by beginners and even intermediate Machine Learning enthusiasts. However, they can be very useful and efficient for certain applications. Note: This is a high-level overview of the algorithm ex\\
\\
5 min read](https://www.geeksforgeeks.org/passive-aggressive-classifiers/?ref=ml_lbp)
[Decision Tree vs. Naive Bayes Classifier\\
\\
\\
Decision Tree and Naive Bayes are two popular classification algorithms. Both are widely used in various applications such as spam filtering, fraud detection, and medical diagnosis. However, they are based on different theoretical foundations, and their performance varies depending on the nature of\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-vs-naive-bayes-classifier/?ref=ml_lbp)
[Gaussian Naive Bayes\\
\\
\\
Gaussian Naive Bayes is a type of Naive Bayes method working on continuous attributes and the data features that follows Gaussian distribution throughout the dataset. Before diving deep into this topic we must gain a basic understanding of principles on which Gaussian Naive Bayes work. Here are some\\
\\
5 min read](https://www.geeksforgeeks.org/gaussian-naive-bayes/?ref=ml_lbp)
[Multinomial Naive Bayes Classifier in R\\
\\
\\
The Multinomial Naive Bayes (MNB) classifier is a popular machine learning algorithm, especially useful for text classification tasks such as spam detection, sentiment analysis, and document categorization. In this article, we discuss about the basics of the MNB classifier and how to implement it in\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes-classifier-in-r/?ref=ml_lbp)
[Ridge Classifier\\
\\
\\
Supervised Learning is the type of Machine Learning that uses labelled data to train the model. Both Regression and Classification belong to the category of Supervised Learning. Regression: This is used to predict a continuous range of values using one or more features. These features act as the ind\\
\\
10 min read](https://www.geeksforgeeks.org/ridge-classifier/?ref=ml_lbp)
[Building Naive Bayesian classifier with WEKA\\
\\
\\
The use of the Naive Bayesian classifier in Weka is demonstrated in this article. The â€œweather-nominalâ€ data set used in this experiment is available in ARFF format. This paper assumes that the data has been properly preprocessed. The Bayes' Theorem is used to build a set of classification algorithm\\
\\
3 min read](https://www.geeksforgeeks.org/building-naive-bayesian-classifier-with-weka/?ref=ml_lbp)
[Naive Bayes vs. SVM for Text Classification\\
\\
\\
Text classification is a fundamental task in natural language processing (NLP), with applications ranging from spam detection to sentiment analysis and document categorization. Two popular machine learning algorithms for text classification are Naive Bayes classifier (NB) and Support Vector Machines\\
\\
9 min read](https://www.geeksforgeeks.org/naive-bayes-vs-svm-for-text-classification/?ref=ml_lbp)
[Multinomial Naive Bayes\\
\\
\\
Multinomial Naive Bayes is one of the variation of Naive Bayes algorithm. A classification algorithm based on Bayes' Theorem ideal for discrete data and is typically used in text classification problems. It models the frequency of words as counts and assumes each feature or word is multinomially dis\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes/?ref=ml_lbp)
[Bernoulli Naive Bayes\\
\\
\\
Bernoulli Naive Bayes is a subcategory of the Naive Bayes Algorithm. It is typically used when the data is binary and it models the occurrence of features using Bernoulli distribution. It is used for the classification of binary features such as 'Yes' or 'No', '1' or '0', 'True' or 'False' etc. Here\\
\\
5 min read](https://www.geeksforgeeks.org/bernoulli-naive-bayes/?ref=ml_lbp)
[Rule-Based Classifier - Machine Learning\\
\\
\\
Rule-based classifiers are just another type of classifier which makes the class decision depending by using various "if..else" rules. These rules are easily interpretable and thus these classifiers are generally used to generate descriptive models. The condition used with "if" is called the anteced\\
\\
4 min read](https://www.geeksforgeeks.org/rule-based-classifier-machine-learning/?ref=ml_lbp)

Like191

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/naive-bayes-classifiers/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=156766953.1745056369&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1082642744)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)