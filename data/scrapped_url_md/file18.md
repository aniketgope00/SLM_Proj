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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/creating-a-simple-machine-learning-model/?type%3Darticle%26id%3D204691&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Machine Learning Model Evaluation\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/machine-learning-model-evaluation/)

# Creating a simple machine learning model

Last Updated : 15 Apr, 2025

Comments

Improve

Suggest changes

10 Likes

Like

Report

Machine Learning models are the core of smart applications. To get a better insight into how machine learning models work, let us discuss one of the most basic algorithms: Linear Regression. This is employed in predicting a dependent variable based on one or multiple independent variables by utilizing a linear relationship.

In this article, we will go through step by step procedure to create and test a Linear Regression model in Python with synthetically generated data.

## What is Linear Regression?

In machine learning, linear regression is a supervised learning algorithm for predicting a continuous outcome variable based on one or more predictor variables. To understand the working of linear regression mathematically, you might take a look at Linear Regression.

The equation of a linear model is:

> Y=w1X1+w2X2+wnXn+bY= w\_1X\_1 + w\_2X\_2 + w\_nX\_n + bY=w1​X1​+w2​X2​+wn​Xn​+b

Where:

- w1, w2 are the coefficients (weights),
- b is the bias or intercept term.

we’ll generate data such that the output is given by:

> Y = a+ 2b+ 3c

### Step 1: Generating the Training Dataset

We will generate 100 random data samples, each with 3 features `a`, `b`, and `c`.

Python`
# python library to generate random numbers
from random import randint
# limit under which the samples are to be generated
TRAIN_SET_LIMIT = 1000
# count of samples
TRAIN_SET_COUNT = 100
# Lists to store input and corresponding outputs
TRAIN_INPUT = []
TRAIN_OUTPUT = []
# loop to create 100 data  items with three columns each
for _ in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    # Output based on the formula: a + 2b + 3c
    output = a + (2 * b) + (3 * c)
    TRAIN_INPUT.append([a, b, c])

    # adding each output to output list
    TRAIN_OUTPUT.append(output)
`

### Step 2: Training the Linear Regression Model

We’ll use the `LinearRegression` class from `scikit-learn` to train our model on the generated dataset.

Python`
from sklearn.linear_model import LinearRegression
# Initialize the Linear Regression model
model = LinearRegression(n_jobs=-1)
# Train the model
model.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
`

**Output:**

![linear_model](https://media.geeksforgeeks.org/wp-content/uploads/20250415111045640784/linear_model.png)

### Step 3: Testing the Model

We will test the trained model with a known input and compare the predictions with the expected output.

Python`
X_TEST = [[10, 20, 30]] #Test data
predicted_output = model.predict(X=X_TEST)
# Getting the coefficients
coefficients = model.coef_
print(f"Outcome : {predicted_output}")
print(f"Coefficients : {coefficients}")
`

**Output:**

```
Outcome : [140.]
 Coefficients : [1. 2. 3.]
```

Now to check if the output is correct

> Y= 10 + (2∗20) + (3∗30) = 10 +40 +90 = 140

Hence, the model perfectly learned the underlying relationship in the training dataset with the help of linear regression.

**Related Article:**

- [Linear Regression in Machine learning](https://www.geeksforgeeks.org/ml-linear-regression/)
- [Linear Regression Formula](https://www.geeksforgeeks.org/linear-regression-formula/)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/machine-learning-model-evaluation/)

[Machine Learning Model Evaluation](https://www.geeksforgeeks.org/machine-learning-model-evaluation/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

10

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

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

Like10

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/creating-a-simple-machine-learning-model/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1046329052.1745055239&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=1974129975)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055239666&cv=11&fst=1745055239666&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fcreating-a-simple-machine-learning-model%2F%3Fref%3Dlbp&_ng=1&hn=www.googleadservices.com&frm=0&tiba=Creating%20a%20simple%20machine%20learning%20model%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=14742215.1745055240&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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