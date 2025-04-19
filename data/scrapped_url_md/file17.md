- [Deep Learning Tutorial](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Data Analysis Tutorial](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Python â€“ Data visualization tutorial](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [R](https://www.geeksforgeeks.org/r-tutorial/)
- [Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/)
- [Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning-projects/)_)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)_)
- [Machine Learning Mathematics](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Deep Learning Project](https://www.geeksforgeeks.org/5-deep-learning-project-ideas-for-beginners/)_)
- [Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning-interview-questions/)_)
- [Computer Vision Tutorial](https://www.geeksforgeeks.org/computer-vision/)
- [Computer Vision Projects](https://www.geeksforgeeks.org/computer-vision-projects/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [NLP Project](https://www.geeksforgeeks.org/nlp-project-ideas-for-beginners/))
- [NLP Interview Questions](https://www.geeksforgeeks.org/nlp-interview-questions/))
- [Statistics with Python](https://www.geeksforgeeks.org/statistics-with-python/)
- [100 Days of Machine Learning](https://www.geeksforgeeks.org/100-days-of-machine-learning/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/?type%3Darticle%26id%3D477532&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Creating a simple machine learning model\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/creating-a-simple-machine-learning-model/)

# Flowchart for basic Machine Learning models

Last Updated : 05 Sep, 2020

Comments

Improve

Suggest changes

9 Likes

Like

Report

Machine learning tasks have been divided into three categories, depending upon the feedback available:

1. **Supervised Learning:** These are human builds models based on input and output.
2. **Unsupervised Learning:** These are models that depend on human input. No labels are given to the learning algorithm, the model has to figure out the structure by itself.
3. **Reinforcement learning:** These are the models that are feed with human inputs. No labels are given to the learning algorithm. The algorithm learns by the rewards and penalties given.

The algorithms that can be used for each of the categories are:

| **Algorithm** | **Supervised** | **Unsupervised** | **Reinforcement** |
| --- | --- | --- | --- |
| Linear | 1 | 0 | 0 |
| Logistic | 1 | 0 | 0 |
| K-Means | 1 | 1 | 0 |
| Anomaly Detection | 1 | 1 | 0 |
| Neural Net | 1 | 1 | 1 |
| KNN | 1 | 0 | 0 |
| Decision Tee | 1 | 0 | 0 |
| Random Forest | 1 | 0 | 0 |
| SVM | 1 | 0 | 0 |
| Naive Bayes | 1 | 0 | 0 |

The machine learning functions and uses for various tasks are given in the below table. To know more about the Algorithms [click here.](https://www.geeksforgeeks.org/choosing-a-suitable-machine-learning-algorithm/)

| **Category** | **Algorithm** | **Function** | **Use** |
| --- | --- | --- | --- |
| **Basic Regression** | _Linear_ | linear\_model.LinearRegression() | Lots of numerical data |
| _Logistic_ | linear\_model.LogisticRegression() | Target variable is categorical |
| **Cluster Analysis** | _K-Means_ | cluster.KMeans() | Similar datum into groups based on centroids |
| _Anomaly Detection_ | covariance.EllipticalEnvelope() | Finding outliers through grouping |
| **Classification** | _Neural Net_ | neural\_network.MLPClassifier() | Complex relationships. Prone to over fitting. |
| _K-NN_ | neighbors.KNeighborsClassifier() | Group membership based on proximity |
| _Decision Tee_ | tree.DecisionTreeClassifier() | If/then/else. Non-contiguous data. Can also be regression. |
| _Random Forest_ | ensemble.RandomForestClassifier() | Find best split randomly. Can also be regression |
| _SVM_ | svm.SVC() <br>svm.LinearSVC() | Maximum margin classifier. Fundamental. Data Science algorithm |
| _Naive Bayes_ | GaussianNB() MultinominalNB() BernoulliNB() | Updating knowledge step by step with new info |
| **Feature Reduction** | _T-DISTRIB Stochastic NEIB Embedding_ | manifold.TSNE() | Visual high dimensional data. Convert similarity to joint probabilities |
| _Principle Component Analysis_ | decomposition.PCA() | Distill feature space into components that describe the greatest variance |
| _Canonical Correlation Analysis_ | decomposition.CCA() | Making sense of cross-correlation matrices |
| _Linear Discriminant Analysis_ | lda.LDA() | Linear combination of features that separates classes |

> The flowchart given below will help you give a rough guide of each estimator that will help to know more about the task and the ways to solve it using various ML techniques.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200820022507/tusss-660x312.JPG)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/creating-a-simple-machine-learning-model/)

[Creating a simple machine learning model](https://www.geeksforgeeks.org/creating-a-simple-machine-learning-model/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

9

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Deep-Learning](https://www.geeksforgeeks.org/tag/deep-learning/)

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

Like9

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1836382360.1745055237&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=53941639)

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

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)