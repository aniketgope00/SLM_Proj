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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-using-svm-to-perform-classification-on-a-non-linear-dataset/?type%3Darticle%26id%3D265882&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Exposing ML/DL Models as REST APIs\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/exposing-ml-dl-models-as-rest-apis/)

# ML \| Using SVM to perform classification on a non-linear dataset

Last Updated : 15 Jan, 2019

Comments

Improve

Suggest changes

15 Likes

Like

Report

**Prerequisite:** [Support Vector Machines](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

**Definition of a hyperplane and SVM classifier:**

For a linearly separable dataset having n features (thereby needing n dimensions for representation), a hyperplane is basically an (n – 1) dimensional subspace used for separating the dataset into two sets, each set containing data points belonging to a different class. For example, for a dataset having two features X and Y (therefore lying in a 2-dimensional space), the separating hyperplane is a line (a 1-dimensional subspace). Similarly, for a dataset having 3-dimensions, we have a 2-dimensional separating hyperplane, and so on.

In machine learning, Support Vector Machine (SVM) is a non-probabilistic, linear, binary classifier used for classifying data by learning a hyperplane separating the data.

**Classifying a non-linearly separable dataset using a SVM – a linear classifier:**

As mentioned above SVM is a linear classifier which learns an (n – 1)-dimensional classifier for classification of data into two classes. However, it can be used for classifying a non-linear dataset. This can be done by projecting the dataset into a higher dimension in which it is linearly separable!

To get a better understanding, let’s consider circles dataset.

|     |
| --- |
| `# importing libraries `<br>`import` `numpy as np `<br>`import` `matplotlib.pyplot as plt `<br>`from` `sklearn.datasets ` `import` `make_circles `<br>`from` `mpl_toolkits.mplot3d ` `import` `Axes3D `<br>` `<br>`# generating data `<br>`X, Y ` `=` `make_circles(n_samples ` `=` `500` `, noise ` `=` `0.02` `) `<br>` `<br>`# visualizing data `<br>`plt.scatter(X[:, ` `0` `], X[:, ` `1` `], c ` `=` `Y, marker ` `=` `'.'` `) `<br>`plt.show() ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/Circles.png)

The dataset is clearly a non-linear dataset and consists of two features (say, X and Y).

In order to use SVM for classifying this data, introduce another feature Z = X2 \+ Y2 into the dataset. Thus, projecting the 2-dimensional data into 3-dimensional space. The first dimension representing the feature X, second representing Y and third representing Z (which, mathematically, is equal to the radius of the circle of which the point (x, y) is a part of). Now, clearly, for the data shown above, the ‘yellow’ data points belong to a circle of smaller radius and the ‘purple’ data points belong to a circle of larger radius. Thus, the data becomes linearly separable along the Z-axis.

|     |
| --- |
| `# adding a new dimension to X `<br>`X1 ` `=` `X[:, ` `0` `].reshape((` `-` `1` `, ` `1` `)) `<br>`X2 ` `=` `X[:, ` `1` `].reshape((` `-` `1` `, ` `1` `)) `<br>`X3 ` `=` `(X1` `*` `*` `2` `+` `X2` `*` `*` `2` `) `<br>`X ` `=` `np.hstack((X, X3)) `<br>` `<br>`# visualizing data in higher dimension `<br>`fig ` `=` `plt.figure() `<br>`axes ` `=` `fig.add_subplot(` `111` `, projection ` `=` `'3d'` `) `<br>`axes.scatter(X1, X2, X1` `*` `*` `2` `+` `X2` `*` `*` `2` `, c ` `=` `Y, depthshade ` `=` `True` `) `<br>`plt.show() ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/3d1.png)

Now, we can use SVM (or, for that matter, any other linear classifier) to learn a 2-dimensional separating hyperplane. This is how the hyperplane would look like:

|     |
| --- |
| `# create support vector classifier using a linear kernel `<br>`from` `sklearn ` `import` `svm `<br>` `<br>`svc ` `=` `svm.SVC(kernel ` `=` `'linear'` `) `<br>`svc.fit(X, Y) `<br>`w ` `=` `svc.coef_ `<br>`b ` `=` `svc.intercept_ `<br>` `<br>`# plotting the separating hyperplane `<br>`x1 ` `=` `X[:, ` `0` `].reshape((` `-` `1` `, ` `1` `)) `<br>`x2 ` `=` `X[:, ` `1` `].reshape((` `-` `1` `, ` `1` `)) `<br>`x1, x2 ` `=` `np.meshgrid(x1, x2) `<br>`x3 ` `=` `-` `(w[` `0` `][` `0` `]` `*` `x1 ` `+` `w[` `0` `][` `1` `]` `*` `x2 ` `+` `b) ` `/` `w[` `0` `][` `2` `] `<br>` `<br>`fig ` `=` `plt.figure() `<br>`axes2 ` `=` `fig.add_subplot(` `111` `, projection ` `=` `'3d'` `) `<br>`axes2.scatter(X1, X2, X1` `*` `*` `2` `+` `X2` `*` `*` `2` `, c ` `=` `Y, depthshade ` `=` `True` `) `<br>`axes1 ` `=` `fig.gca(projection ` `=` `'3d'` `) `<br>`axes1.plot_surface(x1, x2, x3, alpha ` `=` `0.01` `) `<br>`plt.show() ` |

```

```

```

```

![](https://media.geeksforgeeks.org/wp-content/uploads/hyperplane.png)

Thus, using a linear classifier we can separate a non-linearly separable dataset.

**A brief introduction to kernels in machine learning:**

In machine learning, a trick known as “kernel trick” is used to learn a linear classifier to classify a non-linear dataset. It transforms the linearly inseparable data into a linearly separable one by projecting it into a higher dimension. A kernel function is applied on each data instance to map the original non-linear data points into some higher dimensional space in which they become linearly separable.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/exposing-ml-dl-models-as-rest-apis/)

[Exposing ML/DL Models as REST APIs](https://www.geeksforgeeks.org/exposing-ml-dl-models-as-rest-apis/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

15

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Classification on Imbalanced data using Tensorflow\\
\\
\\
In the modern days of machine learning, imbalanced datasets are like a curse that degrades the overall model performance in classification tasks. In this article, we will implement a Deep learning model using TensorFlow for classification on a highly imbalanced dataset. Classification on Imbalanced\\
\\
7 min read](https://www.geeksforgeeks.org/classification-on-imbalanced-data-using-tensorflow/)
[Image classification using Support Vector Machine (SVM) in Python\\
\\
\\
Support Vector Machines (SVMs) are a type of supervised machine learning algorithm that can be used for classification and regression tasks. In this article, we will focus on using SVMs for image classification. When a computer processes an image, it perceives it as a two-dimensional array of pixels\\
\\
9 min read](https://www.geeksforgeeks.org/image-classification-using-support-vector-machine-svm-in-python/)
[Multi-class classification using Support Vector Machines (SVM)\\
\\
\\
Support Vector Machines (SVM) are widely recognized for their effectiveness in binary classification tasks. However, real-world problems often require distinguishing between more than two classes. This is where multi-class classification comes into play. While SVMs are inherently binary classifiers,\\
\\
6 min read](https://www.geeksforgeeks.org/multi-class-classification-using-support-vector-machines-svm/)
[Classification of text documents using sparse features in Python Scikit Learn\\
\\
\\
Classification is a type of machine learning algorithm in which the model is trained, so as to categorize or label the given input based on the provided features for example classifying the input image as an image of a dog or a cat (binary classification) or to classify the provided picture of a liv\\
\\
5 min read](https://www.geeksforgeeks.org/classification-of-text-documents-using-sparse-features-in-python-scikit-learn/)
[Linear vs. Non-linear Classification: Analyzing Differences Using the Kernel Trick\\
\\
\\
Classification is a fundamental task in machine learning, where the goal is to assign a class label to a given input. There are two primary approaches to classification: linear and non-linear. Support Vector Machines (SVMs) are a popular choice for classification tasks due to their robustness and ef\\
\\
8 min read](https://www.geeksforgeeks.org/linear-vs-non-linear-classification-analyzing-differences-using-the-kernel-trick/)
[Classification of Data Mining Systems\\
\\
\\
Data Mining is considered as an interdisciplinary field. It includes a set of various disciplines such as statistics, database systems, machine learning, visualization and information sciences.Classification of the data mining system helps users to understand the system and match their requirements\\
\\
1 min read](https://www.geeksforgeeks.org/classification-of-data-mining-systems/)
[Text Classification using Decision Trees in Python\\
\\
\\
Text classification is the process of classifying the text documents into predefined categories. In this article, we are going to explore how we can leverage decision trees to classify the textual data. Text Classification and Decision Trees Text classification involves assigning predefined categori\\
\\
5 min read](https://www.geeksforgeeks.org/text-classification-using-decision-trees-in-python/)
[Classifying data using Support Vector Machines(SVMs) in Python\\
\\
\\
Introduction to SVMs: In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. A Support Vector Machine (SVM) is a discriminative classifier\\
\\
4 min read](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)
[ML \| Cancer cell classification using Scikit-learn\\
\\
\\
Machine learning is used in solving real-world problems including medical diagnostics. One such application is classifying cancer cells based on their features and determining whether they are 'malignant' or 'benign'. In this article, we will use Scikit-learn to build a classifier for cancer cell de\\
\\
4 min read](https://www.geeksforgeeks.org/ml-cancer-cell-classification-using-scikit-learn/)
[Classifying data using Support Vector Machines(SVMs) in R\\
\\
\\
In machine learning, Support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. It is mostly used in classification problems. In this algorithm, each data item is plotted as a point in n-dimensio\\
\\
5 min read](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-r/)

Like15

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-using-svm-to-perform-classification-on-a-non-linear-dataset/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1472087338.1745055903&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=1436651291)

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