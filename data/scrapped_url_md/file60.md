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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/?type%3Darticle%26id%3D291528&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Prediction of Wine type using Deep Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/prediction-of-wine-type-using-deep-learning/)

# k-nearest neighbor algorithm in Python

Last Updated : 28 Jan, 2025

Comments

Improve

Suggest changes

18 Likes

Like

Report

[K-Nearest Neighbors (KNN)](https://www.geeksforgeeks.org/k-nearest-neighbours/) is a non-parametric, instance-based learning method. It operates for classification as well as regression:

1. **Classification**: For a new data point, the algorithm identifies its nearest neighbors based on a distance metric (e.g., Euclidean distance). The predicted class is determined by the majority class among these neighbors.
2. **Regression**: The algorithm predicts the value for a new data point by averaging the values of its nearest neighbors.

**Quick Revision :** It works by identifying the **‘k’ nearest data points** (neighbors) to a given input and predicting its **class** or **value** based on the majority class or the average of its neighbors. In this article, we will explore the concept of the KNN algorithm and demonstrate its implementation using **Python’s Scikit-Learn library**.

## Implementation of KNN : Step-by-Step

Choosing the optimal **k-value** is critical before building the model for balancing the model’s performance.

- A **smaller k** value makes the model sensitive to noise, leading to overfitting (complex models).
- A **larger k** value results in smoother boundaries, reducing model complexity but possibly underfitting.

Python`
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
irisData = load_iris()
X = irisData.data
y = irisData.target
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print(knn.predict(X_test))
`

In the example shown above following steps are performed:

1. The k-nearest neighbor algorithm is imported from the scikit-learn package.
2. Create feature and target variables.
3. Split data into training and test data.
4. Generate a k-NN model using neighbors value.
5. Train or fit the data into the model.
6. Predict the future.

We have seen how we can use K-NN algorithm to solve the supervised machine learning problem. But how to measure the accuracy of the model?

Consider an example shown below where we predicted the performance of the above model:

Python`
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# Loading data
irisData = load_iris()
# Create feature and target arrays
X = irisData.data
y = irisData.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
# Calculate the accuracy of the model
print(knn.score(X_test, y_test))
`

**Model Accuracy:** So far so good. But how to decide the right k-value for the dataset?

Obviously, we need to be familiar to data to get the range of expected k-value, but to get the exact k-value we need to test the model for each and every expected k-value. Refer to the example shown below.

Python`
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
irisData = load_iris()
# Create feature and target arrays
X = irisData.data
y = irisData.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
`

**Output:**

![](https://media.geeksforgeeks.org/wp-content/uploads/20190409134306/knn_algo_gfg.png)

Here in the example shown above, we are creating a plot to see the k-value for which we have high accuracy.

**Note:** This is a technique which is not used industry-wide to choose the correct value of n\_neighbors. Instead, we do hyperparameter tuning to choose the value that gives the best performance. We will be covering this in future posts.

**Summary –**

In this post, we have understood what supervised learning is and what are its categories. After having a basic understanding of Supervised learning we explored the k-nearest neighbor algorithm which is used to solve supervised machine learning problems. We also explored measuring the accuracy of the model.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/prediction-of-wine-type-using-deep-learning/)

[Prediction of Wine type using Deep Learning](https://www.geeksforgeeks.org/prediction-of-wine-type-using-deep-learning/)

[T](https://www.geeksforgeeks.org/user/tavishaggarwal1993/)

[tavishaggarwal1993](https://www.geeksforgeeks.org/user/tavishaggarwal1993/)

Follow

18

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)
- [ML-Regression](https://www.geeksforgeeks.org/tag/ml-regression/)

+1 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[K-Nearest Neighbor(KNN) Algorithm\\
\\
\\
K-Nearest Neighbors (KNN) is a simple way to classify things by looking at whatâ€™s nearby. Imagine a streaming service wants to predict if a new user is likely to cancel their subscription (churn) based on their age. They checks the ages of its existing users and whether they churned or stayed. If mo\\
\\
10 min read](https://www.geeksforgeeks.org/k-nearest-neighbours/?ref=ml_lbp)
[kNN: k-Nearest Neighbour Algorithm in R From Scratch\\
\\
\\
In this article, we are going to discuss what is KNN algorithm, how it is coded in R Programming Language, its application, advantages and disadvantages of the KNN algorithm. kNN algorithm in RKNN can be defined as a K-nearest neighbor algorithm. It is a supervised learning algorithm that can be use\\
\\
15+ min read](https://www.geeksforgeeks.org/knn-k-nearest-neighbour-algorithm-in-r-from-scratch/?ref=ml_lbp)
[K Nearest Neighbors with Python \| ML\\
\\
\\
K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection. The K-Nearest Neighbors (KNN) algorithm is a simple, easy\\
\\
5 min read](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/?ref=ml_lbp)
[Implementation of K Nearest Neighbors\\
\\
\\
Prerequisite: K nearest neighbors Introduction Say we are given a data set of items, each having numerically valued features (like Height, Weight, Age, etc). If the count of features is n, we can represent the items as points in an n-dimensional grid. Given a new item, we can calculate the distance\\
\\
10 min read](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/?ref=ml_lbp)
[How To Predict Diabetes using K-Nearest Neighbor in R\\
\\
\\
In this article, we are going to predict Diabetes using the K-Nearest Neighbour algorithm and analyze on Diabetes dataset using the R Programming Language. What is the K-Nearest Neighbor algorithm?The K-Nearest Neighbor (KNN) algorithm is a popular supervised learning classifier frequently used by d\\
\\
13 min read](https://www.geeksforgeeks.org/how-to-predict-diabetes-using-k-nearest-neighbor-in-r/?ref=ml_lbp)
[K-Nearest Neighbors and Curse of Dimensionality\\
\\
\\
In high-dimensional data, the performance of the k-nearest neighbor (k-NN) algorithm often deteriorates due to increased computational complexity and the breakdown of the assumption that similar points are proximate. These challenges hinder the algorithm's accuracy and efficiency in high-dimensional\\
\\
6 min read](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/?ref=ml_lbp)
[Implementation of K-Nearest Neighbors from Scratch using Python\\
\\
\\
Instance-Based LearningK Nearest Neighbors Classification is one of the classification techniques based on instance-based learning. Models based on instance-based learning to generalize beyond the training examples. To do so, they store the training examples first. When it encounters a new instance\\
\\
8 min read](https://www.geeksforgeeks.org/implementation-of-k-nearest-neighbors-from-scratch-using-python/?ref=ml_lbp)
[r-Nearest neighbors\\
\\
\\
r-Nearest neighbors are a modified version of the k-nearest neighbors. The issue with k-nearest neighbors is the choice of k. With a smaller k, the classifier would be more sensitive to outliers. If the value of k is large, then the classifier would be including many points from other classes. It is\\
\\
5 min read](https://www.geeksforgeeks.org/r-nearest-neighbors/?ref=ml_lbp)
[Mathematical explanation of K-Nearest Neighbour\\
\\
\\
KNN stands for K-nearest neighbour is a popular algorithm in Supervised Learning commonly used for classification tasks. It works by classifying data based on its similarity to neighboring data points. The core idea of KNN is straightforward when a new data point is introduced the algorithm finds it\\
\\
4 min read](https://www.geeksforgeeks.org/mathematical-explanation-of-k-nearest-neighbour/?ref=ml_lbp)
[Finding k-nearest Neighbor for Only One Point Using R\\
\\
\\
The k-nearest neighbors (k-NN) algorithm is a simple yet powerful tool used in various machine learning and data mining applications. While k-NN is often applied to an entire dataset to classify or predict values for multiple points, there are scenarios where you may need to find the k-nearest neigh\\
\\
3 min read](https://www.geeksforgeeks.org/finding-k-nearest-neighbor-for-only-one-point-using-r/?ref=ml_lbp)

Like18

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2096201982.1745056366&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=1827340257)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056366145&cv=11&fst=1745056366145&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb858768136&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fk-nearest-neighbor-algorithm-in-python%2F&hn=www.googleadservices.com&frm=0&tiba=k-nearest%20neighbor%20algorithm%20in%20Python%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1685361074.1745056366&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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

[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)