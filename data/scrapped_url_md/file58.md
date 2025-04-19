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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/k-nearest-neighbours/?type%3Darticle%26id%3D141822&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Classifying data using Support Vector Machines(SVMs) in Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

# K-Nearest Neighbor(KNN) Algorithm

Last Updated : 29 Jan, 2025

Comments

Improve

Suggest changes

57 Likes

Like

Report

K-Nearest Neighbors (KNN) is a simple way to classify things by looking at what’s nearby. _**Imagine a streaming service wants to predict if a new user is likely to cancel their subscription (churn) based on their age**_. They checks the ages of its existing users and whether they churned or stayed. _**If most of the “K” closest users in age of new user canceled their subscription KNN will predict the new user might churn too. The key idea is that users with similar ages tend to have similar behaviors and KNN uses this closeness to make decisions.**_

## Getting Started with K-Nearest Neighbors

K-Nearest Neighbors is also called as a **lazy learner algorithm** because it does not learn from the training set immediately instead it stores the dataset and at the time of classification it performs an action on the dataset.

As an example, consider the following table of data points containing two features:

![KNN Algorithm working visualization](https://media.geeksforgeeks.org/wp-content/uploads/20200616145419/Untitled2781.png)

KNN Algorithm working visualization

The new point is classified as **Category 2** because most of its closest neighbors are blue squares. KNN assigns the category based on the majority of nearby points.

The image shows how KNN predicts the category of a **new data point** based on its closest neighbours.

- The **red diamonds** represent **Category 1** and the **blue squares** represent **Category 2**.
- The **new data point** checks its closest neighbours (circled points).
- Since the majority of its closest neighbours are blue squares (Category 2) KNN predicts the new data point belongs to Category 2.

KNN works by using proximity and majority voting to make predictions.

## What is ‘K’ in K Nearest Neighbour ?

In the **k-Nearest Neighbours (k-NN)** algorithm **k** is just a number that tells the algorithm how many nearby points (neighbours) to look at when it makes a decision.

### Example:

Imagine you’re deciding which fruit it is based on its shape and size. You compare it to fruits you already know.

- If **k = 3**, the algorithm looks at the 3 closest fruits to the new one.
- If 2 of those 3 fruits are apples and 1 is a banana, the algorithm says the new fruit is an apple because most of its neighbours are apples.

### How to choose the value of k for KNN Algorithm?

The value of k is critical in KNN as it determines the number of neighbors to consider when making predictions. Selecting the optimal value of k depends on the characteristics of the input data. **If the dataset has significant outliers or noise a higher k can help smooth out the predictions and reduce the influence of noisy data. However choosing very high value can lead to underfitting where the model becomes too simplistic.**

**Statistical Methods for Selecting k**:

- **Cross-Validation**: A robust method for selecting the best k is to perform k-fold [cross-validation](https://www.geeksforgeeks.org/cross-validation-machine-learning/). This involves splitting the data into k subsets training the model on some subsets and testing it on the remaining ones and repeating this for each subset. The value of k that results in the highest average validation accuracy is usually the best choice.
- **Elbow Method**: In the [**elbow method**](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/) we plot the model’s error rate or accuracy for different values of k. As we increase k the error usually decreases initially. However after a certain point the error rate starts to decrease more slowly. This point where the curve forms an “elbow” that point is considered as best k.
- **Odd Values for k**: It’s also recommended to choose an odd value for k especially in classification tasks to avoid ties when deciding the majority class.

## Distance Metrics Used in KNN Algorithm

KNN uses distance metrics to identify nearest neighbour, these neighbours are used for classification and regression task. To identify nearest neighbour we use below distance metrics:

### 1\. Euclidean Distance

Euclidean distance is defined as the straight-line distance between two points in a plane or space. You can think of it like the shortest path you would walk if you were to go directly from one point to another.

distance(x,Xi)=∑j=1d(xj–Xij)2\] \\text{distance}(x, X\_i) = \\sqrt{\\sum\_{j=1}^{d} (x\_j – X\_{i\_j})^2} \]distance(x,Xi​)=∑j=1d​(xj​–Xij​​)2​\]

### 2\. Manhattan Distance

This is the total distance you would travel if you could only move along horizontal and vertical lines (like a grid or city streets). It’s also called “taxicab distance” because a taxi can only drive along the grid-like streets of a city.

d(x,y)=∑i=1n∣xi−yi∣d\\left ( x,y \\right )={\\sum\_{i=1}^{n}\\left \| x\_i-y\_i \\right \|}d(x,y)=∑i=1n​∣xi​−yi​∣

### 3\. Minkowski Distance

Minkowski distance is like a family of distances, which includes both **Euclidean** and **Manhattan distances** as special cases.

d(x,y)=(∑i=1n(xi−yi)p)1pd\\left ( x,y \\right )=\\left ( {\\sum\_{i=1}^{n}\\left ( x\_i-y\_i \\right )^p} \\right )^{\\frac{1}{p}}d(x,y)=(∑i=1n​(xi​−yi​)p)p1​

From the formula above we can say that when p = 2 then it is the same as the formula for the Euclidean distance and when p = 1 then we obtain the formula for the Manhattan distance.

So, you can think of Minkowski as a flexible distance formula that can look like either Manhattan or Euclidean distance depending on the value of p

## Working of KNN algorithm

Thе K-Nearest Neighbors (KNN) algorithm operates on the principle of similarity where it predicts the label or value of a new data point by considering the labels or values of its K nearest neighbors in the training dataset.

![Workings of KNN algorithm](https://media.geeksforgeeks.org/wp-content/uploads/20231207103856/KNN-Algorithm-(1).png)

Step-by-Step explanation of how KNN works is discussed below:

### Step 1: Selecting the optimal value of K

- K represents the number of nearest neighbors that needs to be considered while making prediction.

### Step 2: Calculating distance

- To measure the similarity between target and training data points Euclidean distance is used. Distance is calculated between data points in the dataset and target point.

### Step 3: Finding Nearest Neighbors

- The k data points with the smallest distances to the target point are nearest neighbors.

### Step 4: Voting for Classification or Taking Average for Regression

- When you want to classify a data point into a category (like spam or not spam), the K-NN algorithm looks at the **K closest points** in the dataset. These closest points are called neighbors. The algorithm then looks at which category the neighbors belong to and picks the one that appears the most. This is called **majority voting**.
- In regression, the algorithm still looks for the **K closest points**. But instead of voting for a class in classification, it takes the **average** of the values of those K neighbors. This average is the predicted value for the new point for the algorithm.

![knn](https://media.geeksforgeeks.org/wp-content/uploads/20250127150953958573/knn.gif)

Working of KNN Algorithm

It shows how a test point is classified based on its nearest neighbors. As the test point moves the algorithm identifies the closest ‘k’ data points i.e 5 in this case and assigns test point the majority class label that is grey label class here.

## **Python Implementation of KNN Algorithm**

**1\. Importing Libraries**:

Python`
import numpy as np
from collections import Counter
`

- **`Counter`**: is used to count the occurrences of elements in a list or iterable. In KNN after finding the `k` nearest neighbors labels `Counter` helps count how many times each label appears.

**2\. Defining the Euclidean Distance Function**:

Python`
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))
`

- **euclidean\_distance**: to calculate euclidean distance between points

**3\. KNN Prediction Function**:

Python`
def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]
`

- **distances.append**: Each distance is paired with the corresponding label ( `training_labels[i]`) of the training data. This pair is stored in a list called `distances`.
- **distances.sort**: The list of distances is sorted in ascending order so that the closest points are at the beginning of the list.
- **k\_nearest\_labels**: The function then selects the labels of the `k` closest neighbors.
- The labels of the `k` nearest neighbors are counted using the `Counter` class, and the most frequent label is returned as the prediction for the `test_point`. This is based on the majority vote of the k neighbors.

**4\. Training Data, Labels and Test Point**:

Python`
training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3
`

**5\. Prediction and Output**:

Python`
prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)
`

**Output:**

> A

The algorithm calculates the distances of the test point `[4, 5]` to all training points, selects the 3 closest points (as `k = 3`), and determines their labels. Since the majority of the closest points are labelled **‘A’**, the test point is classified as **‘A’**.

In machine learning we can also use Scikit Learn python library which has in built functions to perform KNN machine learning model and for that you refer to [Implementation of KNN classifier using Sklearn.](https://www.geeksforgeeks.org/ml-implementation-of-knn-classifier-using-sklearn/)

## Applications of the KNN Algorithm

Here are some real life applications of KNN Algorithm.

- **Recommendation Systems**: Many recommendation systems, such as those used by Netflix or Amazon, rely on KNN to suggest products or content. KNN observes at user behavior and finds similar users. If user A and user B have similar preferences, KNN might recommend movies that user A liked to user B.
- **Spam Detection**: KNN is widely used in filtering spam emails. By comparing the features of a new email with those of previously labeled spam and non-spam emails, KNN can predict whether a new email is spam or not.
- **Customer Segmentation**: In marketing firms, KNN is used to segment customers based on their purchasing behavior . By comparing new customers to existing customers, KNN can easily group customers into segments with similar choices and preferences. This helps businesses target the right customers with right products or advertisements.
- **Speech Recognition**: KNN is often used in speech recognition systems to transcribe spoken words into text. The algorithm compares the features of the spoken input with those of known speech patterns. It then predicts the most likely word or command based on the closest matches.

## Advantages and Disadvantages of the KNN Algorithm

**Advantages:**

- **Easy to implement:** The KNN algorithm is easy to implement because its complexity is relatively low as compared to other machine learning algorithms.
- **No training required:** KNN stores all data in memory and doesn’t require any training so when new data points are added it automatically adjusts and uses the new data for future predictions.
- **Few Hyperparameters:** The only parameters which are required in the training of a KNN algorithm are the value of k and the choice of the distance metric which we would like to choose from our evaluation metric.
- **Flexible**: It works for **Classification** problem like is this email spam or not?and also work for **Regression task** like predicting house prices based on nearby similar houses.

**Disadvantages:**

- **Doesn’t scale well:** KNN is considered as a “lazy” algorithm as it is very slow especially with large datasets
- **Curse of Dimensionality:** When the number of features increases KNN struggles to classify data accurately a problem known as [curse of dimensionality](https://www.geeksforgeeks.org/videos/curse-of-dimensionality-in-machine-learning).
- **Prone to Overfitting:** As the algorithm is affected due to the curse of dimensionality it is prone to the problem of overfitting as well.

> **Also Check for more understanding:**
>
> - [K Nearest Neighbors with Python \| ML](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python)
> - [Implementation of K-Nearest Neighbors from Scratch using Python](https://www.geeksforgeeks.org/implementation-of-k-nearest-neighbors-from-scratch-using-python)
> - [Mathematical explanation of K-Nearest Neighbour](https://www.geeksforgeeks.org/mathematical-explanation-of-k-nearest-neighbour)
> - [Weighted K-NN](https://www.geeksforgeeks.org/weighted-k-nn)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

[Classifying data using Support Vector Machines(SVMs) in Python](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

57

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[k-nearest neighbor algorithm in Python\\
\\
\\
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning method. It operates for classification as well as regression: Classification: For a new data point, the algorithm identifies its nearest neighbors based on a distance metric (e.g., Euclidean distance). The predicted class is dete\\
\\
4 min read](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/)
[kNN: k-Nearest Neighbour Algorithm in R From Scratch\\
\\
\\
In this article, we are going to discuss what is KNN algorithm, how it is coded in R Programming Language, its application, advantages and disadvantages of the KNN algorithm. kNN algorithm in RKNN can be defined as a K-nearest neighbor algorithm. It is a supervised learning algorithm that can be use\\
\\
15+ min read](https://www.geeksforgeeks.org/knn-k-nearest-neighbour-algorithm-in-r-from-scratch/)
[r-Nearest neighbors\\
\\
\\
r-Nearest neighbors are a modified version of the k-nearest neighbors. The issue with k-nearest neighbors is the choice of k. With a smaller k, the classifier would be more sensitive to outliers. If the value of k is large, then the classifier would be including many points from other classes. It is\\
\\
5 min read](https://www.geeksforgeeks.org/r-nearest-neighbors/)
[ML \| K-means++ Algorithm\\
\\
\\
Clustering is one of the most common tasks in machine learning where we group similar data points together. K-Means Clustering is one of the simplest and most popular clustering algorithms but it has one major drawback â€” the random initialization of cluster centers often leads to poor clustering res\\
\\
5 min read](https://www.geeksforgeeks.org/ml-k-means-algorithm/)
[Implementation of K Nearest Neighbors\\
\\
\\
Prerequisite: K nearest neighbors Introduction Say we are given a data set of items, each having numerically valued features (like Height, Weight, Age, etc). If the count of features is n, we can represent the items as points in an n-dimensional grid. Given a new item, we can calculate the distance\\
\\
10 min read](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)
[K Nearest Neighbors with Python \| ML\\
\\
\\
K-Nearest Neighbors is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection. The K-Nearest Neighbors (KNN) algorithm is a simple, easy\\
\\
5 min read](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/)
[K-Nearest Neighbors and Curse of Dimensionality\\
\\
\\
In high-dimensional data, the performance of the k-nearest neighbor (k-NN) algorithm often deteriorates due to increased computational complexity and the breakdown of the assumption that similar points are proximate. These challenges hinder the algorithm's accuracy and efficiency in high-dimensional\\
\\
6 min read](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/)
[Basic Understanding of CURE Algorithm\\
\\
\\
CURE(Clustering Using Representatives) It is a hierarchical based clustering technique, that adopts a middle ground between the centroid based and the all-point extremes. Hierarchical clustering is a type of clustering, that starts with a single point cluster, and moves to merge with another cluster\\
\\
2 min read](https://www.geeksforgeeks.org/basic-understanding-of-cure-algorithm/)
[Understanding Decision Boundaries in K-Nearest Neighbors (KNN)\\
\\
\\
Decision boundary is an imaginary line or surface that separates different classes in a classification problem. It represents regions as one class versus another based on model assigns them. K-Nearest Neighbors (KNN) algorithm operates on the principle that similar data points exist in close proximi\\
\\
5 min read](https://www.geeksforgeeks.org/understanding-decision-boundaries-in-k-nearest-neighbors-knn/)
[Mathematical explanation of K-Nearest Neighbour\\
\\
\\
KNN stands for K-nearest neighbour is a popular algorithm in Supervised Learning commonly used for classification tasks. It works by classifying data based on its similarity to neighboring data points. The core idea of KNN is straightforward when a new data point is introduced the algorithm finds it\\
\\
4 min read](https://www.geeksforgeeks.org/mathematical-explanation-of-k-nearest-neighbour/)

Like57

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/k-nearest-neighbours/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=768892203.1745055906&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102015666~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1076887063)