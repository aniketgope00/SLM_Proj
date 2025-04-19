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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/understanding-decision-boundaries-in-k-nearest-neighbors-knn/?type%3Darticle%26id%3D1322662&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
100 Deep Learning Terms Explained\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/100-deep-learning-terms-explained/)

# Understanding Decision Boundaries in K-Nearest Neighbors (KNN)

Last Updated : 30 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Decision boundary is an imaginary line or surface that separates different classes in a classification problem. It represents regions as one class versus another based on model assigns them. [K-Nearest Neighbors (KNN) algorithm](https://www.geeksforgeeks.org/k-nearest-neighbours/) operates on the principle that similar data points exist in close proximity within a feature space.

## Starting with decision boundaries

For algorithms like KNN - decision boundaries are directly influenced by the number of neighbors “K” and the spatial distribution of data points in training set. For example given a dataset with two classes the decision boundary can be visualized as the line or curve dividing the two regions where each class is predicted. _**For a 1-nearest neighbor (1-NN) classifier the decision boundary can be visualized using a Voronoi diagram**_.

The boundaries are equidistant lines between data points of different classes. These diagrams help in understanding how KNN adapts to data density and forms complex decision boundaries naturally.

### Mathematical Formation of Decision Boundaries

Decision boundary in KNN is where the classification decision changes. **Mathematically the boundary is defined as the set of points for which the decision criterion (majority vote) is exactly at the border between different classes.**

- The decision boundary occurs at points where the distances to the k-nearest neighbors result in an equal number of neighbors from different classes. For example in a binary classification with k=3, the boundary might be where the nearest three neighbors include two points from class A and one point from class B or vice versa.
- A [**Voronoi diagram**](https://www.geeksforgeeks.org/voronoi-diagram/) helps visualize how KNN forms decision boundaries. It divides a plane into regions based on distance to specific points (called seed points). Each region is known as a **Voronoi cell and** contains all points that are closer to its corresponding seed point than to any other seed point. This visualization makes it easier to understand how KNN assigns classifications.
- The boundaries between Voronoi cells are equidistant lines between pairs of seed points. If pip\_ipi​ and pjp\_jpj​ are two seed points, the boundary between their Voronoi cells is the perpendicular bisector of the line segment connecting piandpjp\_i \\space and \\space p\_jpi​andpj​.

![knn-decision-boundafries](https://media.geeksforgeeks.org/wp-content/uploads/20240910123947/knn-decision-boundafries.webp)Formation of Decision Boundaries

### Relationship Between KNN Decision Boundaries and Voronoi Diagrams

In two-dimensional space the decision boundaries of KNN can be visualized as Voronoi diagrams. Here’s how:

- **KNN Boundaries:** The decision boundary for KNN is determined by regions where the classification changes based on the nearest neighbors. As k approaches infinity, these boundaries approach the Voronoi diagram boundaries.
- **Voronoi Diagram as a Special Case:** When k=1 KNN’s decision boundaries directly correspond to the Voronoi diagram of the training points. Each region in the Voronoi diagram represents the area where the nearest training point is closest.

## How KNN Defines Decision Boundaries

In KNN, decision boundaries are influenced by the choice of k and the distance metric used:

**1\. Impact of 'K' on Decision Boundaries**: The number of neighbors (k) affects the shape and smoothness of the decision boundary.

- **Small k:** When k is small, the decision boundary can become very complex, closely following the training data. This can lead to overfitting.
- **Large k:** When k is large, the decision boundary smooths out and becomes less sensitive to individual data points, potentially leading to underfitting.

**2\. Distance Metric**: The decision boundary is also affected by the distance metric used (e.g Euclidean, Manhattan). Different metrics can lead to different boundary shapes.

- **Euclidean Distance:** Commonly used leading to circular or elliptical decision boundaries in two-dimensional space.
- **Manhattan Distance:** Results in axis-aligned decision boundaries.

## Exploring KNN Decision Boundaries with Case Studies

Visualizing decision boundaries helps in understanding how a KNN model classifies data. For a two-dimensional dataset decision boundaries can be plotted by:

- **Creating a Grid**: Generate a grid of points covering the feature space.
- **Classifying Grid Points:** Use the KNN algorithm to classify each point in the grid based on its neighbors.
- **Plotting:** Color the grid points according to their class labels and draw the boundaries where the class changes.

### Case Study of Binary Classification with Varying k

Consider a [binary classification](https://www.geeksforgeeks.org/getting-started-with-classification/) problem with two features where the goal is to visualize how the K-Nearest Neighbors (KNN) decision boundary changes as k varies. This example uses synthetic data to illustrate the impact of different k values on the decision boundary.

Python`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
k_values = [1, 3, 5, 10]
for ax, k in zip(axs.flat, k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k',
               cmap=plt.cm.Paired, marker='o')
    ax.set_title(f'KNN Decision Boundaries (k={k})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
plt.tight_layout()
plt.show()
`

**Output:**

![casestudy1](https://media.geeksforgeeks.org/wp-content/uploads/20240910125132/casestudy1.webp)Binary Classification with Varying k

- For small k the boundary is highly sensitive to local variations and can be irregular.
- For larger k the boundary smooths out, reflecting a more generalized view of the data distribution.

## Factors Affecting Decision Boundaries

Several factors can influence the shape and quality of decision boundaries in KNN:

- **Feature Scaling:** KNN is sensitive to scale of features.
- **Noise in Data:** Outliers and noisy data points can distort the decision boundary.
- D **ata Distribution:** The distribution of training data impacts the decision boundary.

The shape and complexity of the decision boundary impact the model’s performance. A well-defined boundary that accurately separates classes can lead to high classification accuracy, while a poorly defined boundary can result in misclassifications.

The decision boundaries in KNN are a reflection of the algorithm's adaptability to data density and its sensitivity to the choice of 'k'. Understanding these boundaries helps in optimizing KNN's performance for specific datasets. By carefully choosing 'k' and preprocessing data, KNN can be used to wide range of application like pattern recognition to recommendation systems

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/100-deep-learning-terms-explained/)

[100 Deep Learning Terms Explained](https://www.geeksforgeeks.org/100-deep-learning-terms-explained/)

[F](https://www.geeksforgeeks.org/user/frisbevhwy/)

[frisbevhwy](https://www.geeksforgeeks.org/user/frisbevhwy/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [ML-Classification](https://www.geeksforgeeks.org/tag/ml-classification/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[kNN: k-Nearest Neighbour Algorithm in R From Scratch\\
\\
\\
In this article, we are going to discuss what is KNN algorithm, how it is coded in R Programming Language, its application, advantages and disadvantages of the KNN algorithm. kNN algorithm in RKNN can be defined as a K-nearest neighbor algorithm. It is a supervised learning algorithm that can be use\\
\\
15+ min read](https://www.geeksforgeeks.org/knn-k-nearest-neighbour-algorithm-in-r-from-scratch/)
[K-Nearest Neighbors and Curse of Dimensionality\\
\\
\\
In high-dimensional data, the performance of the k-nearest neighbor (k-NN) algorithm often deteriorates due to increased computational complexity and the breakdown of the assumption that similar points are proximate. These challenges hinder the algorithm's accuracy and efficiency in high-dimensional\\
\\
6 min read](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/)
[How To Predict Diabetes using K-Nearest Neighbor in R\\
\\
\\
In this article, we are going to predict Diabetes using the K-Nearest Neighbour algorithm and analyze on Diabetes dataset using the R Programming Language. What is the K-Nearest Neighbor algorithm?The K-Nearest Neighbor (KNN) algorithm is a popular supervised learning classifier frequently used by d\\
\\
13 min read](https://www.geeksforgeeks.org/how-to-predict-diabetes-using-k-nearest-neighbor-in-r/)
[How to Draw Decision Boundaries in R\\
\\
\\
Decision boundaries are essential concepts in machine learning, especially for classification tasks. They define the regions in feature space where the model predicts different classes. Visualizing decision boundaries helps us understand how a classifier separates different classes. In this article,\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-draw-decision-boundaries-in-r/)
[K-Nearest Neighbor(KNN) Algorithm\\
\\
\\
K-Nearest Neighbors (KNN) is a simple way to classify things by looking at whatâ€™s nearby. Imagine a streaming service wants to predict if a new user is likely to cancel their subscription (churn) based on their age. They checks the ages of its existing users and whether they churned or stayed. If mo\\
\\
10 min read](https://www.geeksforgeeks.org/k-nearest-neighbours/)
[k-nearest neighbor algorithm in Python\\
\\
\\
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning method. It operates for classification as well as regression: Classification: For a new data point, the algorithm identifies its nearest neighbors based on a distance metric (e.g., Euclidean distance). The predicted class is dete\\
\\
4 min read](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/)
[Finding k-nearest Neighbor for Only One Point Using R\\
\\
\\
The k-nearest neighbors (k-NN) algorithm is a simple yet powerful tool used in various machine learning and data mining applications. While k-NN is often applied to an entire dataset to classify or predict values for multiple points, there are scenarios where you may need to find the k-nearest neigh\\
\\
3 min read](https://www.geeksforgeeks.org/finding-k-nearest-neighbor-for-only-one-point-using-r/)
[Logistic Regression vs K Nearest Neighbors in Machine Learning\\
\\
\\
Machine learning algorithms play a crucial role in training the data and decision-making processes. Logistic Regression and K Nearest Neighbors (KNN) are two popular algorithms in machine learning used for classification tasks. In this article, we'll delve into the concepts of Logistic Regression an\\
\\
4 min read](https://www.geeksforgeeks.org/logistic-regression-vs-k-nearest-neighbors-in-machine-learning/)
[Implementation of K Nearest Neighbors\\
\\
\\
Prerequisite: K nearest neighbors Introduction Say we are given a data set of items, each having numerically valued features (like Height, Weight, Age, etc). If the count of features is n, we can represent the items as points in an n-dimensional grid. Given a new item, we can calculate the distance\\
\\
10 min read](https://www.geeksforgeeks.org/implementation-k-nearest-neighbors/)
[Implementation of K-Nearest Neighbors from Scratch using Python\\
\\
\\
Instance-Based LearningK Nearest Neighbors Classification is one of the classification techniques based on instance-based learning. Models based on instance-based learning to generalize beyond the training examples. To do so, they store the training examples first. When it encounters a new instance\\
\\
8 min read](https://www.geeksforgeeks.org/implementation-of-k-nearest-neighbors-from-scratch-using-python/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/understanding-decision-boundaries-in-k-nearest-neighbors-knn/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1165159106.1745056364&gtm=45je54g3h1v884918195za200zb858768136&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=1251986335)