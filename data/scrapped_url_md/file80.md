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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/?type%3Darticle%26id%3D311093&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Implementing DBSCAN algorithm using Sklearn\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/)

# Elbow Method for optimal value of k in KMeans

Last Updated : 02 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Choosing the optimal number of clusters is a crucial step in any unsupervised learning algorithm. Since we don’t have predefined cluster counts in unsupervised learning, we need a systematic approach to determine the best k value. The **Elbow Method** is a popular technique used for this purpose in K-Means clustering.

In this article, we will explore how to select the best number of clusters (k) when using the [K-Means clustering algorithm.](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

## Elbow Method in K-Means Clustering

In [K-Means clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/), we start by randomly initializing k clusters and iteratively adjusting these clusters until they stabilize at an equilibrium point. However, before we can do this, we need to decide how many clusters (k) we should use.

The Elbow Method helps us find this optimal k value. Here’s how it works:

1. We iterate over a range of k values, typically from 1 to n (where n is a hyper-parameter you choose).
2. For each k, we calculate the **Within-Cluster Sum of Squares (WCSS)**.

> WCSS measures **how well the data points are clustered around their respective centroids**. It is defined as the **sum of the squared distances between each point and its cluster centroid:**
>
> WCSS=∑i=1k∑j=1nidistance(xj(i),ci)2\\text{WCSS} = \\sum\_{i=1}^{k} \\sum\_{j=1}^{n\_i} \\text{distance}(x\_j^{(i)}, c\_i)^2WCSS=∑i=1k​∑j=1ni​​distance(xj(i)​,ci​)2
>
> where,
>
> distance(xj(i),ci)\\text{distance}(x\_j^{(i)}, c\_i) distance(xj(i)​,ci​)represents the distance between the j-th data point xj(i)x\_j^{(i)}xj(i)​​ in cluster i and the centroid cic\_i ci​of that cluster.

### The Elbow Point: Optimal k Value

The Elbow Method works in below steps:

- **We calculate a distance measure called WCSS (Within-Cluster Sum of Squares).** This tells us how spread out the data points are within each cluster.
- **We try different k values (number of clusters).** For each k, we run KMeans and calculate the WCSS.
- **We plot a graph with k on the X-axis and WCSS on the Y-axis.**
- **Identifying the Elbow Point**: As we increase kkk, the WCSS typically decreases because we’re creating more clusters, which tend to capture more data variations. However, there comes a point where adding more clusters results in only a marginal decrease in WCSS. This is where we observe an “elbow” shape in the graph.
  - **Before the elbow**: Increasing kkk significantly reduces WCSS, indicating that new clusters effectively capture more of the data’s variability.
  - **After the elbow**: Adding more clusters results in a minimal reduction in WCSS, suggesting that these extra clusters may not be necessary and could lead to overfitting.

![Elbow-Method](https://media.geeksforgeeks.org/wp-content/uploads/20241028173908396970/Elbow-Method.png)

Elbow Point

The goal is to identify the point where the rate of decrease in WCSS sharply changes, indicating that adding more clusters (beyond this point) yields diminishing returns. This “elbow” point suggests the optimal number of clusters.

> There are many more techniques to find optimal value of k and for that please refer to this article:
>
> [Determine the optimal value of K in K-Means Clustering](https://www.geeksforgeeks.org/ml-determine-the-optimal-value-of-k-in-k-means-clustering/)

## Understanding Distortion and Inertia in K-Means Clustering

In K-Means clustering, we aim to group similar data points together. To evaluate the quality of these groupings, we use two key metrics: Distortion and Inertia.

### **1\. Distortion**

Distortion measures the average squared distance between each data point and its assigned cluster center. It’s a measure of how well the clusters represent the data. A lower distortion value indicates better clustering.

Distortion=1n∑i=1nmin⁡c∈clusters∥xi–c∥2\\text{Distortion} = \\frac{1}{n} \\sum\_{i=1}^{n} \\min\_{c \\in \\text{clusters}} \\left\\\| x\_i – c \\right\\\|^2
Distortion=n1​∑i=1n​minc∈clusters​∥xi​–c∥2

where,

- xix\_ixi​​ is the ithi^{th}ith data point
- ccc is a cluster center from the set of all cluster centroids
- ∥xi–c∥2\\left\\\| x\_i – c \\right\\\|^2∥xi​–c∥2 is the **squared Euclidean distance** between the data point and the cluster center
- nnn is the total number of data points

### **2\. Inertia**

Inertia is the sum of squared distances of each data point to its closest cluster center. It’s essentially the total squared error of the clustering. Like distortion, a lower inertia value suggests better clustering.

Inertia=∑i=1ndistance(xi,cj∗)2\\text{Inertia} = \\sum\_{i=1}^{n} \\text{distance}(x\_i, c\_j^\*)^2Inertia=∑i=1n​distance(xi​,cj∗​)2

> **Inertia** is the numerator of the **Distortion** formula, **Distortion** is the average inertia per data point.

In the Elbow Method, we calculate the distortion or inertia for different values of k (number of clusters). We then plot these values to identify the “elbow point”, where the rate of decrease in distortion or inertia starts to slow down. This elbow point often indicates the optimal number of clusters.

### A Lower Distortion or Inertia is Generally Better

_**A lower distortion or inertia implies that the data points are more closely grouped around their respective cluster centers**_. However, it’s important to balance this with the number of clusters. Too few clusters might not capture the underlying structure of the data, while too many clusters can lead to overfitting.

By understanding distortion and inertia, we can effectively evaluate the quality of K-Means clustering and select the optimal number of clusters.

## Implementation of Elbow Method Using in Python

In this section, we will demonstrate how to implement the Elbow Method to determine the optimal number of clusters (k) using Python’s Scikit-learn library. We will create a random dataset, apply K-means clustering, calculate the Within-Cluster Sum of Squares (WCSS) for different values of k, and visualize the results to determine the optimal number of clusters.

### **Step 1: Importing the required libraries**

Python`
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
`

### **Step 2: Creating and Visualizing the data**

We will create a random array and visualize its distribution

Python`
# Creating the dataset
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6,\
               7, 8, 9, 8, 9, 9, 8, 4, 4, 5, 4])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7,\
               1, 2, 1, 2, 3, 2, 3, 9, 10, 9, 10])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
# Visualizing the data
plt.scatter(x1, x2, marker='o')
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
`

**Output:**

![Visualizing the data using matplotlib library](https://media.geeksforgeeks.org/wp-content/uploads/20230418183303/download-(6).png)

Visualizing the data using the matplotlib library

From the above visualization, we can see that the optimal number of clusters should be around 3. But visualizing the data alone cannot always give the right answer. Hence we demonstrate the following steps.

### Step 3: Building the Clustering Model and Calculating Distortion and Inertia

In this step, we will fit the K-means model for different values of k (number of clusters) and calculate both the distortion and inertia for each value.

Python`
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / X.shape[0])

    inertias.append(kmeanModel.inertia_)

    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]
`

### Step 4: Tabulating and Visualizing the Results

**a) Displaying Distortion Values**

Python`
print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
`

**Output:**

> Distortion values:
>
> 1 : 14.90249433106576
>
> 2 : 5.146258503401359
>
> 3 : 1.8817838246409675
>
> 4 : 0.856122448979592
>
> 5 : 0.7166666666666667
>
> 6 : 0.5484126984126984
>
> 7 : 0.4325396825396825
>
> 8 : 0.3817460317460318
>
> 9 : 0.3341269841269841

![distortion](https://media.geeksforgeeks.org/wp-content/uploads/20241028180149199900/distortion.png)

Plotting Distortion Values

**b) Displaying Inertia Values:**

Python`
print("Inertia values:")
for key, val in mapping2.items():
    print(f'{key} : {val}')
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()
`

**Output:**

> Inertia values:
>
> 1 : 312.95238095238096
>
> 2 : 108.07142857142854
>
> 3 : 39.51746031746032
>
> 4 : 17.978571428571428
>
> 5 : 15.049999999999997
>
> 6 : 11.516666666666666
>
> 7 : 9.083333333333334
>
> 8 : 8.016666666666667
>
> 9 : 7.0166666666666675

![intertia](https://media.geeksforgeeks.org/wp-content/uploads/20241028180319403957/intertia.png)

Inertia Values

### Step 5: Clustered Data Points For Different k Values

We will plot images of data points clustered for different values of k. For this, we will apply the k-means algorithm on the dataset by iterating on a range of k values.

Python`
k_range = range(1, 5)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='red', label='Centroids', edgecolor='k')
    plt.title(f'K-means Clustering (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.show()
`

**Output:**

![Visualizing-Clustered-Data-Points](https://media.geeksforgeeks.org/wp-content/uploads/20241028180742360657/Visualizing-Clustered-Data-Points.png)

Visualizing Clustered Data Points

## **Key Takeaways**

- The Elbow Method helps you choose the optimal number of clusters (k) in KMeans clustering.
- It analyzes how adding more clusters (increasing k) affects the spread of data points within each cluster (WCSS).
- The k value corresponding to the “elbow” in the WCSS vs k graph is considered the optimal choice.
- The Elbow Method provides a good starting point, but consider your specific data and goals when finalizing k.

> You can download the source code from here: [Source Code](https://media.geeksforgeeks.org/wp-content/uploads/20250402153631238117/Elbow_Method_for_optimal_value_of_k_in_KMeans.ipynb)

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Elbow Method for optimal value of k in KMeans

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/)

[Implementing DBSCAN algorithm using Sklearn](https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/)

[A](https://www.geeksforgeeks.org/user/AlindGupta/)

[AlindGupta](https://www.geeksforgeeks.org/user/AlindGupta/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-Clustering](https://www.geeksforgeeks.org/tag/ml-clustering/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[ML \| Determine the optimal value of K in K-Means Clustering\\
\\
\\
Determining optimal value of K in k means clustering is a hectic task as a optimal value can help us to find better data pattern and model prediction. Choosing is value manually is very difficult so we use various techniques to find its value. In this article we will discuss about these techniques.\\
\\
5 min read](https://www.geeksforgeeks.org/ml-determine-the-optimal-value-of-k-in-k-means-clustering/)
[How to Find The Optimal Value of K in KNN\\
\\
\\
In K-Nearest Neighbors (KNN) algorithm one of the key decision that directly impacts performance of the model is choosing the optimal value of K. It represents number of nearest neighbors to be considered while classifying a data point. If K is too small or too large it can lead to overfitting or un\\
\\
6 min read](https://www.geeksforgeeks.org/how-to-find-the-optimal-value-of-k-in-knn/)
[How to Change the Value of k in KNN Using R?\\
\\
\\
The k-Nearest Neighbors (KNN) algorithm is a simple, yet powerful, non-parametric method used for classification and regression. One of the critical parameters in KNN is the value of k, which represents the number of nearest neighbors to consider when making a prediction. In this article, we'll expl\\
\\
5 min read](https://www.geeksforgeeks.org/how-to-change-the-value-of-k-in-knn-using-r/)
[Gap statistics for optimal number of cluster\\
\\
\\
To get the optimal number of clusters in a dataset, we use Gap Statistics. It compares the performance of clustering algorithms against a null reference distribution of the data, allowing for a more objective decision on the number of clusters. Letâ€™s explore Gap Statistics in more detail and discove\\
\\
7 min read](https://www.geeksforgeeks.org/gap-statistics-for-optimal-number-of-cluster/)
[How to Visualize KNN in Python\\
\\
\\
Visualizing the K-Nearest Neighbors (KNN) algorithm in Python is a great way to understand how this supervised learning method works and how it makes predictions. In essence, visualizing KNN involves plotting the decision boundaries that the algorithm creates based on the number of nearest neighbors\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-visualize-knn-in-python/)
[Silhouette Algorithm to determine the optimal value of k\\
\\
\\
One of the fundamental steps of an unsupervised learning algorithm is to determine the number of clusters into which the data may be divided. The silhouette algorithm is one of the many algorithms to determine the optimal number of clusters for an unsupervised learning technique. In the Silhouette a\\
\\
3 min read](https://www.geeksforgeeks.org/silhouette-algorithm-to-determine-the-optimal-value-of-k/)
[K-Means vs K-Means++ Clustering Algorithm\\
\\
\\
Clustering is a fundamental technique in unsupervised learning, widely used for grouping data into clusters based on similarity. Among the clustering algorithms, K-Means and its improved version, K-Means++, are popular choices. This article explores how both algorithms work, their advantages and lim\\
\\
6 min read](https://www.geeksforgeeks.org/k-means-vs-k-means-clustering-algorithm/)
[How do k-means clustering methods differ from k-nearest neighbor methods\\
\\
\\
K-Means is an unsupervised learningmethod used for clustering, while KNN is a supervised learning algorithm used for classification (or regression). K-Means clusters data into groups, and the centroids represent the center of each group. KNN creates decision boundaries based on labeled training data\\
\\
3 min read](https://www.geeksforgeeks.org/how-do-k-means-clustering-methods-differ-from-k-nearest-neighbor-methods/)
[What Does cl Parameter in knn Function in R Mean?\\
\\
\\
The knn function in R is a powerful tool for implementing the k-Nearest Neighbors (k-NN) algorithm, a simple and intuitive method for classification and regression tasks. The function is part of the class package, which provides functions for classification. Among its various parameters, the cl para\\
\\
4 min read](https://www.geeksforgeeks.org/what-does-cl-parameter-in-knn-function-in-r-mean/)
[ML \| K-means++ Algorithm\\
\\
\\
Clustering is one of the most common tasks in machine learning where we group similar data points together. K-Means Clustering is one of the simplest and most popular clustering algorithms but it has one major drawback â€” the random initialization of cluster centers often leads to poor clustering res\\
\\
5 min read](https://www.geeksforgeeks.org/ml-k-means-algorithm/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1837950505.1745056596&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=855582453)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056595928&cv=11&fst=1745056595928&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Felbow-method-for-optimal-value-of-k-in-kmeans%2F&hn=www.googleadservices.com&frm=0&tiba=Elbow%20Method%20for%20optimal%20value%20of%20k%20in%20KMeans%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=245800113.1745056596&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)