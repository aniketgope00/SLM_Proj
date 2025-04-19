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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/principal-component-analysis-pca/?type%3Darticle%26id%3D210343&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
ML \| T-distributed Stochastic Neighbor Embedding (t-SNE) Algorithm\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/)

# Principal Component Analysis(PCA)

Last Updated : 03 Feb, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Having too many features in data can cause problems like overfitting (good on training data but poor on new data), slower computation, and lower accuracy. This is called the [curse of dimensionality](https://www.geeksforgeeks.org/videos/curse-of-dimensionality-in-machine-learning/), where more features exponentially increase the data needed for reliable results.

The explosion of feature combinations makes sampling harder In high-dimensional data and tasks like **clustering or classification more complex and slow.**

> To tackle this problem, we use Feature engineering Techniques ,such as **feature selection** (choosing the most important features) and **feature extraction** (creating new features from the original ones). One popular feature extraction method is [**dimensionality reduction**](https://www.geeksforgeeks.org/dimensionality-reduction/), which reduces the number of features while keeping as much important information as possible.

One of the most widely used dimensionality reduction techniques is **Principal Component Analysis (PCA)**.

## How PCA Works for Dimensionality Reduction?

PCA is a statistical technique introduced by mathematician Karl Pearson in 1901. _**It works by transforming high-dimensional data into a lower-dimensional space while maximizing the variance (or spread) of the data in the new space**_. This helps preserve the most important patterns and relationships in the data.

> _**Note: It prioritizes the directions where the data varies the most (because more variation = more useful information.**_

Let’s understand it’s working in simple terms:

Imagine you’re looking at a messy cloud of data points (like stars in the sky) and want to simplify it. PCA helps you find the “most important angles” to view this cloud so you don’t miss the big patterns. Here’s how it works, step by step:

### **Step 1: Standardize the Data**

Make sure all features (e.g., height, weight, age) are on the **same scale**. Why? A feature like “salary” (ranging 0–100,000) could dominate “age” (0–100) otherwise.

[Standardizing](https://www.geeksforgeeks.org/normalization-vs-standardization/) our dataset to ensures that each variable has a mean of 0 and a standard deviation of 1.

Z=X−μσZ = \\frac{X-\\mu}{\\sigma}Z=σX−μ​

Here,

- μ\\mu






μ is the mean of independent features  μ={μ1,μ2,⋯,μm}\\mu = \\left \\{ \\mu\_1, \\mu\_2, \\cdots, \\mu\_m \\right \\}






μ={μ1​,μ2​,⋯,μm​}
- σ\\sigma




σ is the [standard deviation](https://www.geeksforgeeks.org/mathematics-mean-variance-and-standard-deviation/) of independent features  σ={σ1,σ2,⋯,σm}\\sigma = \\left \\{ \\sigma\_1, \\sigma\_2, \\cdots, \\sigma\_m \\right \\}




σ={σ1​,σ2​,⋯,σm​}

### **Step 2: Find Relationships**

Calculate how features **move together** using a _covariance matrix_. [Covariance](https://www.geeksforgeeks.org/mathematics-covariance-and-correlation/) measures the strength of joint variability between two or more variables, indicating how much they change in relation to each other. To find the covariance we can use the formula:

cov(x1,x2)=∑i=1n(x1i−x1ˉ)(x2i−x2ˉ)n−1cov(x1,x2) = \\frac{\\sum\_{i=1}^{n}(x1\_i-\\bar{x1})(x2\_i-\\bar{x2})}{n-1}cov(x1,x2)=n−1∑i=1n​(x1i​−x1ˉ)(x2i​−x2ˉ)​

The value of covariance can be positive, negative, or zeros.

- **Positive:** As the x1 increases x2 also increases.
- **Negative:** As the x1 increases x2 also decreases.
- **Zeros:** No direct relation.

### **Step 3: Find the “Magic Directions” (Principal Components)**

- PCA identifies **new axes** (like rotating a camera) where the data spreads out the most:
  - **1st Principal Component (PC1):** The direction of maximum variance (most spread).
  - **2nd Principal Component (PC2):** The next best direction, _perpendicular to PC1_, and so on.
- These directions are calculated using [Eigenvalues and Eigenvectors](https://www.geeksforgeeks.org/applications-of-eigenvalues-and-eigenvectors/#:~:text=Eigenvalues%20and%20eigenvectors%20are%20mathematical,scaled%20by%20its%20corresponding%20eigenvalue.) **where: eigenvectors** (math tools that find these axes), and their importance is ranked by **eigenvalues** (how much variance each captures).

For a square matrix A, an **eigenvector** X (a non-zero vector) and its corresponding **eigenvalue** λ (a scalar) satisfy:

AX=λXAX = \\lambda XAX=λX

This means:

- When _A_ acts on X, it only stretches or shrinks X by the scalar λ.
- The direction of X remains unchanged (hence, eigenvectors define “stable directions” of A).

It can also be written as :

AX−λX=0(A−λI)X=0\\begin{aligned} AX-\\lambda X &= 0 \\\ (A-\\lambda I)X &= 0 \\end{aligned}AX−λX(A−λI)X​=0=0​

where I is the identity matrix of the same shape as matrix A. And the above conditions will be true only if (A–λI)(A – \\lambda I)(A–λI) will be non-invertible (i.e. singular matrix). That means,

∣A–λI∣=0\|A – \\lambda I\| = 0∣A–λI∣=0

This determinant equation is called the **characteristic equation**.

- Solving it gives the eigenvalues \\lambda,
- and therefore corresponding eigenvector can be found using the equation AX=λXAX = \\lambda XAX=λX.

> **How This Connects to PCA**?
>
> - In PCA, the covariance matrix C (from Step 2) acts as matrix A.
> - Eigenvectors of _C_ are the **principal components** (PCs).
> - Eigenvalues represent the **variance** captured by each PC.

### **Step 4: Pick the Top Directions & Transform Data**

- Keep only the top 2–3 directions (or enough to capture ~95% of the variance).
- Project the data onto these directions to get a simplified, lower-dimensional version.

PCA is an [unsupervised learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/) **algorithm**, meaning it doesn’t require prior knowledge of target variables. It’s commonly used in exploratory data analysis and machine learning to **simplify datasets without losing critical information.**

> **We know everything sound complicated, let’s understand again with help of visual image where,** **x-axis (Radius)** and **y-axis (Area)** represent two original features in the dataset.

![Principal Component Analysis - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230420165431/Principal-Componenent-Analysisi.webp)

Transform this 2D dataset into a 1D representation while preserving as much variance as possible.

**Principal Components (PCs):**

- **PC₁ (First Principal Component):** The direction along which the data has the maximum variance. It captures the most important information.
- **PC₂ (Second Principal Component):** The direction orthogonal (perpendicular) to PC₁. It captures the remaining variance but is less significant.

Now, The **red dashed lines** indicate the spread (variance) of data along different directions . The variance along **PC₁ is greater than PC₂**, which means that PC₁ carries more useful information about the dataset.

- The data points (blue dots) are projected onto PC₁, effectively reducing the dataset from two dimensions (Radius, Area) to one dimension (PC₁).
- This transformation simplifies the dataset while retaining most of the original variability.

> The image visually explains why **PCA selects the direction with the highest variance** (PC₁). By removing PC₂, we reduce redundancy while keeping essential information. The transformation helps in **data compression, visualization, and improved model performance**.

## Principal Component Analysis Implementation in Python

Hence, PCA employs a linear transformation that is based on preserving the most variance in the data using the least number of dimensions. It involves the following steps:

Python`
import pandas as pd
import numpy as np
# Here we are using inbuilt dataset of scikit learn
from sklearn.datasets import load_breast_cancer
# instantiating
cancer = load_breast_cancer(as_frame=True)
# creating dataframe
df = cancer.frame
# checking shape
print('Original Dataframe shape :',df.shape)
# Input features
X = df[cancer['feature_names']]
print('Inputs Dataframe shape   :', X.shape)
`

**Output**:

```
Original Dataframe shape : (569, 31)
Inputs Dataframe shape   : (569, 30)
```

Now we will apply the first most step **which is to standardize the data and for that, we will have to first calculate the mean and standard deviation of each feature in the feature space.**

Python`
# Mean
X_mean = X.mean()
# Standard deviation
X_std = X.std()
# Standardization
Z = (X - X_mean) / X_std
`

The [covariance matrix](https://www.geeksforgeeks.org/covariance-matrix/) helps us visualize how strong the dependency of two features is with each other in the feature space.

Python`
# covariance
c = Z.cov()
# Plot the covariance matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(c)
plt.show()
`

**Output**:

![Covariance Matrix (PCA)-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230726123116/download-(2)-(1).webp)

Now we will compute the [eigenvectors](https://www.geeksforgeeks.org/eigen-values/) and [eigenvalues](https://www.geeksforgeeks.org/eigen-values/) for our feature space which serve a great purpose in identifying the principal components for our feature space.

Python`
eigenvalues, eigenvectors = np.linalg.eig(c)
print('Eigen values:\n', eigenvalues)
print('Eigen values Shape:', eigenvalues.shape)
print('Eigen Vector Shape:', eigenvectors.shape)
`

**Output**:

```
Eigen values:
 [1.32816077e+01 5.69135461e+00 2.81794898e+00 1.98064047e+00\
 1.64873055e+00 1.20735661e+00 6.75220114e-01 4.76617140e-01\
 4.16894812e-01 3.50693457e-01 2.93915696e-01 2.61161370e-01\
 2.41357496e-01 1.57009724e-01 9.41349650e-02 7.98628010e-02\
 5.93990378e-02 5.26187835e-02 4.94775918e-02 1.33044823e-04\
 7.48803097e-04 1.58933787e-03 6.90046388e-03 8.17763986e-03\
 1.54812714e-02 1.80550070e-02 2.43408378e-02 2.74394025e-02\
 3.11594025e-02 2.99728939e-02]
Eigen values Shape: (30,)
Eigen Vector Shape: (30, 30)

```

Sort the eigenvalues in descending order and sort the corresponding eigenvectors accordingly.

Python`
# Index the eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
# Sort the eigenvalues in descending order
eigenvalues = eigenvalues[idx]
# sort the corresponding eigenvectors accordingly
eigenvectors = eigenvectors[:,idx]
`

Explained variance is the term that gives us an idea of the amount of the total variance which has been retained by selecting the principal components instead of the original feature space.

Python`
explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
explained_var
`

**Output**:

```
array([0.44272026, 0.63243208, 0.72636371, 0.79238506, 0.84734274,\
       0.88758796, 0.9100953 , 0.92598254, 0.93987903, 0.95156881,\
       0.961366  , 0.97007138, 0.97811663, 0.98335029, 0.98648812,\
       0.98915022, 0.99113018, 0.99288414, 0.9945334 , 0.99557204,\
       0.99657114, 0.99748579, 0.99829715, 0.99889898, 0.99941502,\
       0.99968761, 0.99991763, 0.99997061, 0.99999557, 1.        ])

```

**Determine the Number of Principal Components**

Here we can either consider the number of principal components of any value of our choice or by limiting the explained variance. Here I am considering explained variance more than equal to 50%. Let’s check how many principal components come into this.

Python`
n_components = np.argmax(explained_var >= 0.50) + 1
n_components
`

**Output**:

```
2

```

**Project the Data onto the Selected Principal Components**

- Instead of storing full **(x, y)** coordinates, PCA **stores only the projection values** along the principal component, simplifying data processing.
- Projection matrix: is a matrix of **eigenvectors corresponding to the largest eigenvalues of the covariance matrix of the data**. it projects the high-dimensional dataset onto a lower-dimensional subspace.

Python`
# PCA component or unit matrix
u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u,
                             index = cancer['feature_names'],
                             columns = ['PC1','PC2']
                            )
# plotting heatmap
plt.figure(figsize =(5, 7))
sns.heatmap(pca_component)
plt.title('PCA Component')
plt.show()
`

**Output**:

![Project the feature on Principal COmponent-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230726123942/download-(4).webp)

Then, we project our dataset using the formula:

ProjPi(u)=Pi⋅u∣u∣=Pi⋅u\\begin{aligned} Proj\_{P\_i}(u) &= \\frac{P\_i\\cdot u}{\|u\|} \\\ &=P\_i\\cdot u \\end{aligned}ProjPi​​(u)​=∣u∣Pi​⋅u​=Pi​⋅u​

![Finding Projection in PCA - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230420165637/Finding-Projection-in-PCA.webp)

Finding Projection in PCA

The **principal component u** (green vector) maximizes data variance and serves as the new axis for projection. The **data point P1(x1,y1)** (red vector) is an original observation, and its **projection onto u** (blue line) represents its transformed coordinate in the reduced dimension. This projection simplifies the data while preserving its key characteristics.

Python`
# Matrix multiplication or dot Product
Z_pca = Z @ pca_component
# Rename the columns name
Z_pca.rename({'PC1': 'PCA1', 'PC2': 'PCA2'}, axis=1, inplace=True)
# Print the  Pricipal Component values
print(Z_pca)
`

**Output**:

```
          PCA1       PCA2
0     9.184755   1.946870
1     2.385703  -3.764859
2     5.728855  -1.074229
3     7.116691  10.266556
4     3.931842  -1.946359
..         ...        ...
564   6.433655  -3.573673
565   3.790048  -3.580897
566   1.255075  -1.900624
567  10.365673   1.670540
568  -5.470430  -0.670047
[569 rows x 2 columns]

```

> The eigenvectors of the covariance matrix of the data are referred to as the principal axes of the data, **and the projection of the data instances onto these principal axes are called the principal components.**

Dimensionality reduction is then obtained by only retaining those axes (dimensions) that account for most of the variance, and discarding all others.

## PCA using Using Sklearn

There are different libraries in which the whole process of the principal component analysis has been automated by implementing it in a package as a function and we just have to pass the number of principal components which we would like to have. Sklearn is one such library that can be used for the PCA as shown below.

Python`
# Importing PCA
from sklearn.decomposition import PCA
# Let's say, components = 2
pca = PCA(n_components=2)
pca.fit(Z)
x_pca = pca.transform(Z)
# Create the dataframe
df_pca1 = pd.DataFrame(x_pca,
                       columns=['PC{}'.\
                       format(i+1)\
                        for i in range(n_components)])
print(df_pca1)
`

**Output:**

```
           PC1        PC2
0     9.184755   1.946870
1     2.385703  -3.764859
2     5.728855  -1.074229
3     7.116691  10.266556
4     3.931842  -1.946359
..         ...        ...
564   6.433655  -3.573673
565   3.790048  -3.580897
566   1.255075  -1.900624
567  10.365673   1.670540
568  -5.470430  -0.670047
[569 rows x 2 columns]

```

We can match from the above Z\_pca result from it is exactly the same values.

Python`
# giving a larger plot
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1],
            c=cancer['target'],
            cmap='plasma')
# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
`

**Output:**

![Visualizing the evaluated principal Component -Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230726125541/download-(3)-(1).webp)

Python`
# components
pca.components_
`

**Output**:

```
array([[ 0.21890244,  0.10372458,  0.22753729,  0.22099499,  0.14258969,\
         0.23928535,  0.25840048,  0.26085376,  0.13816696,  0.06436335,\
         0.20597878,  0.01742803,  0.21132592,  0.20286964,  0.01453145,\
         0.17039345,  0.15358979,  0.1834174 ,  0.04249842,  0.10256832,\
         0.22799663,  0.10446933,  0.23663968,  0.22487053,  0.12795256,\
         0.21009588,  0.22876753,  0.25088597,  0.12290456,  0.13178394],\
       [-0.23385713, -0.05970609, -0.21518136, -0.23107671,  0.18611302,\
         0.15189161,  0.06016536, -0.0347675 ,  0.19034877,  0.36657547,\
        -0.10555215,  0.08997968, -0.08945723, -0.15229263,  0.20443045,\
         0.2327159 ,  0.19720728,  0.13032156,  0.183848  ,  0.28009203,\
        -0.21986638, -0.0454673 , -0.19987843, -0.21935186,  0.17230435,\
         0.14359317,  0.09796411, -0.00825724,  0.14188335,  0.27533947]])

```

**Apart from what we’ve discussed, there are many more subtle advantages and limitations to PCA.**

## Advantages and Disadvantages of Principal Component Analysis

**Advantages of Principal Component Analysis**

1. **Multicollinearity Handling:** Creates new, uncorrelated variables to address issues when original features are highly correlated.
2. **Noise Reduction:** Eliminates components with low variance (assumed to be noise), enhancing data clarity.
3. **Data Compression:** Represents data with fewer components, reducing storage needs and speeding up processing.
4. **Outlier Detection:** Identifies unusual data points by showing which ones deviate significantly in the reduced space.

**Disadvantages of Principal Component Analysis**

1. **Interpretation Challenges:** The new components are combinations of original variables, which can be hard to explain.
2. **Data Scaling Sensitivity:** Requires proper scaling of data before application, or results may be misleading.
3. **Information Loss:** Reducing dimensions may lose some important information if too few components are kept.
4. **Assumption of Linearity:** Works best when relationships between variables are linear, and may struggle with non-linear data.
5. **Computational Complexity:** Can be slow and resource-intensive on very large datasets.
6. **Risk of Overfitting:** Using too many components or working with a small dataset might lead to models that don’t generalize well.

## Conclusion

In summary, PCA helps in distilling complex data into its most informative elements, making it simpler and more efficient to analyze.

1. It identifies the directions (called **principal components**) where the data varies the most.
2. It projects the data onto these directions, reducing the number of dimensions while retaining as much information as possible.
3. The new set of uncorrelated variables (principal components) is easier to work with and can be used for tasks like regression, classification, or visualization.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/)

[ML \| T-distributed Stochastic Neighbor Embedding (t-SNE) Algorithm](https://www.geeksforgeeks.org/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/)

[![author](https://media.geeksforgeeks.org/auth/profile/ukpd91lonq812ahq26j5)](https://www.geeksforgeeks.org/user/aishwarya.27/)

[aishwarya.27](https://www.geeksforgeeks.org/user/aishwarya.27/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [ML-EDA](https://www.geeksforgeeks.org/tag/ml-eda/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Machine Learning Algorithms\\
\\
\\
Machine learning algorithms are essentially sets of instructions that allow computers to learn from data, make predictions, and improve their performance over time without being explicitly programmed. Machine learning algorithms are broadly categorized into three types: Supervised Learning: Algorith\\
\\
9 min read](https://www.geeksforgeeks.org/machine-learning-algorithms/)
[Top 15 Machine Learning Algorithms Every Data Scientist Should Know in 2025\\
\\
\\
Machine Learning (ML) Algorithms are the backbone of everything from Netflix recommendations to fraud detection in financial institutions. These algorithms form the core of intelligent systems, empowering organizations to analyze patterns, predict outcomes, and automate decision-making processes. Wi\\
\\
15 min read](https://www.geeksforgeeks.org/top-10-algorithms-every-machine-learning-engineer-should-know/)

## Linear Model Regression

- [Ordinary Least Squares (OLS) using statsmodels\\
\\
\\
Ordinary Least Squares (OLS) is a widely used statistical method for estimating the parameters of a linear regression model. It minimizes the sum of squared residuals between observed and predicted values. In this article we will learn how to implement Ordinary Least Squares (OLS) regression using P\\
\\
3 min read](https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/)

* * *

- [Linear Regression (Python Implementation)\\
\\
\\
Linear regression is a statistical method that is used to predict a continuous dependent variable i.e target variable based on one or more independent variables. This technique assumes a linear relationship between the dependent and independent variables which means the dependent variable changes pr\\
\\
14 min read](https://www.geeksforgeeks.org/linear-regression-python-implementation/)

* * *

- [ML \| Multiple Linear Regression using Python\\
\\
\\
Linear regression is a fundamental statistical method widely used for predictive analysis. It models the relationship between a dependent variable and a single independent variable by fitting a linear equation to the data. Multiple Linear Regression is an extension of this concept that allows us to\\
\\
4 min read](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/)

* * *

- [Polynomial Regression ( From Scratch using Python )\\
\\
\\
Prerequisites Linear RegressionGradient DescentIntroductionLinear Regression finds the correlation between the dependent variable ( or target variable ) and independent variables ( or features ). In short, it is a linear model to fit the data linearly. But it fails to fit and catch the pattern in no\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/)

* * *

- [Bayesian Linear Regression\\
\\
\\
Linear regression is based on the assumption that the underlying data is normally distributed and that all relevant predictor variables have a linear relationship with the outcome. But In the real world, this is not always possible, it will follows these assumptions, Bayesian regression could be the\\
\\
11 min read](https://www.geeksforgeeks.org/implementation-of-bayesian-regression/)

* * *

- [How to Perform Quantile Regression in Python\\
\\
\\
In this article, we are going to see how to perform quantile regression in Python. Linear regression is defined as the statistical method that constructs a relationship between a dependent variable and an independent variable as per the given set of variables. While performing linear regression we a\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-perform-quantile-regression-in-python/)

* * *

- [Isotonic Regression in Scikit Learn\\
\\
\\
Isotonic regression is a regression technique in which the predictor variable is monotonically related to the target variable. This means that as the value of the predictor variable increases, the value of the target variable either increases or decreases in a consistent, non-oscillating manner. Mat\\
\\
6 min read](https://www.geeksforgeeks.org/isotonic-regression-in-scikit-learn/)

* * *

- [Stepwise Regression in Python\\
\\
\\
Stepwise regression is a method of fitting a regression model by iteratively adding or removing variables. It is used to build a model that is accurate and parsimonious, meaning that it has the smallest number of variables that can explain the data. There are two main types of stepwise regression: F\\
\\
6 min read](https://www.geeksforgeeks.org/stepwise-regression-in-python/)

* * *

- [Least Angle Regression (LARS)\\
\\
\\
Regression is a supervised machine learning task that can predict continuous values (real numbers), as compared to classification, that can predict categorical or discrete values. Before we begin, if you are a beginner, I highly recommend this article. Least Angle Regression (LARS) is an algorithm u\\
\\
3 min read](https://www.geeksforgeeks.org/least-angle-regression-lars/)

* * *


## Linear Model Classification

- [Logistic Regression in Machine Learning\\
\\
\\
In our previous discussion, we explored the fundamentals of machine learning and walked through a hands-on implementation of Linear Regression. Now, let's take a step forward and dive into one of the first and most widely used classification algorithms â€” Logistic Regression What is Logistic Regressi\\
\\
13 min read](https://www.geeksforgeeks.org/understanding-logistic-regression/)

* * *

- [Understanding Activation Functions in Depth\\
\\
\\
In artificial neural networks, the activation function of a neuron determines its output for a given input. This output serves as the input for subsequent neurons in the network, continuing the process until the network solves the original problem. Consider a binary classification problem, where the\\
\\
6 min read](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)

* * *


## Regularization

- [Implementation of Lasso Regression From Scratch using Python\\
\\
\\
Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that combines prediction with feature selection. It does this by adding a penalty term to the cost function shrinking less relevant feature's coefficients to zero. This makes it effective for high-dim\\
\\
7 min read](https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/)

* * *

- [Implementation of Ridge Regression from Scratch using Python\\
\\
\\
Prerequisites: Linear Regression Gradient Descent Introduction: Ridge Regression ( or L2 Regularization ) is a variation of Linear Regression. In Linear Regression, it minimizes the Residual Sum of Squares ( or RSS or cost function ) to fit the training examples perfectly as possible. The cost funct\\
\\
4 min read](https://www.geeksforgeeks.org/implementation-of-ridge-regression-from-scratch-using-python/)

* * *

- [Implementation of Elastic Net Regression From Scratch\\
\\
\\
Prerequisites: Linear RegressionGradient DescentLasso & Ridge RegressionIntroduction: Elastic-Net Regression is a modification of Linear Regression which shares the same hypothetical function for prediction. The cost function of Linear Regression is represented by J. \[Tex\]\\frac{1}{m} \\sum\_{i=1}^\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)

* * *


## K-Nearest Neighbors (KNN)

- [Implementation of Elastic Net Regression From Scratch\\
\\
\\
Prerequisites: Linear RegressionGradient DescentLasso & Ridge RegressionIntroduction: Elastic-Net Regression is a modification of Linear Regression which shares the same hypothetical function for prediction. The cost function of Linear Regression is represented by J. \[Tex\]\\frac{1}{m} \\sum\_{i=1}^\\
\\
5 min read](https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/)

* * *

- [Brute Force Approach and its pros and cons\\
\\
\\
In this article, we will discuss the Brute Force Algorithm and what are its pros and cons. What is the Brute Force Algorithm?A brute force algorithm is a simple, comprehensive search strategy that systematically explores every option until a problem's answer is discovered. It's a generic approach to\\
\\
3 min read](https://www.geeksforgeeks.org/brute-force-approach-and-its-pros-and-cons/)

* * *

- [Implementation of KNN classifier using Scikit - learn - Python\\
\\
\\
K-Nearest Neighbors isÂ aÂ mostÂ simpleÂ butÂ fundamentalÂ classifierÂ algorithmÂ in Machine Learning. ItÂ isÂ underÂ the supervised learningÂ categoryÂ andÂ usedÂ withÂ greatÂ intensityÂ forÂ pattern recognition, data mining andÂ analysis ofÂ intrusion.Â It is widely disposable in real-life scenarios since it is non-par\\
\\
3 min read](https://www.geeksforgeeks.org/ml-implementation-of-knn-classifier-using-sklearn/)

* * *

- [Regression using k-Nearest Neighbors in R Programming\\
\\
\\
Machine learning is a subset of Artificial Intelligence that provides a machine with the ability to learn automatically without being explicitly programmed. The machine in such cases improves from the experience without human intervention and adjusts actions accordingly. It is primarily of 3 types:\\
\\
5 min read](https://www.geeksforgeeks.org/regression-using-k-nearest-neighbors-in-r-programming/)

* * *


## Support Vector Machines

- [Support Vector Machine (SVM) Algorithm\\
\\
\\
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. While it can handle regression problems, SVM is particularly well-suited for classification tasks. SVM aims to find the optimal hyperplane in an N-dimensional space to separate data\\
\\
10 min read](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)

* * *

- [Classifying data using Support Vector Machines(SVMs) in Python\\
\\
\\
Introduction to SVMs: In machine learning, support vector machines (SVMs, also support vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. A Support Vector Machine (SVM) is a discriminative classifier\\
\\
4 min read](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/)

* * *

- [Support Vector Regression (SVR) using Linear and Non-Linear Kernels in Scikit Learn\\
\\
\\
Support vector regression (SVR) is a type of support vector machine (SVM) that is used for regression tasks. It tries to find a function that best predicts the continuous output value for a given input value. SVR can use both linear and non-linear kernels. A linear kernel is a simple dot product bet\\
\\
5 min read](https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/)

* * *

- [Major Kernel Functions in Support Vector Machine (SVM)\\
\\
\\
In previous article we have discussed about SVM(Support Vector Machine) in Machine Learning. Now we are going to learnÂ  in detail about SVM Kernel and Different Kernel Functions and its examples. Types of SVM Kernel FunctionsSVM algorithm use the mathematical function defined by the kernel. Kernel F\\
\\
4 min read](https://www.geeksforgeeks.org/major-kernel-functions-in-support-vector-machine-svm/)

* * *


[ML \| Stochastic Gradient Descent (SGD)\\
\\
\\
Stochastic Gradient Descent (SGD) is an optimization algorithm in machine learning, particularly when dealing with large datasets. It is a variant of the traditional gradient descent algorithm but offers several advantages in terms of efficiency and scalability, making it the go-to method for many d\\
\\
8 min read](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)

## Decision Tree

- [Major Kernel Functions in Support Vector Machine (SVM)\\
\\
\\
In previous article we have discussed about SVM(Support Vector Machine) in Machine Learning. Now we are going to learnÂ  in detail about SVM Kernel and Different Kernel Functions and its examples. Types of SVM Kernel FunctionsSVM algorithm use the mathematical function defined by the kernel. Kernel F\\
\\
4 min read](https://www.geeksforgeeks.org/major-kernel-functions-in-support-vector-machine-svm/)

* * *

- [CART (Classification And Regression Tree) in Machine Learning\\
\\
\\
CART( Classification And Regression Trees) is a variation of the decision tree algorithm. It can handle both classification and regression tasks. Scikit-Learn uses the Classification And Regression Tree (CART) algorithm to train Decision Trees (also called â€œgrowingâ€ trees). CART was first produced b\\
\\
11 min read](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/)

* * *

- [Decision Tree Classifiers in R Programming\\
\\
\\
Classification is the task in which objects of several categories are categorized into their respective classes using the properties of classes. A classification model is typically used to, Predict the class label for a new unlabeled data objectProvide a descriptive model explaining what features ch\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-classifiers-in-r-programming/)

* * *

- [Python \| Decision Tree Regression using sklearn\\
\\
\\
When it comes to predicting continuous values, Decision Tree Regression is a powerful and intuitive machine learning technique. Unlike traditional linear regression, which assumes a straight-line relationship between input features and the target variable, Decision Tree Regression is a non-linear re\\
\\
4 min read](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/)

* * *


## Ensemble Learning

- [Ensemble Methods in Python\\
\\
\\
Ensemble means a group of elements viewed as a whole rather than individually. An Ensemble method creates multiple models and combines them to solve it. Ensemble methods help to improve the robustness/generalizability of the model. In this article, we will discuss some methods with their implementat\\
\\
11 min read](https://www.geeksforgeeks.org/ensemble-methods-in-python/)

* * *

- [Random Forest Regression in Python\\
\\
\\
A random forest is an ensemble learning method that combines the predictions from multiple decision trees to produce a more accurate and stable prediction. It is a type of supervised learning algorithm that can be used for both classification and regression tasks. In regression task we can use Rando\\
\\
9 min read](https://www.geeksforgeeks.org/random-forest-regression-in-python/)

* * *

- [ML \| Extra Tree Classifier for Feature Selection\\
\\
\\
Prerequisites: Decision Tree Classifier Extremely Randomized Trees Classifier(Extra Trees Classifier) is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a "forest" to output it's classification result. In concept, it is very si\\
\\
6 min read](https://www.geeksforgeeks.org/ml-extra-tree-classifier-for-feature-selection/)

* * *

- [Implementing the AdaBoost Algorithm From Scratch\\
\\
\\
AdaBoost means Adaptive Boosting and it is a is a powerful ensemble learning technique that combines multiple weak classifiers to create a strong classifier. It works by sequentially adding classifiers to correct the errors made by previous models giving more weight to the misclassified data points.\\
\\
3 min read](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/)

* * *

- [XGBoost\\
\\
\\
Traditional machine learning models like decision trees and random forests are easy to interpret but often struggle with accuracy on complex datasets. XGBoost, short for eXtreme Gradient Boosting, is an advanced machine learning algorithm designed for efficiency, speed, and high performance. What is\\
\\
9 min read](https://www.geeksforgeeks.org/xgboost/)

* * *

- [CatBoost in Machine Learning\\
\\
\\
When working with machine learning, we often deal with datasets that include categorical data. We use techniques like One-Hot Encoding or Label Encoding to convert these categorical features into numerical values. However One-Hot Encoding can lead to sparse matrix and cause overfitting. This is wher\\
\\
7 min read](https://www.geeksforgeeks.org/catboost-ml/)

* * *

- [LightGBM (Light Gradient Boosting Machine)\\
\\
\\
LightGBM is an open-source high-performance framework developed by Microsoft. It is an ensemble learning framework that uses gradient boosting method which constructs a strong learner by sequentially adding weak learners in a gradient descent manner. It's designed for efficiency, scalability and hig\\
\\
7 min read](https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/)

* * *

- [Stacking in Machine Learning\\
\\
\\
Stacking is a way to ensemble multiple classifications or regression model. There are many ways to ensemble models, the widely known models are Bagging or Boosting. Bagging allows multiple similar models with high variance are averaged to decrease variance. Boosting builds multiple incremental model\\
\\
2 min read](https://www.geeksforgeeks.org/stacking-in-machine-learning/)

* * *


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/principal-component-analysis-pca/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2136860372.1745056620&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1605314793)