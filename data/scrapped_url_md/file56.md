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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-non-linear-svm/?type%3Darticle%26id%3D301879&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Dunn index and DB index - Cluster Validity indices \| Set 1\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/)

# ML \| Non-Linear SVM

Last Updated : 22 Jan, 2025

Comments

Improve

Suggest changes

6 Likes

Like

Report

[Support Vector Machines (SVM)](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) are powerful algorithms for classification and regression tasks. However, the standard (linear) SVM can only classify data that is **linearly separable**, meaning the classes can be separated by a straight line (in 2D) or a hyperplane (in higher dimensions). _**Non-Linear SVM extends SVM to handle complex, non-linearly separable data using kernels. Kernels enable SVM to work in higher dimensions where data can become linearly separable.**_

## Understanding Non-Linear SVM

Nonlinear SVM was introduced when the **data cannot be separated by a linear decision boundary in the original feature space**. The kernel function computes the similarity between data points allowing SVM to capture complex patterns and nonlinear relationships between features. This enables nonlinear SVM **to form curved or circular decision boundaries with help of kernel.**

### What is Kernel Trick?

Instead of explicitly computing the transformation, **the kernel trick computes the dot product of data points in the higher-dimensional space directly** that helps a model find patterns in complex data and transforming the data into a higher-dimensional space where it becomes easier to separate different classes or detect relationships.

**For example, imagine we have data points shaped like two concentric circles**: one circle represents one class and the other circle represents another class. If we try to separate these classes with a straight line it can’t be done because the data is not linearly separable in its current form.

![non-linear-svm](https://media.geeksforgeeks.org/wp-content/uploads/20250122160409140383/non-linear-svm.webp)

Non-Linear SVM

When we use a kernel function, it transforms the original 2D data (like the concentric circles) into a higher-dimensional space where the data becomes linearly separable. In that higher-dimensional space, the SVM finds a simple straight-line decision boundary to separate the classes.

When we bring this straight-line decision boundary back to the original 2D space, it no longer looks like a straight line. Instead, it appears as a circular boundary that perfectly separates the two classes. This happens because the kernel trick allows the SVM to “see” the data in a new way, enabling it to draw a boundary that fits the original shape of the data.

### Popular kernel functions in SVM

- **Radial Basis Function (RBF)**: Captures patterns in data by measuring the distance between points and is ideal for circular or spherical relationships.
- **Linear Kernel**: Works for data that is linearly separable problem without complex transformations.
- **Polynomial Kernel**: Models more complex relationships using polynomial equations.
- **Sigmoid Kernel**: Mimics neural network behavior using sigmoid function and is suitable for specific non-linear problems.

### Example :  Non-Linear SVM Classification

Below is the Python implementation for Non linear SVM in circular decision boundary.

Python`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the non-linear SVM with an RBF kernel
svm = SVC(kernel='rbf', C=1, gamma=0.5)  # C and gamma are hyperparameters
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title("Non-linear SVM with RBF Kernel")
    plt.show()
plot_decision_boundary(X, y, svm)
`

**Output**:

![download4](https://media.geeksforgeeks.org/wp-content/uploads/20250121153712588915/download4.png)

Non Linear SVM with RBF kernel

Non linear SVM provided a decision boundary where the SVM successfully separates the two circular classes (inner and outer circles) using a curved boundary with help of RBF kernel.

Now we will see how different kernel works. **We will be using polynomial kernel function for dataset with radial curve pattern.**

Python`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the SVM with a Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1, coef0=1)  # degree and coef0 are key hyperparameters
svm_poly.fit(X_train, y_train)
y_pred = svm_poly.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title("Non-linear SVM with Polynomial Kernel")
    plt.show()
plot_decision_boundary(X, y, svm_poly)
`

**Output:**

![download5](https://media.geeksforgeeks.org/wp-content/uploads/20250121155050429170/download5.png)

Non linear SVM with Polynomial Kernel

Polynomial kernel creates a smooth, non-linear decision boundary that effectively separates the two curved regions.

## **Applications**

1. **Image Classification**: They are widely used for image recognition tasks such as handwritten digit recognition like MNIST dataset, where the data classes are not linearly separable.
2. **Bioinformatics**: Used in gene expression analysis and protein classification where the relationships between variables are complex and non-linear.
3. **Natural Language Processing (NLP)**: Used for text classification tasks like spam filtering or sentiment analysis where non-linear relationships exist between words and sentiments.
4. **Medical Diagnosis**: Effective for classifying diseases based on patient data such as tumor classification where data have non-linear patterns.
5. **Fraud Detection**: They can identify fraudulent activities by detecting unusual patterns in transactional data.
6. **Voice and Speech Recognition**: Useful for separating different voice signals or identifying speech patterns where non-linear decision boundaries are needed.
7. **Customer Segmentation**: Helps in customer segments with non-linear patterns in purchasing behavior.

Non-Linear SVM is a versatile and powerful machine learning algorithm that excels in handling complex datasets where linear separation is not possible. By using kernel functions such as RBF, polynomial and sigmoid it transforms data into higher-dimensional spaces to form decision boundaries.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/)

[Dunn index and DB index - Cluster Validity indices \| Set 1](https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/)

[P](https://www.geeksforgeeks.org/user/Praveen%20Sinha/)

[Praveen Sinha](https://www.geeksforgeeks.org/user/Praveen%20Sinha/)

Follow

6

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Linear mapping\\
\\
\\
Linear mapping is a mathematical operation that transforms a set of input values into a set of output values using a linear function. In machine learning, linear mapping is often used as a preprocessing step to transform the input data into a more suitable format for analysis. Linear mapping can als\\
\\
7 min read](https://www.geeksforgeeks.org/linear-mapping/?ref=ml_lbp)
[SVM vs KNN in Machine Learning\\
\\
\\
Support Vector Machine(SVM) and K Nearest Neighbours(KNN) both are very popular supervised machine learning algorithms used for classification and regression purpose. Both SVM and KNN play an important role in Supervised Learning. Table of Content Support Vector Machine(SVM)K Nearest Neighbour(KNN)S\\
\\
5 min read](https://www.geeksforgeeks.org/svm-vs-knn-in-machine-learning/?ref=ml_lbp)
[PCA and SVM Pipeline in Python\\
\\
\\
Principal Component Analysis (PCA) and Support Vector Machines (SVM) are powerful techniques used in machine learning for dimensionality reduction and classification, respectively. Combining them into a pipeline can enhance the performance of the overall system, especially when dealing with high-dim\\
\\
5 min read](https://www.geeksforgeeks.org/pca-and-svm-pipeline-in-python/?ref=ml_lbp)
[ML \| Multiple Linear Regression using Python\\
\\
\\
Linear regression is a fundamental statistical method widely used for predictive analysis. It models the relationship between a dependent variable and a single independent variable by fitting a linear equation to the data. Multiple Linear Regression is an extension of this concept that allows us to\\
\\
4 min read](https://www.geeksforgeeks.org/ml-multiple-linear-regression-using-python/?ref=ml_lbp)
[Linear Regression using PyTorch\\
\\
\\
Linear Regression is a very commonly used statistical method that allows us to determine and study the relationship between two continuous variables. The various properties of linear regression and its Python implementation have been covered in this article previously. Now, we shall find out how to\\
\\
4 min read](https://www.geeksforgeeks.org/linear-regression-using-pytorch/?ref=ml_lbp)
[ML \| Normal Equation in Linear Regression\\
\\
\\
Linear regression is a popular method for understanding how different factors (independent variables) affect an outcome (dependent variable. At its core, linear regression aims to find the best-fitting line that minimizes the error between observed data points and predicted values. One efficient met\\
\\
8 min read](https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/?ref=ml_lbp)
[Polynomial Regression for Non-Linear Data - ML\\
\\
\\
Non-linear data is usually encountered in daily life. Consider some of the equations of motion as studied in physics. Projectile Motion: The height of a projectile is calculated as h = -Â½ gt2 +ut +ho Equation of motion under free fall: The distance travelled by an object after falling freely under g\\
\\
5 min read](https://www.geeksforgeeks.org/polynomial-regression-for-non-linear-data-ml/?ref=ml_lbp)
[Separating Hyperplanes in SVM\\
\\
\\
Support Vector Machine is the supervised machine learning algorithm, that is used in both classification and regression of models. The idea behind it is simple to just find a plane or a boundary that separates the data between two classes. Support Vectors: Support vectors are the data points that ar\\
\\
7 min read](https://www.geeksforgeeks.org/separating-hyperplanes-in-svm/?ref=ml_lbp)
[ML \| Locally weighted Linear Regression\\
\\
\\
Linear Regression is a supervised learning algorithm used for computing linear relationships between input (X) and output (Y). The steps involved in ordinary linear regression are: Training phase: Compute \[Tex\]\\theta \[/Tex\]to minimize the cost. \[Tex\]J(\\theta) = $\\sum\_{i=1}^{m} (\\theta^Tx^{(i)} - y^{\\
\\
3 min read](https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/?ref=ml_lbp)
[SVD in Recommendation Systems\\
\\
\\
Recommender systems have become a vital part of our digital lives, guiding us towards products, services, and content we might like. Among the various techniques used to power these systems, Singular Value Decomposition (SVD) and Matrix Factorization (MF) are prominent methods. This article explores\\
\\
3 min read](https://www.geeksforgeeks.org/svd-in-recommendation-systems/?ref=ml_lbp)

Like6

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-non-linear-svm/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1223139165.1745055901&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=242194550)

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