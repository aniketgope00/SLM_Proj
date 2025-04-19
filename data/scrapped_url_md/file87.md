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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/dimensionality-reduction/?type%3Darticle%26id%3D147579&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
ML \| Introduction to Kernel PCA\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca/)

# Introduction to Dimensionality Reduction

Last Updated : 22 Mar, 2025

Comments

Improve

Suggest changes

86 Likes

Like

Report

When working with machine learning models, datasets with too many features can cause issues like slow computation and overfitting. Dimensionality reduction helps by reducing the number of features while retaining key information.

Techniques like [**principal component analysis (PCA)**](https://www.geeksforgeeks.org/principal-component-analysis-pca/), [**singular value decomposition (SVD)**](https://www.geeksforgeeks.org/singular-value-decomposition-svd/) and [**linear discriminant analysis (LDA)**](https://www.geeksforgeeks.org/videos/linear-discriminant-analysis-lda-implementation-machine-learning/) project data onto a lower-dimensional space, preserving important details.

**Example:**

> when you are building a model to predict house prices with features like bedrooms, square footage, and location. If you add too many features, such as room condition or flooring type, the dataset becomes large and complex.

### Before Dimensionality Reduction

With too many features, training can slow down and the model may focus on irrelevant details, like flooring type, which could lead to inaccurate predictions.

## How Dimensionality Reduction Works?

Lets understand how dimensionality Reduction is used with the help of the figure below:

![](https://media.geeksforgeeks.org/wp-content/uploads/Dimensionality_Reduction_1.jpg)

- On the left, data points exist in a **3D space** (X, Y, Z), but the Z-dimension appears unnecessary since the data primarily varies along the X and Y axes. The goal of dimensionality reduction is to remove less important dimensions without losing valuable information.
- On the right, after reducing the dimensionality, the data is represented in **lower-dimensional spaces**. The top plot (X-Y) maintains the meaningful structure, while the bottom plot (Z-Y) shows that the Z-dimension contributed little useful information.

This process makes [data analysis](https://www.geeksforgeeks.org/what-is-data-analysis/) more efficient, improving computation speed and visualization while minimizing redundancy

## Dimensionality Reduction Techniques

Dimensionality reduction techniques can be broadly divided into two categories:

### Feature Selection

[Feature selection](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) **chooses** the most relevant features from the dataset without altering them. It helps remove redundant or irrelevant features, improving model efficiency. There are several methods for feature selection including **filter methods, wrapper methods and embedded methods**.

- [Filter methods](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) rank the features based on their relevance to the target variable.
- [Wrapper methods](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) use the model performance as the criteria for selecting features.
- [Embedded methods](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) combine feature selection with the model training process.

> Please refer to [Feature Selection Techniques](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) for better in depth understanding about the techniques.

### Feature Extraction

[Feature extraction](https://www.geeksforgeeks.org/what-is-feature-extraction/) involves creating new features by combining or transforming the original features. There are several methods for feature extraction stated above in the introductory part which is responsible for creating and transforming the features. [PCA](https://www.geeksforgeeks.org/principal-component-analysis-pca/) is a popular technique that projects the original features onto a lower-dimensional space while preserving as much of the variance as possible.

**Although one can perform dimensionality reduction with several techniques, the following are the most commonly used ones:**

1. [**Principal Component Analysis (PCA):**](https://www.geeksforgeeks.org/principal-component-analysis-pca/) Converts correlated variables into uncorrelated ‘principal components,’ reducing dimensionality while maintaining as much variance as possible, enabling more efficient analysis.
2. [**Missing Value Ratio:**](https://www.geeksforgeeks.org/ml-handling-missing-values/) Variables with missing data beyond a set threshold are removed, improving dataset reliability.
3. [**Backward Feature Elimination**:](https://www.geeksforgeeks.org/ml-multiple-linear-regression-backward-elimination-technique/) Starts with all features and removes the least significant ones in each iteration. The process continues until only the most impactful features remain, optimizing model performance.
4. [**Forward Feature Selection:**](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/) Forward Feature SelectionBegins with one feature, adds others incrementally, and keeps those improving model performance.
5. [**Random Forest**](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/): Random forest Uses [decision trees](https://www.geeksforgeeks.org/decision-tree/) to evaluate feature importance, automatically selecting the most relevant features without the need for manual coding, enhancing model accuracy.
6. [**Factor Analysis**:](https://www.geeksforgeeks.org/introduction-to-factor-analytics/) Groups variables by correlation and keeps the most relevant ones for further analysis.
7. [**Independent Component Analysis (ICA)**:](https://www.geeksforgeeks.org/ml-independent-component-analysis/) Identifies statistically independent components, ideal for applications like ‘blind source separation’ where traditional correlation-based methods fall short.

## **Dimensionality Reduction Examples**

Dimensionality reduction plays a crucial role in many real-world applications, such as text categorization, image retrieval, gene expression analysis, and more. Here are a few examples:

1. **Text Categorization:** With vast amounts of online data, dimensionality reduction helps classify text documents into predefined categories by reducing the feature space (like word or phrase features) while maintaining accuracy.
2. **Image Retrieval**: As image data grows, indexing based on visual content (color, texture, shape) rather than just text descriptions has become essential. This allows for better retrieval of images from large databases.
3. **Gene Expression Analysis**: Dimensionality reduction accelerates gene expression analysis, helping classify samples (e.g., leukemia) by identifying key features, improving both speed and accuracy.
4. **Intrusion Detection**: In [cybersecurity](https://www.geeksforgeeks.org/what-is-cyber-security/), dimensionality reduction helps analyze user activity patterns to detect suspicious behaviors and intrusions by identifying optimal features for network monitoring.

## **Advantages of Dimensionality Reduction**

As seen earlier, high dimensionality makes models inefficient. Let’s now summarize the key advantages of reducing dimensionality.

- **Faster Computation**: With fewer features, [machine learning](https://www.geeksforgeeks.org/machine-learning/) algorithms can process data more quickly. This results in faster model training and testing, which is particularly useful when working with large datasets.
- **Better Visualization**: As we saw in the earlier figure, reducing dimensions makes it easier to visualize data, revealing hidden patterns.
- **Prevent Overfitting**: With fewer features, models are less likely to memorize the training data and [overfit](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/). This helps the model generalize better to new, unseen data, improving its ability to make accurate predictions.

## **Disadvantages of Dimensionality Reduction**

- **Data Loss & Reduced Accuracy** – Some important information may be lost during dimensionality reduction, potentially affecting model performance.
- **Choosing the Right Components** – Deciding how many dimensions to keep is difficult, as keeping too few may lose valuable information, while keeping too many can lead to overfitting.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca/)

[ML \| Introduction to Kernel PCA](https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca/)

A

Anannya Uberoi

86

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Machine Learning with R\\
\\
\\
Machine Learning as the name suggests is the field of study that allows computers to learn and take decisions on their own i.e. without being explicitly programmed. These decisions are based on the available data that is available through experiences or instructions. It gives the computer that makes\\
\\
2 min read](https://www.geeksforgeeks.org/machine-learning-with-r/)

## Getting Started With Machine Learning In R

- [Introduction to Machine Learning in R\\
\\
\\
The word Machine Learning was first coined by Arthur Samuel in 1959. The definition of machine learning can be defined as that machine learning gives computers the ability to learn without being explicitly programmed. Also in 1997, Tom Mitchell defined machine learning that â€œA computer program is sa\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-to-machine-learning-in-r/)

* * *

- [What is Machine Learning?\\
\\
\\
Machine learning is a branch of artificial intelligence that enables algorithms to uncover hidden patterns within datasets. It allows them to predict new, similar data without explicit programming for each task. Machine learning finds applications in diverse fields such as image and speech recogniti\\
\\
9 min read](https://www.geeksforgeeks.org/ml-machine-learning/)

* * *

- [Setting up Environment for Machine Learning with R Programming\\
\\
\\
Machine Learning is a subset of Artificial Intelligence (AI), which is used to create intelligent systems that are able to learn without being programmed explicitly. In machine learning, we create algorithms and models which is used by an intelligent system to predict outcomes based on particular pa\\
\\
6 min read](https://www.geeksforgeeks.org/setting-up-environment-for-machine-learning-with-r-programming/)

* * *

- [Supervised and Unsupervised Learning in R Programming\\
\\
\\
Arthur Samuel, a pioneer in the field of artificial intelligence and computer gaming, coined the term â€œMachine Learningâ€. He defined machine learning as â€“ â€œField of study that gives computers the capability to learn without being explicitly programmedâ€. In a very layman manner, Machine Learning(ML)\\
\\
8 min read](https://www.geeksforgeeks.org/supervised-and-unsupervised-clustering-in-r-programming/)

* * *


## Data Processing

- [Introduction to Data in Machine Learning\\
\\
\\
Data refers to the set of observations or measurements to train a machine learning models. The performance of such models is heavily influenced by both the quality and quantity of data available for training and testing. Machine learningÂ algorithmsÂ cannotÂ be trained without data.Â Cutting-edgeÂ develo\\
\\
4 min read](https://www.geeksforgeeks.org/ml-introduction-data-machine-learning/)

* * *

- [ML \| Understanding Data Processing\\
\\
\\
In machine learning, data is the most important aspect, but the raw data is messy, incomplete, or unstructured. So, we process the raw data to transform it into a clean, structured format for analysis, and this step in the data science pipeline is known as data processing. Without data processing, e\\
\\
5 min read](https://www.geeksforgeeks.org/ml-understanding-data-processing/)

* * *

- [ML \| Overview of Data Cleaning\\
\\
\\
Data cleaning is a important step in the machine learning (ML) pipeline as it involves identifying and removing any missing duplicate or irrelevant data. The goal of data cleaning is to ensure that the data is accurate, consistent and free of errors as raw data is often noisy, incomplete and inconsi\\
\\
14 min read](https://www.geeksforgeeks.org/data-cleansing-introduction/)

* * *

- [ML \| Feature Scaling - Part 1\\
\\
\\
Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing. Working: Given a data-set with features- Age, Salary, BHK Apartment with the data size of 5000 people, each having these independent data featu\\
\\
3 min read](https://www.geeksforgeeks.org/ml-feature-scaling-part-1/)

* * *


## Supervised Learning

- [Simple Linear Regression in R\\
\\
\\
Regression shows a line or curve that passes through all the data points on the target-predictor graph in such a way that the vertical distance between the data points and the regression line is minimum What is Linear Regression?Linear Regression is a commonly used type of predictive analysis. Linea\\
\\
12 min read](https://www.geeksforgeeks.org/simple-linear-regression-using-r/)

* * *

- [Multiple Linear Regression using R\\
\\
\\
Prerequisite: Simple Linear-Regression using RLinear Regression: It is the basic and commonly used type for predictive analysis. It is a statistical approach for modeling the relationship between a dependent variable and a given set of independent variables.These are of two types: Simple linear Regr\\
\\
3 min read](https://www.geeksforgeeks.org/multiple-linear-regression-using-r/)

* * *

- [Decision Tree for Regression in R Programming\\
\\
\\
Decision tree is a type of algorithm in machine learning that uses decisions as the features to represent the result in the form of a tree-like structure. It is a common tool used to visually represent the decisions made by the algorithm. Decision trees use both classification and regression. Regres\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-for-regression-in-r-programming/)

* * *

- [Decision Tree Classifiers in R Programming\\
\\
\\
Classification is the task in which objects of several categories are categorized into their respective classes using the properties of classes. A classification model is typically used to, Predict the class label for a new unlabeled data objectProvide a descriptive model explaining what features ch\\
\\
4 min read](https://www.geeksforgeeks.org/decision-tree-classifiers-in-r-programming/)

* * *

- [Random Forest Approach in R Programming\\
\\
\\
Random Forest in R Programming is an ensemble of decision trees. It builds and combines multiple decision trees to get more accurate predictions. It's a non-linear classification algorithm. Each decision tree model is used when employed on its own. An error estimate of cases is made that is not used\\
\\
4 min read](https://www.geeksforgeeks.org/random-forest-approach-in-r-programming/)

* * *

- [Random Forest Approach for Regression in R Programming\\
\\
\\
Random Forest approach is a supervised learning algorithm. It builds the multiple decision trees which are known as forest and glue them together to urge a more accurate and stable prediction. The random forest approach is similar to the ensemble technique called as Bagging. In this approach, multip\\
\\
3 min read](https://www.geeksforgeeks.org/random-forest-approach-for-regression-in-r-programming/)

* * *

- [Random Forest Approach for Classification in R Programming\\
\\
\\
Random forest approach is supervised nonlinear classification and regression algorithm. Classification is a process of classifying a group of datasets in categories or classes. As random forest approach can use classification or regression techniques depending upon the user and target or categories\\
\\
4 min read](https://www.geeksforgeeks.org/random-forest-approach-for-classification-in-r-programming/)

* * *

- [Classifying data using Support Vector Machines(SVMs) in R\\
\\
\\
In machine learning, Support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. It is mostly used in classification problems. In this algorithm, each data item is plotted as a point in n-dimensio\\
\\
5 min read](https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-r/)

* * *

- [Support Vector Machine Classifier Implementation in R with Caret package\\
\\
\\
One of the most crucial aspects of machine learning that most data scientists run against in their careers is the classification problem. The goal of a classification algorithm is to foretell whether a particular activity will take place or not. Depending on the data available, classification algori\\
\\
7 min read](https://www.geeksforgeeks.org/support-vector-machine-classifier-implementation-in-r-with-caret-package/)

* * *

- [KNN Classifier in R Programming\\
\\
\\
K-Nearest Neighbor or KNN is a Supervised Non-linear classification algorithm. KNN in R Programming Language is a Non-parametric algorithm i.e. it doesn't make any assumption about underlying data or its distribution. KNN in R is one of the simplest and most widely used algorithms which depends on i\\
\\
7 min read](https://www.geeksforgeeks.org/k-nn-classifier-in-r-programming/)

* * *


## Evaluation Metrics

- [Precision, Recall and F1-Score using R\\
\\
\\
In machine learning, evaluating model performance is critical. Three widely used metricsâ€”Precision, Recall, and F1-Scoreâ€”help assess the quality of classification models. Here's what each metric represents: Recall: Measures the proportion of actual positive cases correctly identified. Also known as\\
\\
3 min read](https://www.geeksforgeeks.org/precision-recall-and-f1-score-using-r/)

* * *

- [How to Calculate F1 Score in R?\\
\\
\\
In this article, we will be looking at the approach to calculate F1 Score using the various packages and their various functionalities in the R language. F1 Score The F-score or F-measure is a measure of a test's accuracy. It is calculated from the precision and recall of the test, where the precisi\\
\\
5 min read](https://www.geeksforgeeks.org/how-to-calculate-f1-score-in-r/)

* * *


## Unsupervised Learning

- [K-Means Clustering in R Programming\\
\\
\\
K Means Clustering in R Programming is an Unsupervised Non-linear algorithm that cluster data based on similarity or similar groups. It seeks to partition the observations into a pre-specified number of clusters. Segmentation of data takes place to assign each training example to a segment called a\\
\\
3 min read](https://www.geeksforgeeks.org/k-means-clustering-in-r-programming/)

* * *

- [Hierarchical Clustering in R Programming\\
\\
\\
Hierarchical clustering in R Programming Language is an Unsupervised non-linear algorithm in which clusters are created such that they have a hierarchy(or a pre-determined ordering). For example, consider a family of up to three generations. A grandfather and mother have their children that become f\\
\\
3 min read](https://www.geeksforgeeks.org/hierarchical-clustering-in-r-programming/)

* * *

- [How to Perform Hierarchical Cluster Analysis using R Programming?\\
\\
\\
Cluster analysis or clustering is a technique to find subgroups of data points within a data set. The data points belonging to the same subgroup have similar features or properties. Clustering is an unsupervised machine learning approach and has a wide variety of applications such as market research\\
\\
5 min read](https://www.geeksforgeeks.org/how-to-perform-hierarchical-cluster-analysis-using-r-programming/)

* * *

- [Linear Discriminant Analysis in R Programming\\
\\
\\
One of the most popular or well established Machine Learning technique is Linear Discriminant Analysis (LDA ). It is mainly used to solve classification problems rather than supervised classification problems. It is basically a dimensionality reduction technique. Using the Linear combinations of pre\\
\\
6 min read](https://www.geeksforgeeks.org/linear-discriminant-analysis-in-r-programming/)

* * *


## Model Selection and Evaluation

- [Cross-Validation in R programming\\
\\
\\
The major challenge in designing a machine learning model is to make it work accurately on the unseen data. To know whether the designed model is working fine or not, we have to test it against those data points which were not present during the training of the model. These data points will serve th\\
\\
9 min read](https://www.geeksforgeeks.org/cross-validation-in-r-programming/)

* * *

- [LOOCV (Leave One Out Cross-Validation) in R Programming\\
\\
\\
LOOCV (Leave-One-Out Cross-Validation) is a cross-validation technique where each individual observation in the dataset is used once as the validation set, while the remaining observations are used as the training set. This process is repeated for all observations, with each one serving as the valid\\
\\
4 min read](https://www.geeksforgeeks.org/loocvleave-one-out-cross-validation-in-r-programming/)

* * *

- [Bias-Variance Trade Off - Machine Learning\\
\\
\\
It is important to understand prediction errors (bias and variance) when it comes to accuracy in any machine-learning algorithm. There is a tradeoff between a modelâ€™s ability to minimize bias and variance which is referred to as the best solution for selecting a value of Regularization constant. A p\\
\\
3 min read](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/)

* * *


## Reinforcement Learning

- [Markov Decision Process\\
\\
\\
Reinforcement Learning:Reinforcement Learning is a type of Machine Learning. It allows machines and software agents to automatically determine the ideal behavior within a specific context, in order to maximize its performance. Simple reward feedback is required for the agent to learn its behavior; t\\
\\
4 min read](https://www.geeksforgeeks.org/markov-decision-process/)

* * *

- [Q-Learning in Reinforcement Learning\\
\\
\\
Q-learning is a model-free reinforcement learning algorithm used to train agents (computer programs) to make optimal decisions by interacting with an environment. It helps the agent explore different actions and learn which ones lead to better outcomes. The agent uses trial and error to determine wh\\
\\
9 min read](https://www.geeksforgeeks.org/q-learning-in-python/)

* * *

- [Deep Q-Learning in Reinforcement Learning\\
\\
\\
Deep Q-Learning integrates deep neural networks into the decision-making process. This combination allows agents to handle high-dimensional state spaces, making it possible to solve complex tasks such as playing video games or controlling robots. Before diving into Deep Q-Learning, itâ€™s important to\\
\\
4 min read](https://www.geeksforgeeks.org/deep-q-learning/)

* * *


## Dimensionality Reduction

- [Introduction to Dimensionality Reduction\\
\\
\\
When working with machine learning models, datasets with too many features can cause issues like slow computation and overfitting. Dimensionality reduction helps by reducing the number of features while retaining key information. Techniques like principal component analysis (PCA), singular value dec\\
\\
5 min read](https://www.geeksforgeeks.org/dimensionality-reduction/)

* * *

- [ML \| Introduction to Kernel PCA\\
\\
\\
PRINCIPAL COMPONENT ANALYSIS: is a tool which is used to reduce the dimension of the data. It allows us to reduce the dimension of the data without much loss of information. PCA reduces the dimension by finding a few orthogonal linear combinations (principal components) of the original variables wit\\
\\
6 min read](https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca/)

* * *

- [Principal Component Analysis with R Programming\\
\\
\\
Principal component analysis(PCA) in R programming is an analysis of the linear components of all existing attributes. Principal components are linear combinations (orthogonal transformation) of the original predictor in the dataset. It is a useful technique for EDA(Exploratory data analysis) and al\\
\\
3 min read](https://www.geeksforgeeks.org/principal-component-analysis-with-r-programming/)

* * *


## Advanced Topics

- [Kolmogorov-Smirnov Test in R Programming\\
\\
\\
Kolmogorov-Smirnov (K-S) test is a non-parametric test employed to check whether the probability distributions of a sample and a control distribution, or two samples are equal. It is constructed based on the cumulative distribution function (CDF) and calculates the greatest difference between the em\\
\\
4 min read](https://www.geeksforgeeks.org/kolmogorov-smirnov-test-in-r-programming/)

* * *

- [Moore â€“ Penrose Pseudoinverse in R Programming\\
\\
\\
The concept used to generalize the solution of a linear equation is known as Moore â€“ Penrose Pseudoinverse of a matrix. Moore â€“ Penrose inverse is the most widely known type of matrix pseudoinverse. In linear algebra pseudoinverse \[Tex\]A^{+} \[/Tex\]of a matrix A is a generalization of the inverse mat\\
\\
5 min read](https://www.geeksforgeeks.org/moore-penrose-pseudoinverse-in-r-programming/)

* * *

- [Spearman Correlation Testing in R Programming\\
\\
\\
Correlation is a key statistical concept used to measure the strength and direction of the relationship between two variables. Unlike Pearsonâ€™s correlation, which assumes a linear relationship and continuous data, Spearmanâ€™s rank correlation coefficient is a non-parametric measure that assesses how\\
\\
3 min read](https://www.geeksforgeeks.org/spearman-correlation-testing-in-r-programming/)

* * *

- [Poisson Functions in R Programming\\
\\
\\
The Poisson distribution represents the probability of a provided number of cases happening in a set period of space or time if these cases happen with an identified constant mean rate (free of the period since the ultimate event). Poisson distribution has been named after SimÃ©on Denis Poisson(Frenc\\
\\
3 min read](https://www.geeksforgeeks.org/poisson-functions-in-r-programming/)

* * *

- [Feature Engineering in R Programming\\
\\
\\
Feature engineering is the process of transforming raw data into features that can be used in a machine-learning model. In R programming, feature engineering can be done using a variety of built-in functions and packages. One common approach to feature engineering is to use the dplyr package to mani\\
\\
7 min read](https://www.geeksforgeeks.org/feature-engineering-in-r-programming/)

* * *

- [Adjusted Coefficient of Determination in R Programming\\
\\
\\
Prerequisite: Multiple Linear Regression using R A well-fitting regression model produces predicted values close to the observed data values. The mean model, which uses the mean for every predicted value, commonly would be used if there were no informative predictor variables. The fit of a proposed\\
\\
3 min read](https://www.geeksforgeeks.org/adjusted-coefficient-of-determination-in-r-programming/)

* * *

- [Mann Whitney U Test in R Programming\\
\\
\\
A popular nonparametric(distribution-free) test to compare outcomes between two independent groups is the Mann Whitney U test. When comparing two independent samples, when the outcome is not normally distributed and the samples are small, a nonparametric test is appropriate. It is used to see the di\\
\\
4 min read](https://www.geeksforgeeks.org/mann-whitney-u-test-in-r-programming/)

* * *

- [Bootstrap Confidence Interval with R Programming\\
\\
\\
Bootstrapping is a statistical method for inference about a population using sample data. It can be used to estimate the confidence interval(CI) by drawing samples with replacement from sample data. Bootstrapping can be used to assign CI to various statistics that have no closed-form or complicated\\
\\
5 min read](https://www.geeksforgeeks.org/bootstrap-confidence-interval-with-r-programming/)

* * *


Like86

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/dimensionality-reduction/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1298241461.1745056617&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1939314905)

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