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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/decision-tree-algorithms/?type%3Darticle%26id%3D1093408&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Hinge-loss & relationship with Support Vector Machines\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/hinge-loss-relationship-with-support-vector-machines/)

# Decision Tree Algorithms

Last Updated : 30 Jan, 2025

Comments

Improve

Suggest changes

5 Likes

Like

Report

[Decision trees](https://www.geeksforgeeks.org/decision-tree/) are widely used machine learning algorithm and can be used for both classification and regression tasks. **These models work by splitting data into subsets based on feature and this splitting is called as decision making and each leaf node tells us prediction**. This splitting creates a tree-like structure. They are easy to interpret and visualize for understanding the decision-making process.

In machine learning we have various types of decision trees and in this article we will explore them that so that we can use them in machine learning for various task.

## Types of Decision Tree Algorithms

The different decision tree algorithms are listed below:

- ID3(Iterative Dichotomiser 3)
- C4.5
- CART(Classification and Regression Trees)
- CHAID (Chi-Square Automatic Interaction Detection)
- MARS(Multivariate Adaptive Regression Splines)
- Conditional Inference Trees

Each of them have their own working and advantages that we will learn now.

### **1\. ID3 (Iterative Dichotomiser 3)**

[ID3](https://www.geeksforgeeks.org/iterative-dichotomiser-3-id3-algorithm-from-scratch/) is a classic decision tree algorithm commonly used for classification tasks. **It works by greedily choosing the feature that maximizes the information gain at each node**. It calculates **entropy** and **information gain** for each feature and selects the feature with the highest information gain for splitting.

Entropy denoted by H(D) for dataset D, is calculated using the formula:

H(D)=Σi=1npilog2(pi)H(D) = \\Sigma^n \_{i=1}\\;p\_{i}\\; log\_{2}(p\_{i})H(D)=Σi=1n​pi​log2​(pi​)

**Information gain** quantifies the reduction in entropy after splitting the dataset on a feature:

InformationGain=H(D)−Σv=1V∣Dv∣∣D∣H(Dv)Information\\; Gain = H(D) - \\Sigma^V\_{v=1} \\frac{\|D\_{v}\|}{\|D\|}H (D\_{v})InformationGain=H(D)−Σv=1V​∣D∣∣Dv​∣​H(Dv​)

ID3 recursively splits the dataset using the feature with the highest information gain until all examples in a node belong to the same class or no features remain to split. After the tree is constructed it prune branches that don't significantly improve accuracy to reduce overfitting.

**ID3 has limitations:** it tends to overfit the training data and cannot directly handle continuous attributes. These issues are addressed by other algorithms like **C4.5** and **CART**.

> For its implementation you can refer to the article: [Iterative Dichotomiser 3 (ID3) Algorithm From Scratch](https://www.geeksforgeeks.org/iterative-dichotomiser-3-id3-algorithm-from-scratch/)

### **2\. C4.5**

C4.5 uses a modified version of information gain called the **gain ratio** to reduce the bias towards features with many values. The gain ratio is computed by dividing the **information gain** by the **intrinsic information** which measures the amount of data required to describe an attribute’s values:

GainRatio=SplitgainGaininformationGain Ratio = \\frac{Split\\; gain}{Gain\\;\\;information}GainRatio=GaininformationSplitgain​

It addresses several limitations of ID3 including its inability to handle continuous attributes and its tendency to overfit the training set. It handles continuous attributes by first sorting the attribute values and then selecting the midpoint between adjacent values as a potential split point. The split that maximizes information gain or gain ratio is chosen.

It can also generate rules from the decision tree by converting each path from the root to a leaf into a rule, which can be used to make predictions on new data.

This algorithm improves accuracy and reduces overfitting by using gain ratio and **post-pruning**. While effective for both discrete and continuous attributes, C4.5 may still struggle with noisy data and large feature sets.

**C4.5 has limitations:**

- It can be prone to overfitting especially in noisy datasets even if uses pruning techniques.
- Performance may degrade when dealing with datasets that have many features.

### **3\. CART (Classification and Regression Trees)**

[CART](https://www.geeksforgeeks.org/cart-classification-and-regression-tree-in-machine-learning/) is a widely used decision tree algorithm that is used for **classification** and **regression** tasks.

1\. For **classification** CART splits data based on the Gini impurity which measures the likelihood of incorrectly classified randomly selected data. The feature that minimizes the Gini impurity is selected for splitting at each node.

**Gini Impurity (for Classification)** :

Gini(D)=1−Σi=1npi2Gini(D) = 1 - \\Sigma^n \_{i=1}\\; p^2\_{i}Gini(D)=1−Σi=1n​pi2​

where pip\_ipi​​ is the probability of class iii in dataset DDD.

2\. For **regression** CART builds regression trees by minimizing the **variance** of the target variable within each subset. The split that reduces the variance the most is chosen.

To reduce overfitting, CART uses **cost-complexity pruning** after tree construction. This method involves minimizing a cost function that combines the impurity and tree complexity by adding a complexity parameter to the impurity measure. It builds **binary trees** where each internal node has exactly two child nodes simplifying the splitting process and making the resulting tree easier to interpret.

> For its implementation you can refer to the article: [Implementing CART (Classification And Regression Tree) in Python](https://www.geeksforgeeks.org/implementing-cart-classification-and-regression-tree-in-python/)

### **4\. CHAID (Chi-Square Automatic Interaction Detection)**

CHAID uses [**chi-square tests**](https://www.geeksforgeeks.org/chi-square-test/) to determine the best splits especially for **categorical variables**. It recursively divides the data into smaller subsets until each subset contains only data points of the same class or within a specified range of values. It chooses feature for splitting with highest chi-squared statistic indicating the strong relationship with the target variable. This approach is particularly useful for analyzing large datasets with many categorical features.

**Chi-Square Statistic Formula:**

X2=Σ(Oi−Ei)2EiX^2 = \\Sigma \\frac{(O\_{i} - E\_{i})^2}{E\_{i}}X2=ΣEi​(Oi​−Ei​)2​

Where

- Oi represents the observed frequency

Ei represents the expected frequency in each category.

It compares the observed distribution to the expected distribution to determine if there is a significant difference.

CHAID can be applied to both classification and regression tasks. In **classification,** algorithm assigns a class label to new data points by following the tree from the root to a leaf node with leaf node’s class label being assigned to data. In **regression** it predicts the target variable by averaging the values at the leaf node.

### **5\. MARS (Multivariate Adaptive Regression Splines)**

MARS is an extension of the CART algorithm. It uses **splines** to model **non-linear relationships** between variables. It constructs a piecewise linear model where the relationship between the input and output variables is linear but with variable slopes at different points, known as **knots**. It automatically selects and positions these knots based on the data distribution and the need to capture non-linearities.

**Basis Functions**: Each basis function in MARS is a simple linear function defined over a range of the predictor variable. The function is described as:

h(x)={x−tifx>tt−xifx≤t}h(x) = \\Bigg \\{ x - t \\;\\; if \\; x>t \\\ t-x \\;\\; if x \\leq t \\Bigg\\} h(x)={x−tifx>tt−xifx≤t}

Where

- x is a predictor variable
- t is the knot function.

**Knot Function**: The **knots** are the points where the [piecewise linear functions](https://www.geeksforgeeks.org/piecewise-function/) connect. MARS places these knots to best represent the data's non-linear structure.

MARS begins by constructing a model with a single piece and then applies **forward stepwise selection** to iteratively add pieces that reduce the error. The process continues until the model reaches a desired complexity. It is particularly effective for modeling complex relationships in data and is widely used in regression tasks.

### 6\. Conditional Inference Trees

[Conditional Inference Trees](https://www.geeksforgeeks.org/conditional-inference-trees-in-r-programming/) uses statistical tests to choose splits based on the relationship between features and the target variable. **It use permutation tests to select the feature that best splits the data while minimizing bias.**

The algorithm follows a recursive approach. At each node it evaluates the statistical significance of potential splits using tests like the **Chi-squared test** for categorical features and the **F-test** for continuous features. The feature with the strongest relationship to the target is selected for the split. The process continues until the data cannot be further split or meets predefined stopping criteria.

## Summarizing all Algorithms

Here’s a short summary of all decision tree algorithms we have learned so far:

1. **ID3**: Uses **information gain** to split data and works well for classification but it is prone to overfitting and struggles with continuous data.
2. **C4.5**: Advance version of ID3 with **gain ratio** for both discrete and continuous data but struggle with noisy data.
3. **CART**: Used for both classification and regression task. It minimizes **Gini impurity** for classification and **MSE** for regression with **pruning technique** to prevent overfitting.
4. **CHAID**: Uses **chi-square tests** for splitting and is effective for large categorical datasets but not for continuous data.
5. **MARS**: Extended version of CART using **piecewise linear functions** to model non-linear relationships but it is computationally expensive.
6. **Conditional Inference Trees**: Uses **statistical hypothesis testing** for unbiased splits and handles various data types but it is slower than others.

Decision tree algorithms offer interpretable approach for both classification and regression tasks. While each algorithm brings its own strengths understanding their underlying mechanisms is crucial for selecting the best algorithm for a given problem for better accuracy of model.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/hinge-loss-relationship-with-support-vector-machines/)

[Hinge-loss & relationship with Support Vector Machines](https://www.geeksforgeeks.org/hinge-loss-relationship-with-support-vector-machines/)

[S](https://www.geeksforgeeks.org/user/sirvinaysy60t/)

[sirvinaysy60t](https://www.geeksforgeeks.org/user/sirvinaysy60t/)

Follow

5

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Geeks Premier League](https://www.geeksforgeeks.org/category/geeksforgeeks-initiatives/geeks-premier-league/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Python scikit-module](https://www.geeksforgeeks.org/tag/python-scikit-module/)
- [Geeks Premier League 2023](https://www.geeksforgeeks.org/tag/geeks-premier-league-2023/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Decision Tree\\
\\
\\
Decision tree is a simple diagram that shows different choices and their possible results helping you make decisions easily. This article is all about what decision trees are, how they work, their advantages and disadvantages and their applications. Understanding Decision TreeA decision tree is a gr\\
\\
5 min read](https://www.geeksforgeeks.org/decision-tree/)
[Tree Based Machine Learning Algorithms\\
\\
\\
Tree-based algorithms are a fundamental component of machine learning, offering intuitive decision-making processes akin to human reasoning. These algorithms construct decision trees, where each branch represents a decision based on features, ultimately leading to a prediction or classification. By\\
\\
14 min read](https://www.geeksforgeeks.org/tree-based-machine-learning-algorithms/)
[Root Finding Algorithm\\
\\
\\
Root-finding algorithms are tools used in mathematics and computer science to locate the solutions, or "roots," of equations. These algorithms help us find solutions to equations where the function equals zero. For example, if we have an equation like f(x) = 0, a root-finding algorithm will help us\\
\\
8 min read](https://www.geeksforgeeks.org/root-finding-algorithm/)
[Algorithm definition and meaning\\
\\
\\
Algorithm can be defined as - A set of finite rules or instructions to be followed in calculations or other problem-solving operations. An algorithm can be expressed using pseudocode or flowcharts. Properties of Algorithm: An algorithm has several important properties that include: Input: An algorit\\
\\
3 min read](https://www.geeksforgeeks.org/algorithm-definition-and-meaning/)
[Search Algorithms in AI\\
\\
\\
Artificial Intelligence is the study of building agents that act rationally. Most of the time, these agents perform some kind of search algorithm in the background in order to achieve their tasks. A search problem consists of: A State Space. Set of all possible states where you can be.A Start State.\\
\\
10 min read](https://www.geeksforgeeks.org/search-algorithms-in-ai/)
[Apriori Algorithm\\
\\
\\
Apriori Algorithm is a foundational method in data mining used for discovering frequent itemsets and generating association rules. Its significance lies in its ability to identify relationships between items in large datasets which is particularly valuable in market basket analysis. For example, if\\
\\
5 min read](https://www.geeksforgeeks.org/apriori-algorithm/)
[Decision Theory in AI\\
\\
\\
Decision theory is a foundational concept in Artificial Intelligence (AI), enabling machines to make rational and informed decisions based on available data. It combines principles from mathematics, statistics, economics, and psychology to model and improve decision-making processes. In AI, decision\\
\\
8 min read](https://www.geeksforgeeks.org/decision-theory-in-ai/)
[Approximation Algorithms\\
\\
\\
Overview :An approximation algorithm is a way of dealing with NP-completeness for an optimization problem. This technique does not guarantee the best solution. The goal of the approximation algorithm is to come as close as possible to the optimal solution in polynomial time. Such algorithms are call\\
\\
3 min read](https://www.geeksforgeeks.org/approximation-algorithms/)
[Types of Algorithms in Pattern Recognition\\
\\
\\
At the center of pattern recognition are various algorithms designed to process and classify data. These can be broadly classified into statistical, structural and neural network-based methods. Pattern recognition algorithms can be categorized as: Statistical Pattern Recognition â€“ Based on probabili\\
\\
5 min read](https://www.geeksforgeeks.org/types-of-algorithms-in-pattern-recognition/)
[ML \| Find S Algorithm\\
\\
\\
Introduction : The find-S algorithm is a basic concept learning algorithm in machine learning. The find-S algorithm finds the most specific hypothesis that fits all the positive examples. We have to note here that the algorithm considers only those positive training example. The find-S algorithm sta\\
\\
4 min read](https://www.geeksforgeeks.org/ml-find-s-algorithm/)

Like5

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/decision-tree-algorithms/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=407932961.1745055875&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=50643979)

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