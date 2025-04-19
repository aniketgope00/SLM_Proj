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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/decision-tree-introduction-example/?type%3Darticle%26id%3D167523&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Telephone Number\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/telephone-number/)

# Decision Tree in Machine Learning

Last Updated : 08 Apr, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

In the [decision trees](https://www.geeksforgeeks.org/decision-tree/) article, we discussed how decision trees model decisions through a tree-like structure, where internal nodes represent feature tests, branches represent decision rules, and leaf nodes contain the final predictions. This basic understanding is crucial for building and interpreting decision trees, which are widely used for classification and regression tasks.

**Now, let’s take this understanding a step further** and dive into how decision trees are implemented in machine learning. We will explore how to train a decision tree model, make predictions, and evaluate its performance

## Why Decision Tree Structure in ML?

A decision tree is a [supervised learning](https://www.geeksforgeeks.org/supervised-machine-learning/) algorithm used for both [classification](https://www.geeksforgeeks.org/getting-started-with-classification/) and [regression](https://www.geeksforgeeks.org/regression-in-machine-learning/) tasks. It models decisions as a **tree-like structure** where internal nodes represent **attribute tests**, branches represent **attribute values** and leaf nodes represent **final decisions or predictions**. Decision trees are versatile, interpretable, and widely used in machine learning for predictive modeling.

Now we have covered about the very basic of decision tree but its very important to understand the intuition behind the decision tree so lets move towards it.

## Intuition behind the Decision Tree

Here’s an example to make it simple to understand the intuition of decision tree:

Imagine you’re deciding whether to buy an umbrella:

1. **Step 1 – Ask a Question (Root Node):**

_Is it raining?_

If yes, you might decide to buy an **umbrella**. If no, you move to the next question.
2. **Step 2 – More Questions (Internal Nodes):**

If it’s not raining, you might ask:

_Is it likely to rain later?_

If yes, you buy an umbrella; if no, you don’t.
3. **Step 3 – Decision (Leaf Node):**

Based on your answers, you either buy or skip the umbrella

## Approach in Decision Tree

Decision tree uses the **tree** representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree. We can represent any Boolean function on discrete attributes using the decision tree.

![predicting_whether_a_customer_will_buy_a_product](https://media.geeksforgeeks.org/wp-content/uploads/20250408153824016146/predicting_whether_a_customer_will_buy_a_product.webp)

Decision Tree

Let’s consider a **decision tree** for predicting whether a customer will buy a product based on **age**, **income** and **previous purchases**: Here’s how the decision tree works:

**1\. Root Node (Income)**

**First Question**: **“Is the person’s income greater than $50,000?”**

- **If Yes**, proceed to the next question.
- **If No**, predict “No Purchase” (leaf node).

**2\. Internal Node (Age)**:

**If the person’s income is greater than $50,000**, ask: **“Is the person’s age above 30?”**

- **If Yes**, proceed to the next question.
- **If No**, predict “No Purchase” (leaf node).

**3\. Internal Node (Previous Purchases)**:

- If the person is above 30 and has made previous purchases, predict “Purchase” (leaf node).
- If the person is above 30 and has not made previous purchases, predict “No Purchase” (leaf node).

![tree_1_customer_demographics](https://media.geeksforgeeks.org/wp-content/uploads/20250408153952530850/tree_1_customer_demographics.webp)

Decision making with 2 Decision Tree

**Example:** Predicting Whether a Customer Will Buy a Product Using Two Decision Trees

### **Tree 1:** Customer Demographics

First tree asks two questions:

1\. “Income > $50,000?”

- If **Yes**, Proceed to the next question.
- If **No**, “No Purchase”

2\. “Age > 30?”

- **Yes**: “Purchase”
- **No**: “No Purchase”

### Tree 2: Previous Purchases

“Previous Purchases > 0?”

- **Yes**: “Purchase”
- **No**: “No Purchase”

### **Combining Trees: Final Prediction**

Once we have predictions from both trees, we can combine the results to make a final prediction. If **Tree 1** predicts “Purchase” and **Tree 2** predicts “No Purchase”, the final prediction might be “Purchase” or “No Purchase” depending on the weight or confidence assigned to each tree. This can be decided based on the problem context.

## Information Gain and Gini Index in Decision Tree

Till now we have discovered the basic intuition and approach of how decision tree works, so lets just move to the attribute selection measure of decision tree.

We have two popular attribute selection measures used:

1. **Information Gain**
2. **Gini Index**

### **1\. Information Gain:**

Information Gain tells us how useful a question (or feature) is for splitting data into groups. It measures how much the uncertainty decreases after the split. A good question will create clearer groups and the feature with the highest Information Gain is chosen to make the decision.

For example, if we split a dataset of people into “ **Young**” and “ **Old**” based on age, and all young people bought the product while all old people did not, the Information Gain would be high because the split perfectly separates the two groups with no uncertainty left

- Suppose SSS is a set of instances, AA Ais an attribute, SvSv Svis the subset of SS S _,_ vv vrepresents an individual value that the attribute AA Acan take and Values (AAA) is the set of all possible values of AAA, then

Gain(S,A)=Entropy(S)–∑vA∣Sv∣∣S∣.Entropy(Sv)Gain(S, A) = Entropy(S) – \\sum\_{v}^{A}\\frac{\\left \| S\_{v} \\right \|}{\\left \| S \\right \|}. Entropy(S\_{v})



Gain(S,A)=Entropy(S)–∑vA​∣S∣∣Sv​∣​.Entropy(Sv​)
- **Entropy:** is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy more the information content.

For example, if a dataset has an equal number of “ **Yes**” and “ **No**” outcomes (like 3 people who bought a product and 3 who didn’t), the entropy is high because it’s uncertain which outcome to predict. But if all the outcomes are the same (all “Yes” or all “No”), the entropy is 0, meaning there is no uncertainty left in predicting the outcome

Suppose SS Sis a set of instances, AA Ais an attribute, SvSv Svis the subset of SS Swith AA A= vvv, and Values (AAA) is the set of all possible values of AAA, then

Gain(S,A)=Entropy(S)–∑vϵValues(A)∣Sv∣∣S∣.Entropy(Sv)Gain(S, A) = Entropy(S) – \\sum\_{v \\epsilon Values(A)}\\frac{\\left \| S\_{v} \\right \|}{\\left \| S \\right \|}. Entropy(S\_{v})  Gain(S,A)=Entropy(S)–∑vϵValues(A)​∣S∣∣Sv​∣​.Entropy(Sv​)

**Example:**

```
For the set X = {a,a,a,b,b,b,b,b}
Total instances: 8
Instances of b: 5
Instances of a: 3
```

Entropy H(X)=\[(38)log⁡238+(58)log⁡258\]=−\[0.375(−1.415)+0.625(−0.678)\]=−(−0.53−0.424)=0.954\\begin{aligned}\\text{Entropy } H(X) & =\\left \[ \\left ( \\frac{3}{8} \\right )\\log\_{2}\\frac{3}{8} + \\left ( \\frac{5}{8} \\right )\\log\_{2}\\frac{5}{8} \\right \]\\\& = -\[0.375 (-1.415) + 0.625 (-0.678)\] \\\& = -(-0.53-0.424) \\\& = 0.954\\end{aligned}Entropy H(X)​=\[(83​)log2​83​+(85​)log2​85​\]=−\[0.375(−1.415)+0.625(−0.678)\]=−(−0.53−0.424)=0.954​

### **Building Decision Tree using Information Gain the essentials:**

- Start with all training instances associated with the root node
- Use info gain to choose which attribute to label each node with
- _Note:_ No root-to-leaf path should contain the same discrete attribute twice
- Recursively construct each subtree on the subset of training instances that would be classified down that path in the tree.
- If all positive or all negative training instances remain, the label that node “yes” or “no” accordingly
- If no attributes remain, label with a majority vote of training instances left at that node
- If no instances remain, label with a majority vote of the parent’s training instances.

**Example:** Now, let us draw a Decision Tree for the following data using Information gain. **Training set: 3 features and 2 classes**

| X | Y | Z | C |
| --- | --- | --- | --- |
| 1 | 1 | 1 | I |
| 1 | 1 | 0 | I |
| 0 | 0 | 1 | II |
| 1 | 0 | 0 | II |

Here, we have 3 features and 2 output classes. To build a decision tree using Information gain. We will take each of the features and calculate the information for each feature. ![](https://media.geeksforgeeks.org/wp-content/uploads/tr4.png)

**Split on feature X**

![](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20210317184559/y-attribute.png)

**Split on feature Y**

![](https://media.geeksforgeeks.org/wp-content/uploads/20230202163851/z-attribute.png)

**Split on feature Z**

From the above images, we can see that the information gain is **maximum** when we make a split on feature Y. So, for the root node best-suited feature is feature Y. Now we can see that while splitting the dataset by feature Y, the child contains a pure subset of the target variable. So we don’t need to further split the dataset. The final tree for the above dataset would look like this:

![](https://media.geeksforgeeks.org/wp-content/uploads/tr6.png)

### **2\. Gini Index**

- Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with a lower Gini index should be preferred.
- Sklearn supports “Gini” criteria for Gini Index and by default, it takes “gini” value.

For example, if we have a group of people where all bought the product **(100% “Yes”)**, the Gini Index is **0**, indicating perfect purity. But if the group has an equal mix of “Yes” and “No”, the Gini Index would be **0.5**, showing higher **impurity or uncertainty.**

Formula for Gini Index is given by :

Gini=1–∑i=1npi2Gini = 1 – \\sum\_{i=1}^{n} p\_i^2Gini=1–∑i=1n​pi2​

**Some additional features and characteristics of the Gini Index are:**

1. It is calculated by summing the squared probabilities of each outcome in a distribution and subtracting the result from 1.
2. A lower Gini Index indicates a more homogeneous or pure distribution, while a higher Gini Index indicates a more heterogeneous or impure distribution.
3. In decision trees, the Gini Index is used to evaluate the quality of a split by measuring the difference between the impurity of the parent node and the weighted impurity of the child nodes.
4. Compared to other impurity measures like entropy, the Gini Index is faster to compute and more sensitive to changes in class probabilities.
5. One disadvantage of the Gini Index is that it tends to favour splits that create equally sized child nodes, even if they are not optimal for classification accuracy.
6. In practice, the choice between using the Gini Index or other impurity measures depends on the specific problem and dataset, and often requires experimentation and tuning.

### Understanding Decision Tree with Real life use case:

Till now we have understand about the attributes and components of decision tree. Now lets jump to a real life use case in which how decision tree works step by step.

**Step 1. Start with the Whole Dataset**

We begin with all the data, which is treated as the root node of the decision tree.

**Step 2. Choose the Best Question (Attribute)**

Pick the best question to divide the dataset. For example, ask: _“What is the outlook?”_

- Possible answers: **Sunny**, **Cloudy**, or **Rainy**.

**Step 3. Split the Data into Subsets**

Divide the dataset into groups based on the question:

- If **Sunny**, go to one subset.
- If **Cloudy**, go to another subset.
- If **Rainy**, go to the last subset.

**Step 4. Split Further if Needed (Recursive Splitting)**

For each subset, ask another question to refine the groups. For example:

- If the **Sunny** subset is mixed, ask: _“Is the humidity high or normal?”_
  - High humidity → “Swimming”.
  - Normal humidity → “Hiking”.

**Step 5. Assign Final Decisions (Leaf Nodes)**

When a subset contains only one activity, stop splitting and assign it a label:

- **Cloudy** → “Hiking”.
- **Rainy** → “Stay Inside”.
- **Sunny + High Humidity** → “Swimming”.
- **Sunny + Normal Humidity** → “Hiking”.

**Step 6. Use the Tree for Predictions**

To predict an activity, follow the branches of the tree:

- Example: If the outlook is **Sunny** and the humidity is **High**, follow the tree:
  - Start at _Outlook_.
  - Take the branch for **Sunny**.
  - Then go to _Humidity_ and take the branch for **High Humidity**.
  - Result: “Swimming”.

This is how a decision tree works: by splitting data step-by-step based on the best questions and stopping when a clear decision is made!

Decision trees provide a powerful and intuitive approach to decision-making, offering both simplicity and interpretability, making them an invaluable tool in the field of machine learning for various classification and regression tasks.”

### Frequently Asked Questions (FAQs)

#### 1\. What are the major issues in decision tree learning?

> Major issues in decision tree learning include overfitting, sensitivity to small data changes, and limited generalization. Ensuring proper pruning, tuning, and handling imbalanced data can help mitigate these challenges for more robust decision tree models.

#### 2\. How does decision tree help in decision making?

> Decision trees aid decision-making by representing complex choices in a hierarchical structure. Each node tests specific attributes, guiding decisions based on data values. Leaf nodes provide final outcomes, offering a clear and interpretable path for decision analysis in machine learning.

#### 3\. What is the maximum depth of a decision tree?

> The maximum depth of a decision tree is a hyperparameter that determines the maximum number of levels or nodes from the root to any leaf. It controls the complexity of the tree and helps prevent overfitting.

#### 4\. What is the concept of decision tree?

> A decision tree is a supervised learning algorithm that models decisions based on input features. It forms a tree-like structure where each internal node represents a decision based on an attribute, leading to leaf nodes representing outcomes.

#### 5\. What is entropy in decision tree?

> In decision trees, entropy is a measure of impurity or disorder within a dataset. It quantifies the uncertainty associated with classifying instances, guiding the algorithm to make informative splits for effective decision-making.

#### 6\. What are the Hyperparameters of decision tree?

> 1. **Max Depth:** Maximum depth of the tree.
> 2. **Min Samples Split:** Minimum samples required to split an internal node.
> 3. **Min Samples Leaf:** Minimum samples required in a leaf node.
> 4. **Criterion:** The function used to measure the quality of a split

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/telephone-number/)

[Telephone Number](https://www.geeksforgeeks.org/telephone-number/)

[A](https://www.geeksforgeeks.org/user/Abhishek%20Sharma%2044/)

[Abhishek Sharma 44](https://www.geeksforgeeks.org/user/Abhishek%20Sharma%2044/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [DSA](https://www.geeksforgeeks.org/category/dsa/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[KNN vs Decision Tree in Machine Learning\\
\\
\\
There are numerous machine learning algorithms available, each with its strengths and weaknesses depending on the scenario. Factors such as the size of the training data, the need for accuracy or interpretability, training time, linearity assumptions, the number of features, and whether the problem\\
\\
5 min read](https://www.geeksforgeeks.org/knn-vs-decision-tree-in-machine-learning/?ref=ml_lbp)
[What is Machine Learning?\\
\\
\\
Machine learning is a branch of artificial intelligence that enables algorithms to uncover hidden patterns within datasets. It allows them to predict new, similar data without explicit programming for each task. Machine learning finds applications in diverse fields such as image and speech recogniti\\
\\
9 min read](https://www.geeksforgeeks.org/ml-machine-learning/?ref=ml_lbp)
[Regression in machine learning\\
\\
\\
Regression in machine learning refers to a supervised learning technique where the goal is to predict a continuous numerical value based on one or more independent features. It finds relationships between variables so that predictions can be made. we have two types of variables present in regression\\
\\
5 min read](https://www.geeksforgeeks.org/regression-in-machine-learning/?ref=ml_lbp)
[What is Data Acquisition in Machine Learning?\\
\\
\\
Data acquisition, or DAQ, is the cornerstone of machine learning. It is essential for obtaining high-quality data for model training and optimizing performance. Data-centric techniques are becoming more and more important across a wide range of industries, and DAQ is now a vital tool for improving p\\
\\
12 min read](https://www.geeksforgeeks.org/what-is-data-acquisition-in-machine-learning/?ref=ml_lbp)
[Hypothesis in Machine Learning\\
\\
\\
The concept of a hypothesis is fundamental in Machine Learning and data science endeavours. In the realm of machine learning, a hypothesis serves as an initial assumption made by data scientists and ML professionals when attempting to address a problem. Machine learning involves conducting experimen\\
\\
6 min read](https://www.geeksforgeeks.org/ml-understanding-hypothesis/?ref=ml_lbp)
[Machine Learning - Learning VS Designing\\
\\
\\
In this article, we will learn about Learning and Designing and what are the main differences between them. In Machine learning, the term learning refers to any process by which a system improves performance by using experience and past data. It is kind of an iterative process and every time the sys\\
\\
3 min read](https://www.geeksforgeeks.org/machine-learning-learning-vs-designing/?ref=ml_lbp)
[How does Machine Learning Works?\\
\\
\\
Machine Learning is a subset of Artificial Intelligence that uses datasets to gain insights from it and predict future values. It uses a systematic approach to achieve its goal going through various steps such as data collection, preprocessing, modeling, training, tuning, evaluation, visualization,\\
\\
7 min read](https://www.geeksforgeeks.org/how-does-machine-learning-works/?ref=ml_lbp)
[Classification vs Regression in Machine Learning\\
\\
\\
Classification and regression are two primary tasks in supervised machine learning, where key difference lies in the nature of the output: classification deals with discrete outcomes (e.g., yes/no, categories), while regression handles continuous values (e.g., price, temperature). Both approaches re\\
\\
5 min read](https://www.geeksforgeeks.org/ml-classification-vs-regression/?ref=ml_lbp)
[How to Detect Outliers in Machine Learning\\
\\
\\
In machine learning, an outlier is a data point that stands out a lot from the other data points in a set. The article explores the fundamentals of outlier and how it can be handled to solve machine learning problems. Table of Content What is an outlier?Outlier Detection Methods in Machine LearningT\\
\\
7 min read](https://www.geeksforgeeks.org/machine-learning-outlier/?ref=ml_lbp)
[What is Test Dataset in Machine Learning?\\
\\
\\
In Machine Learning, a Test Dataset plays a crucial role in evaluating the performance of your trained model. In this blog, we will delve into the intricacies of test dataset in machine learning, its significance, and its indispensable role in the data science lifecycle. What is Test Dataset in Mach\\
\\
4 min read](https://www.geeksforgeeks.org/what-is-test-dataset-in-machine-learning/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/decision-tree-introduction-example/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1966827382.1745055863&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1122665068)