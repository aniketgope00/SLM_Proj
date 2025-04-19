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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/complement-naive-bayes-cnb-algorithm/?type%3Darticle%26id%3D458226&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
TensorFlow - How to create one hot tensor\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/tensorflow-how-to-create-one-hot-tensor/)

# Complement Naive Bayes (CNB) Algorithm

Last Updated : 10 Apr, 2023

Comments

Improve

Suggest changes

Like Article

Like

Report

Naive Bayes algorithms are a group of very popular and commonly used Machine Learning algorithms used for classification. There are many different ways the Naive Bayes algorithm is implemented like Gaussian Naive Bayes, Multinomial Naive Bayes, etc. To learn more about the basics of Naive Bayes, you can follow [this](https://www.geeksforgeeks.org/naive-bayes-classifiers/) link.

Complement Naive Bayes is somewhat an adaptation of the standard Multinomial Naive Bayes algorithm. Multinomial Naive Bayes does not perform very well on imbalanced datasets. **Imbalanced datasets** are datasets where the number of examples of some class is higher than the number of examples belonging to other classes. This means that the distribution of examples is not uniform. This type of dataset can be difficult to work with as a model may easily overfit this data in favor of the class with more number of examples.

**How CNB works:**
Complement Naive Bayes is particularly suited to work with imbalanced datasets. In complement Naive Bayes, instead of calculating the probability of an item belonging to a certain class, we calculate the probability of the item belonging to all the classes. This is the literal meaning of the word, **complement** and hence is called Complement Naive Bayes.

A step-by-step high-level overview of the algorithm (without any involved mathematics):

- For each class calculate the probability of the given instance not belonging to it.
- After calculation for all the classes, we check all the calculated values and select the smallest value.
- The smallest value (lowest probability) is selected because it is the lowest probability that it is NOT that particular class. This implies that it has the highest probability to actually belong to that class. So this class is selected.

**Note:** We don’t select the one with the highest value because we are calculating the complement of the probability. The one with the highest value is least likely to be the class that item belongs to.

Now, let us consider an example: Say, we have two classes: **Apples and Bananas** and we have to classify whether a given sentence is related to apples or bananas, given the frequency of a certain number of words. Here is a tabular representation of the simple dataset:

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| Sentence Number | Round | Red | Long | Yellow | Soft | **Class** |
| 1 | 2 | 1 | 1 | 0 | 0 | Apples |
| 2 | 1 | 1 | 0 | 9 | 5 | Bananas |
| 3 | 2 | 1 | 0 | 0 | 1 | Apples |

Total word count in class ‘Apples’ = (2+1+1) + (2+1+1) = 8
Total word count in class ‘Bananas’ = (1 + 1 + 9 + 5) = 16

So, the Probability of a sentence to belong to the class, ‘Apples’,
![\Large p(y = Apples) = {2 \over 3}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-cffe69cb4a945a96b5c961f45a99b00d_l3.svg)

Similarly, the probability of a sentence to belong to the class, ‘Bananas’,
![\Large p(y = Bananas) = {1 \over 3}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-74aa6395053ce85b8401089e5e7ebf3e_l3.svg)

In the above table, we have represented a dataset where the columns signify the frequency of words in a given sentence and then shows which class the sentence belongs to. Before we begin, you must first know about **Bayes’ Theorem**. Bayes’ Theorem is used to find the probability of an event, given that another event occurs. The formula is :
![\Large P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-029e466ef0ee0889166ad2f200eba206_l3.svg)

where A and B are events, P(A) is the probability of occurrence of A, and P(A\|B) is the probability of A to occur given that event B has already occurred. P(B), the probability of event B occurring cannot be 0 since it has already occurred. If you want to learn more about regular Naive Bayes and Bayes Theorem, you can follow [this](https://www.geeksforgeeks.org/naive-bayes-classifiers/) link.

Now let us see how Naive Bayes and Complement Naive Bayes work. The regular Naive Bayes algorithm is,

![argmax \ p(y) \bullet \prod  \frac{1}{p(\omega |y\acute{})^{f_{i}}} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8d09fdcb15d0df993eb7c7e7b84eb439_l3.svg)

where fi is the frequency of some attribute. For example, the number of times certain words occur in a sentence.

However, in complement naive Bayes, the formula is :
![\Large argmin \ p(y) \bullet \prod {1 \over p(w | \hat y)^{f_i}} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8848be78fa72073552bdf5baa86ed1e3_l3.svg)

If you take a closer look at the formulae, you will see that complement Naive Bayes is just the inverse of the regular Naive Bayes. In Naive Bayes, the class with the largest value obtained from the formula is the predicted class. So, since Complement Naive Bayes is just the inverse, the class with the smallest value obtained from the CNB formula is the predicted class.

Now, let us take an example and try to predict it using our dataset and CNB,

|     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| Round | Red | Long | Yellow | Soft | Class |
| 1 | 1 | 0 | 0 | 1 | ? |

So, we need to find,
![\Large p(y = Apples|w_1 = Round, w_2 = Red, w_3 = Soft) ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-a8e9c1e2242d9977349852101ff09c2b_l3.svg)
and
![\Large p(y = Bananas|w_1 = Round, w_2 = Red, w_3 = Soft)](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-7f636d0ed0df3296f8caa24208616c7d_l3.svg)

We need to compare both the values and select the class as the predicted class as the one with the smaller value. We have to do this also for bananas and pick the one with the smallest value. i.e., if the value for (y = Apples) is smaller, the class is predicted as Apples, and if the value for (y = Bananas) is smaller, the class is predicted as Bananas.

Using the Complement Naive Bayes Formula for both the classes,
![\Large p(y=Apples|w_1 = Round, w_2 = Red, w_3 = Soft) = {2 \over 3} \bullet {1 \over { {1 \over 16}^{1} \bullet {5 \over 16}^{1} \bullet {1 \over 16}^{1} } } \approx 6.302 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-61044488285ecb3c8103c5b19a84a0f3_l3.svg)![\Large p(y=Bananas|w_1 = Round, w_2 = Red, w_3 = Soft) = {1 \over 3} \bullet {1 \over { {1 \over 8}^{1} \bullet {1 \over 8}^{1} \bullet {2 \over 8}^{1} } } \approx 85.333 ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b1114f3b9773175c749c63928dc0028d_l3.svg)

Now, since 6.302 < 85.333, the predicted class is **Apples**.

We DON’T use the class with a higher value because a higher value means that it is more likely that a sentence with those words does NOT belong to the class. This is exactly why this algorithm is called **Complement** Naive Bayes.

**When to use CNB?**

- When the dataset is imbalanced: If the dataset on which classification is to be done is imbalanced, Multinomial and Gaussian Naive Bayes may give a low accuracy. However, Complement Naive Bayes will perform quite well and will give relatively higher accuracy.
- For text classification tasks: Complement Naive Bayes outperforms both Gaussian Naive Bayes and Multinomial Naive Bayes in text classification tasks.

**Implementation of CNB in Python:**
For this example, we will use the wine dataset which is slightly imbalanced. It determines the origin of wine from various chemical parameters. To know more about this dataset, you can check [this](https://archive.ics.uci.edu/ml/datasets/wine) link.

To evaluate our model, we will check the accuracy of the test set and the classification report of the classifier. We will use the scikit-learn library to implement the Complement Naive Bayes algorithm.

**Code:**

|     |
| --- |
| `# Import required modules `<br>`from` `sklearn.datasets ` `import` `load_wine `<br>`from` `sklearn.model_selection ` `import` `train_test_split `<br>`from` `sklearn.metrics ` `import` `accuracy_score, classification_report `<br>`from` `sklearn.naive_bayes ` `import` `ComplementNB `<br>` `<br>`# Loading the dataset  `<br>`dataset ` `=` `load_wine() `<br>`X ` `=` `dataset.data `<br>`y ` `=` `dataset.target `<br>` `<br>`# Splitting the data into train and test sets `<br>`X_train, X_test, y_train, y_test ` `=` `train_test_split(X, y, test_size ` `=` `0.15` `, random_state ` `=` `42` `) `<br>` `<br>`# Creating and training the Complement Naive Bayes Classifier `<br>`classifier ` `=` `ComplementNB() `<br>`classifier.fit(X_train, y_train) `<br>` `<br>`# Evaluating the classifier `<br>`prediction ` `=` `classifier.predict(X_test) `<br>`prediction_train ` `=` `classifier.predict(X_train) `<br>` `<br>`print` `(f` `"Training Set Accuracy : {accuracy_score(y_train, prediction_train) * 100} %\n"` `) `<br>`print` `(f` `"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} % \n\n"` `) `<br>`print` `(f` `"Classifier Report : \n\n {classification_report(y_test, prediction)}"` `)` |

```

```

```

```

**OUTPUT**

```
Training Set Accuracy : 65.56291390728477 %

Test Set Accuracy : 66.66666666666666 %

Classifier Report :

               precision    recall  f1-score   support

           0       0.64      1.00      0.78         9
           1       0.67      0.73      0.70        11
           2       1.00      0.14      0.25         7

    accuracy                           0.67        27
   macro avg       0.77      0.62      0.58        27
weighted avg       0.75      0.67      0.61        27

```

We get an accuracy of 65.56% on the training set and an accuracy of 66.66% on the test set. They are pretty much the same and are actually quite good given the quality of the dataset. This dataset is notorious for being difficult to classify with simple classifiers like the one we have used here. So the accuracy is acceptable.
**Conclusion:**
Now that you know what Complement Naive Bayes classifiers are and how they work, next time you come across an unbalanced dataset, you can try using Complement Naive Bayes.
**References:**

- [scikit-learn documentation](https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes).

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/tensorflow-how-to-create-one-hot-tensor/)

[TensorFlow - How to create one hot tensor](https://www.geeksforgeeks.org/tensorflow-how-to-create-one-hot-tensor/)

[![author](https://media.geeksforgeeks.org/auth/profile/i05p7k9001kvhn1r5fkq)](https://www.geeksforgeeks.org/user/alokesh985/)

[alokesh985](https://www.geeksforgeeks.org/user/alokesh985/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [python](https://www.geeksforgeeks.org/tag/python/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Implementing Apriori algorithm in Python\\
\\
\\
Prerequisites: Apriori AlgorithmApriori Algorithm is a Machine Learning algorithm which is used to gain insight into the structured relationships between different items involved. The most prominent practical application of the algorithm is to recommend products based on the products already present\\
\\
4 min read](https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/?ref=ml_lbp)
[Learn-One-Rule Algorithm\\
\\
\\
Prerequisite: Rule-Based Classifier Learn-One-Rule: This method is used in the sequential learning algorithm for learning the rules. It returns a single rule that covers at least some examples (as shown in Fig 1). However, what makes it really powerful is its ability to create relations among the at\\
\\
3 min read](https://www.geeksforgeeks.org/learn-one-rule-algorithm/?ref=ml_lbp)
[ML \| ECLAT Algorithm\\
\\
\\
ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) algorithm is a popular and efficient technique used for association rule mining. It is an improved alternative to the Apriori algorithm, offering better scalability and computational efficiency. Unlike Apriori, which follows a hori\\
\\
3 min read](https://www.geeksforgeeks.org/ml-eclat-algorithm/?ref=ml_lbp)
[Simple Genetic Algorithm (SGA)\\
\\
\\
Prerequisite - Genetic Algorithm Introduction : Simple Genetic Algorithm (SGA) is one of the three types of strategies followed in Genetic algorithm. SGA starts with the creation of an initial population of size N.Then, we evaluate the goodness/fitness of each of the solutions/individuals. After tha\\
\\
1 min read](https://www.geeksforgeeks.org/simple-genetic-algorithm-sga/?ref=ml_lbp)
[ML \| Find S Algorithm\\
\\
\\
Introduction : The find-S algorithm is a basic concept learning algorithm in machine learning. The find-S algorithm finds the most specific hypothesis that fits all the positive examples. We have to note here that the algorithm considers only those positive training example. The find-S algorithm sta\\
\\
4 min read](https://www.geeksforgeeks.org/ml-find-s-algorithm/?ref=ml_lbp)
[Sequential Covering Algorithm\\
\\
\\
Prerequisites: Learn-One-Rule Algorithm Sequential Covering is a popular algorithm based on Rule-Based Classification used for learning a disjunctive set of rules. The basic idea here is to learn one rule, remove the data that it covers, then repeat the same process. In this process, In this way, it\\
\\
3 min read](https://www.geeksforgeeks.org/sequential-covering-algorithm/?ref=ml_lbp)
[Multinomial Naive Bayes\\
\\
\\
Multinomial Naive Bayes is one of the variation of Naive Bayes algorithm. A classification algorithm based on Bayes' Theorem ideal for discrete data and is typically used in text classification problems. It models the frequency of words as counts and assumes each feature or word is multinomially dis\\
\\
6 min read](https://www.geeksforgeeks.org/multinomial-naive-bayes/?ref=ml_lbp)
[Bernoulli Naive Bayes\\
\\
\\
Bernoulli Naive Bayes is a subcategory of the Naive Bayes Algorithm. It is typically used when the data is binary and it models the occurrence of features using Bernoulli distribution. It is used for the classification of binary features such as 'Yes' or 'No', '1' or '0', 'True' or 'False' etc. Here\\
\\
5 min read](https://www.geeksforgeeks.org/bernoulli-naive-bayes/?ref=ml_lbp)
[Machine Learning Algorithms Cheat Sheet\\
\\
\\
This guide is a go to guide for machine learning algorithms for your interview and test preparation. This guide briefly describes key points and applications for each algorithm. This article provides an overview of key algorithms in each category, their purposes, and best use-cases. Types of Machine\\
\\
6 min read](https://www.geeksforgeeks.org/machine-learning-algorithms-cheat-sheet/?ref=ml_lbp)
[Implementing the AdaBoost Algorithm From Scratch\\
\\
\\
AdaBoost means Adaptive Boosting and it is a is a powerful ensemble learning technique that combines multiple weak classifiers to create a strong classifier. It works by sequentially adding classifiers to correct the errors made by previous models giving more weight to the misclassified data points.\\
\\
3 min read](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/complement-naive-bayes-cnb-algorithm/)

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