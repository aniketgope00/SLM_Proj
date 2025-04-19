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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/supervised-machine-learning/?type%3Darticle%26id%3D196898&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is Unsupervised Learning?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/unsupervised-learning/)

# Supervised Machine Learning

Last Updated : 02 Jan, 2025

Comments

Improve

Suggest changes

125 Likes

Like

Report

**Supervised machine learning** is a fundamental approach for machine learning and artificial intelligence. It involves training a model using labeled data, where each input comes with a corresponding correct output. The process is like a teacher guiding a student—hence the term “supervised” learning. In this article, we’ll explore the key components of supervised learning, the different types of supervised machine learning algorithms used, and some practical examples of how it works.

![supervised-machine-learning](https://media.geeksforgeeks.org/wp-content/uploads/20241022160725494723/supervised-machine-learning.webp)

Supervised Machine Learning

## What is Supervised Machine Learning?

As we explained before, **supervised learning** is a type of machine learning where a model is trained on labeled data—meaning each input is paired with the correct output. the model learns by comparing its predictions with the actual answers provided in the training data. Over time, it adjusts itself to minimize errors and improve accuracy. The goal of supervised learning is to make accurate predictions when given new, unseen data. For example, if a model is trained to recognize handwritten digits, it will use what it learned to correctly identify new numbers it hasn’t seen before.

Supervised learning can be applied in various forms, includingsupervised learning classification and supervised learning regression, making it a crucial technique in the field of artificial intelligence and supervised data mining.

A fundamental concept in supervised machine learning is learning a class from examples. This involves providing the model with examples where the correct label is known, such as learning to classify images of cats and dogs by being shown labeled examples of both. The model then learns the distinguishing features of each class and applies this knowledge to classify new images.

## **How Supervised Machine Learning Works?**

Where **supervised learning algorithm** consists of input features and corresponding output labels. The process works through:

- **Training Data:** The model is provided with a training dataset that includes input data (features) and corresponding output data (labels or target variables).
- **Learning Process:** The algorithm processes the training data, learning the relationships between the input features and the output labels. This is achieved by adjusting the model’s parameters to minimize the difference between its predictions and the actual labels.

After training, the model is evaluated using a [test dataset](https://www.geeksforgeeks.org/what-is-test-dataset-in-machine-learning/) to measure its accuracy and performance. Then the model’s performance is optimized by adjusting parameters and using techniques like [cross-validation](https://www.geeksforgeeks.org/cross-validation-machine-learning/) to [balance bias and variance](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/). This ensures the model generalizes well to new, unseen data.

_**In summary, supervised machine learning involves training a model on labeled data to learn patterns and relationships, which it then uses to make accurate predictions on new data.**_

Let’s learn how a supervised machine learning model is trained on a dataset to learn a mapping function between input and output, and then with learned function is used to make predictions on new data:

![training_testing](https://media.geeksforgeeks.org/wp-content/uploads/20230822183232/training_testing.png)

In the image above,

- **Training** phase involves feeding the algorithm labeled data, where each data point is paired with its correct output. The algorithm learns to identify patterns and relationships between the input and output data.
- **Testing** phase involves feeding the algorithm new, unseen data and evaluating its ability to predict the correct output based on the learned patterns.

## **Types of Supervised Learning in Machine Learning**

Now, Supervised learning can be applied to two main types of problems:

- [Classification](https://www.geeksforgeeks.org/getting-started-with-classification/): Where the output is a categorical variable (e.g., spam vs. non-spam emails, yes vs. no).
- [Regression](https://www.geeksforgeeks.org/regression-in-machine-learning/): Where the output is a continuous variable (e.g., predicting house prices, stock prices).

![types-of-SL](https://media.geeksforgeeks.org/wp-content/uploads/20230822183637/types-of-SL.png)

While training the model, data is usually [split in the ratio of 80:20](https://www.geeksforgeeks.org/splitting-data-for-machine-learning-models/) i.e. 80% as training data and the rest as testing data. In training data, we feed input as well as output for 80% of data. The model learns from training data only. We use different **supervised learning algorithms** (which we will discuss in detail in the next section) to build our model. Let’s first understand the classification and regression data through the table below:

![](https://media.geeksforgeeks.org/wp-content/uploads/supervised-data.png)

Both the above figures have labelled data set as follows:

- **Figure A:** It is a dataset of a shopping store that is useful in predicting whether a customer will purchase a particular product under consideration or not based on his/ her gender, age, and salary.

**Input:** Gender, Age, Salary

**Output:** Purchased i.e. 0 or 1; 1 means yes the customer will purchase and 0 means that the customer won’t purchase it.
- **Figure B:** It is a Meteorological dataset that serves the purpose of predicting wind speed based on different parameters.

**Input:** Dew Point, Temperature, Pressure, Relative Humidity, Wind Direction

**Output:** Wind Speed

> **Refer to this article for more information of** [**Types of Machine Learning**](https://www.geeksforgeeks.org/types-of-machine-learning/)

## Practical Examples of Supervised learning

Few practical examples of supervised machine learning across various industries:

- [**Fraud Detection in Banking**](https://www.geeksforgeeks.org/online-payment-fraud-detection-using-machine-learning-in-python/) **:** Utilizes supervised learning algorithms on historical transaction data, training models with labeled datasets of legitimate and fraudulent transactions to accurately predict fraud patterns.
- [**Parkinson Disease Prediction:**](https://www.geeksforgeeks.org/parkinson-disease-prediction-using-machine-learning-python/?ref=lbp) Parkinson’s disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves.
- [**Customer Churn Prediction**](https://www.geeksforgeeks.org/python-customer-churn-analysis-prediction/) **:** Uses supervised learning techniques to analyze historical customer data, identifying features associated with churn rates to predict customer retention effectively.
- [**Cancer cell classification:**](https://www.geeksforgeeks.org/ml-cancer-cell-classification-using-scikit-learn/?ref=lbp) Implements supervised learning for cancer cells based on their features, and identifying them if they are ‘malignant’ or ‘benign.
- [**Stock Price Prediction:**](https://www.geeksforgeeks.org/stock-price-prediction-using-machine-learning-in-python/) Applies supervised learning to predict a signal that indicates whether buying a particular stock will be helpful or not.

## **Supervised Machine Learning Algorithms**

**Supervised learning** can be further divided into several different types, each with its own unique characteristics and applications. Here are some of the most common types of supervised learning algorithms:

- [**Linear Regression**](https://www.geeksforgeeks.org/ml-linear-regression/): Linear regression is a type of supervised learning regression algorithm that is used to predict a continuous output value. It is one of the simplest and most widely used algorithms in supervised learning.
- [**Logistic Regression**](https://www.geeksforgeeks.org/understanding-logistic-regression/): Logistic regression is a type of supervised learning classification algorithm that is used to predict a binary output variable.
- [**Decision Trees**](https://www.geeksforgeeks.org/decision-tree/): Decision tree is a tree-like structure that is used to model decisions and their possible consequences. Each internal node in the tree represents a decision, while each leaf node represents a possible outcome.
- [**Random Forests**](https://www.geeksforgeeks.org/random-forest-regression-in-python/): Random forests again are made up of multiple decision trees that work together to make predictions. Each tree in the forest is trained on a different subset of the input features and data. The final prediction is made by aggregating the predictions of all the trees in the forest.
- [**Support Vector Machine(SVM)**](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) : The SVM algorithm creates a hyperplane to segregate n-dimensional space into classes and identify the correct category of new data points. The extreme cases that help create the hyperplane are called support vectors, hence the name Support Vector Machine.
- [**K-Nearest Neighbors**](https://www.geeksforgeeks.org/k-nearest-neighbours/) **(KNN) :** KNN works by finding k training examples closest to a given input and then predicts the class or value based on the majority class or average value of these neighbors. The performance of KNN can be influenced by the choice of k and the distance metric used to measure proximity.
- [**Gradient Boosting**](https://www.geeksforgeeks.org/ml-gradient-boosting/): Gradient Boosting combines weak learners, like [decision trees](https://www.geeksforgeeks.org/decision-tree/), to create a strong model. It iteratively builds new models that correct errors made by previous ones.
- [**Naive Bayes Algorithm**](https://www.geeksforgeeks.org/naive-bayes-classifiers/): The Naive Bayes algorithm is a supervised machine learning algorithm based on applying [Bayes’ Theorem](https://www.geeksforgeeks.org/bayes-theorem/) with the “naive” assumption that features are independent of each other given the class label.

Let’s summarize the **supervised machine learning algorithms** in table:

| Algorithm | Regression,<br>Classification | Purpose | Method | Use Cases |
| --- | --- | --- | --- | --- |
| Linear Regression | Regression | Predict continuous output values | Linear equation minimizing sum of squares of residuals | Predicting continuous values |
| Logistic Regression | Classification | Predict binary output variable | Logistic function transforming linear relationship | Binary classification tasks |
| Decision Trees | Both | Model decisions and outcomes | Tree-like structure with decisions and outcomes | Classification and Regression tasks |
| Random Forests | Both | Improve classification and regression accuracy | Combining multiple decision trees | Reducing overfitting, improving prediction accuracy |
| SVM | Both | Create hyperplane for classification or predict continuous values | Maximizing margin between classes or predicting continuous values | Classification and Regression tasks |
| KNN | Both | Predict class or value based on k closest neighbors | Finding k closest neighbors and predicting based on majority or average | Classification and Regression tasks, sensitive to noisy data |
| Gradient Boosting | Both | Combine weak learners to create strong model | Iteratively correcting errors with new models | Classification and Regression tasks to improve prediction accuracy |
| Naive Bayes | Classification | Predict class based on feature independence assumption | Bayes’ theorem with feature independence assumption | Text classification, spam filtering, sentiment analysis, medical |

These **types of supervised learning in machine learning** vary based on the problem you’re trying to solve and the dataset you’re working with. In classification problems, the task is to assign inputs to predefined classes, while regression problems involve predicting numerical outcomes.

## Training a Supervised Learning Model: Key Steps

The goal of Supervised learning is to generalize well to unseen data. Training a model for supervised learning involves several crucial steps, each designed to prepare the model to make accurate predictions or decisions based on labeled data. Below are the key steps involved in training a model for supervised machine learning:

1. [**Data Collection and Preprocessing**](https://www.geeksforgeeks.org/data-preprocessing-in-data-mining/): Gather a labeled dataset consisting of input features and target output labels. Clean the data, handle missing values, and scale features as needed to ensure high quality for **supervised learning** algorithms.
2. [**Splitting the Data**](https://www.geeksforgeeks.org/splitting-data-for-machine-learning-models/): Divide the data into [**training set** (80%) and the **test set** (20%)](https://www.geeksforgeeks.org/training-data-vs-testing-data/).
3. **Choosing the Model**: Select appropriate algorithms based on the problem type. This step is crucial for effective **supervised learning** in AI.
4. **Training the Model**: Feed the model input data and output labels, allowing it to learn patterns by adjusting internal parameters.
5. [**Evaluating the Model**](https://www.geeksforgeeks.org/machine-learning-model-evaluation/): Test the trained model on the unseen test set and assess its performance using various metrics.
6. [**Hyperparameter Tuning**](https://www.geeksforgeeks.org/hyperparameter-tuning/) **:** Adjust settings that control the training process (e.g., learning rate) using techniques like grid search and cross-validation.
7. **Final Model Selection and Testing**: Retrain the model on the complete dataset using the best hyperparameters testing its performance on the test set to ensure readiness for deployment.
8. **Model Deployment**: Deploy the validated model to make predictions on new, unseen data.

By following these steps, supervised learning models can be effectively trained to tackle various tasks, from **learning a class from examples** to making predictions in real-world applications.

## A **dvantages and Disadvantages of Supervised Learning**

### **Advantages of Supervised Learning**

The power of **supervised learning** lies in its ability to accurately predict patterns and make data-driven decisions across a variety of applications. Here are some advantages of **supervised learning** listed below:

- **Supervised learning** excels in accurately predicting patterns and making data-driven decisions.
- **Labeled training data** is crucial for enabling **supervised learning models** to learn input-output relationships effectively.
- **Supervised machine learning** encompasses tasks such as **supervised learning classification** and **supervised learning regression**.
- Applications include complex problems like image recognition and natural language processing.
- Established evaluation metrics (accuracy, precision, recall, F1-score) are essential for assessing **supervised learning model** performance.
- Advantages of **supervised learning** include creating complex models for accurate predictions on new data.
- **Supervised learning** requires substantial labeled training data, and its effectiveness hinges on data quality and representativeness.

### **Disadvantages of Supervised Learning**

Despite the benefits of **supervised learning methods**, there are notable **disadvantages of supervised learning**:

1. [**Overfitting**](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/): Models can overfit training data, leading to poor performance on new data due to capturing noise in **supervised machine learning**.
2. [**Feature Engineering**](https://www.geeksforgeeks.org/what-is-feature-engineering/): Extracting relevant features is crucial but can be time-consuming and requires domain expertise in **supervised learning applications**.
3. **Bias in Models:** Bias in the training data may result in unfair predictions in **supervised learning algorithms**.
4. **Dependence on Labeled Data**: **Supervised learning** relies heavily on labeled training data, which can be costly and time-consuming to obtain, posing a challenge for **supervised learning techniques**.

## Conclusion

Supervised learning is a powerful branch of machine learning that revolves around learning a class from examples provided during training. By using supervised learning algorithms, models can be trained to make predictions based on labeled data. The effectiveness of supervised machine learning lies in its ability to generalize from the training data to new, unseen data, making it invaluable for a variety of applications, from image recognition to financial forecasting.

Understanding the types of supervised learning algorithms and the dimensions of supervised machine learning is essential for choosing the appropriate algorithm to solve specific problems. As we continue to explore the different types of supervised learning and refine these supervised learning techniques, the impact of supervised learning in machine learning will only grow, playing a critical role in advancing AI-driven solutions.

Suggested Quiz


10 Questions


In supervised machine learning, what is the primary purpose of using labeled data during the training phase?

- To introduce randomness in the model

- To allow the model to learn the relationship between inputs and outputs

- To increase the complexity of the model

- To reduce the amount of data required for training


Explanation:

Because in Supervised Machine Learning Algorithm without properly labeled data, machine learning algorithms would struggle to understand the underlying patterns and make accurate predictions.

Which of the following algorithms is primarily used for classification tasks in supervised learning?

- Linear Regression

- K-Nearest Neighbors

- Gradient Descent

- Principal Component Analysis


Explanation:

KNN is a supervised Learning Algorithm used for classification task.

What is the main challenge associated with overfitting in supervised learning models?

- The model performs well on unseen data

- The model captures noise in the training data

- The model requires less labeled data

- The model has a high bias


Explanation:

The training data size is too small and does not contain enough data to accurate represent all input data values which lead to overfitting and noise

In the context of supervised learning, what is the role of hyperparameter tuning?

- To randomly select features for the model

- To adjust the model's parameters that are not learned during training

- To eliminate the need for labeled data

- To reduce the size of the training dataset


Explanation:

Hyperparameters are often used to tune the performance of a model, and they can have a significant impact on the model’s accuracy, generalization, and other metrics.

Which of the following metrics is NOT commonly used to evaluate the performance of a supervised learning model?

- Accuracy

- Recall

- Mean Squared Error

- Entropy


Explanation:

Entropy evaluation metrics is majorly used in Decision Trees

What is the primary difference between classification and regression tasks in supervised learning?

- Classification predicts categorical outcomes, while regression predicts continuous outcomes

- Classification requires more data than regression

- Regression is always more complex than classification

- Classification can only be performed on numeric data


Explanation:

Classification is used for categorical output and regression is used for continuous outcome

In supervised learning, what is the purpose of splitting the dataset into training and test sets?

- To increase the overall size of the dataset

- To evaluate the model's performance on unseen data

- To ensure all data points are used in training

- To reduce the complexity of the model


Explanation:

The purpose of splitting data is to test the model performance after training.

Which supervised learning algorithm is particularly known for its ability to handle both classification and regression tasks?

- Decision Trees

- K-Nearest Neighbors

- Naive Bayes

- Linear Regression


Explanation:

Decision trees can used for both classification and regression task

What is the main advantage of using ensemble methods like Random Forests in supervised learning?

- They simplify the model-building process

- They reduce the likelihood of overfitting by combining multiple models

- They require less computational power

- They eliminate the need for labeled data


Explanation:

Random Forests average the predictions of many decision trees, which decreases the chances of overfitting the data.

Which of the following statements correctly describes the concept of bias in supervised learning models?

- Bias refers to the model's inability to learn from the training data

- Bias is the error introduced by approximating a real-world problem with a simplified model

- Bias only affects the performance of regression models

- Bias is always desirable in model training


Explanation:

In Supervised learning bias refers to the error introduced by approximating a real-world problem (which may be complex) with a simplified model.

![](https://media.geeksforgeeks.org/auth-dashboard-uploads/sucess-img.png)

Quiz Completed Successfully


Your Score :   2/10

Accuracy :  0%

Login to View Explanation


1/10 1/10
< Previous

Next >


Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/unsupervised-learning/)

[What is Unsupervised Learning?](https://www.geeksforgeeks.org/unsupervised-learning/)

[![author](https://media.geeksforgeeks.org/auth/profile/71uvm7pynx4cnk0rtuk6)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

[mohit gupta\_omg :)](https://www.geeksforgeeks.org/user/mohit%20gupta_omg%20:)/)

Follow

125

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Computer Subject](https://www.geeksforgeeks.org/category/computer-subject/)
- [Data Science](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Artificial Intelligence Tutorial \| AI Tutorial\\
\\
\\
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. It involves the development of algorithms and computer programs that can perform tasks that typically require human intelligence such as visual perception, speech\\
\\
7 min read](https://www.geeksforgeeks.org/artificial-intelligence/)
[What is Artificial Intelligence(AI)?\\
\\
\\
Artificial Intelligence (AI) refers to the technology that allows machines and computers to replicate human intelligence. It enables systems to perform tasks that require human-like decision-making, such as learning from data, identifying patterns, making informed choices and solving complex problem\\
\\
13 min read](https://www.geeksforgeeks.org/what-is-artificial-intelligence-ai/)
[History of AI\\
\\
\\
The term Artificial Intelligence (AI) is already widely used in everything from smartphones to self-driving cars. AI has come a long way from science fiction stories to practical uses. Yet What is artificial intelligence and how did it go from being an idea in science fiction to a technology that re\\
\\
7 min read](https://www.geeksforgeeks.org/evolution-of-ai/)

## Types of AI

- [Types of Artificial Intelligence (AI)\\
\\
\\
Artificial Intelligence refers to something which is made by humans or non-natural things and Intelligence means the ability to understand or think. AI is not a system but it is implemented in the system. There are many different types of AI, each with its own strengths and weaknesses. This article\\
\\
6 min read](https://www.geeksforgeeks.org/types-of-artificial-intelligence/)

* * *

- [Types of AI Based on Capabilities: An In-Depth Exploration\\
\\
\\
Artificial Intelligence (AI) is not just a single entity but encompasses a wide range of systems and technologies with varying levels of capabilities. To understand the full potential and limitations of AI, it's important to categorize it based on its capabilities. This article delves into the diffe\\
\\
5 min read](https://www.geeksforgeeks.org/types-of-ai-based-on-capabilities-an-in-depth-exploration/)

* * *

- [Types of AI Based on Functionalities\\
\\
\\
Artificial Intelligence (AI) has become an integral part of modern technology, influencing everything from how we interact with our devices to how businesses operate. However, AI is not a monolithic concept; it can be classified into different types based on its functionalities. Understanding these\\
\\
7 min read](https://www.geeksforgeeks.org/types-of-ai-based-on-functionalities/)

* * *


[Agents in AI\\
\\
\\
An AI agent is a software program that can interact with its surroundings, gather information, and use that information to complete tasks on its own to achieve goals set by humans. For instance, an AI agent on an online shopping platform can recommend products, answer customer questions, and process\\
\\
9 min read](https://www.geeksforgeeks.org/agents-artificial-intelligence/)

## Problem Solving in AI

- [Search Algorithms in AI\\
\\
\\
Artificial Intelligence is the study of building agents that act rationally. Most of the time, these agents perform some kind of search algorithm in the background in order to achieve their tasks. A search problem consists of: A State Space. Set of all possible states where you can be.A Start State.\\
\\
10 min read](https://www.geeksforgeeks.org/search-algorithms-in-ai/)

* * *

- [Uninformed Search Algorithms in AI\\
\\
\\
Uninformed search algorithms is also known as blind search algorithms, are a class of search algorithms that do not use any domain-specific knowledge about the problem being solved. Uninformed search algorithms rely on the information provided in the problem definition, such as the initial state, ac\\
\\
8 min read](https://www.geeksforgeeks.org/uniformed-search-algorithms-in-ai/)

* * *

- [Informed Search Algorithms in Artificial Intelligence\\
\\
\\
Informed search algorithms, also known as heuristic search algorithms, are an essential component of Artificial Intelligence (AI). These algorithms use domain-specific knowledge to improve the efficiency of the search process, leading to faster and more optimal solutions compared to uninformed searc\\
\\
10 min read](https://www.geeksforgeeks.org/informed-search-algorithms-in-artificial-intelligence/)

* * *

- [Local Search Algorithm in Artificial Intelligence\\
\\
\\
Local search algorithms are essential tools in artificial intelligence and optimization, employed to find high-quality solutions in large and complex problem spaces. Key algorithms include Hill-Climbing Search, Simulated Annealing, Local Beam Search, Genetic Algorithms, and Tabu Search. Each of thes\\
\\
4 min read](https://www.geeksforgeeks.org/local-search-algorithm-in-artificial-intelligence/)

* * *

- [Adversarial Search Algorithms in Artificial Intelligence (AI)\\
\\
\\
Adversarial search algorithms are the backbone of strategic decision-making in artificial intelligence, it enables the agents to navigate competitive scenarios effectively. This article offers concise yet comprehensive advantages of these algorithms from their foundational principles to practical ap\\
\\
15+ min read](https://www.geeksforgeeks.org/adversarial-search-algorithms/)

* * *

- [Constraint Satisfaction Problems (CSP) in Artificial Intelligence\\
\\
\\
Constraint Satisfaction Problems (CSP) play a crucial role in artificial intelligence (AI) as they help solve various problems that require decision-making under certain constraints. CSPs represent a class of problems where the goal is to find a solution that satisfies a set of constraints. These pr\\
\\
14 min read](https://www.geeksforgeeks.org/constraint-satisfaction-problems-csp-in-artificial-intelligence/)

* * *


## Knowledge, Reasoning and Planning in AI

- [How do knowledge representation and reasoning techniques support intelligent systems?\\
\\
\\
In artificial intelligence (AI), knowledge representation and reasoning (KR&R) stands as a fundamental pillar, crucial for enabling machines to emulate complex decision-making and problem-solving abilities akin to those of humans. This article explores the intricate relationship between KR&R\\
\\
5 min read](https://www.geeksforgeeks.org/knowledge-representation-and-reasoning-techniques-support-intelligent-systems/)

* * *

- [First-Order Logic in Artificial Intelligence\\
\\
\\
First-order logic (FOL) is also known as predicate logic. It is a foundational framework used in mathematics, philosophy, linguistics, and computer science. In artificial intelligence (AI), FOL is important for knowledge representation, automated reasoning, and NLP. FOL extends propositional logic b\\
\\
3 min read](https://www.geeksforgeeks.org/first-order-logic-in-artificial-intelligence/)

* * *

- [Types of Reasoning in Artificial Intelligence\\
\\
\\
In today's tech-driven world, machines are being designed to mimic human intelligence and actions. One key aspect of this is reasoning, a logical process that enables machines to conclude, make predictions, and solve problems just like humans. Artificial Intelligence (AI) employs various types of re\\
\\
6 min read](https://www.geeksforgeeks.org/types-of-reasoning-in-artificial-intelligence/)

* * *

- [What is the Role of Planning in Artificial Intelligence?\\
\\
\\
Artificial Intelligence (AI) is reshaping the future, playing a pivotal role in domains like intelligent robotics, self-driving cars, and smart cities. At the heart of AI systemsâ€™ ability to perform tasks autonomously is AI planning, which is critical in guiding AI systems to make informed decisions\\
\\
7 min read](https://www.geeksforgeeks.org/what-is-the-role-of-planning-in-artificial-intelligence/)

* * *

- [Representing Knowledge in an Uncertain Domain in AI\\
\\
\\
Artificial Intelligence (AI) systems often operate in environments where uncertainty is a fundamental aspect. Representing and reasoning about knowledge in such uncertain domains is crucial for building robust and intelligent systems. This article explores the various methods and techniques used in\\
\\
6 min read](https://www.geeksforgeeks.org/representing-knowledge-in-an-uncertain-domain-in-ai/)

* * *


## Learning in AI

- [Supervised Machine Learning\\
\\
\\
Supervised machine learning is a fundamental approach for machine learning and artificial intelligence. It involves training a model using labeled data, where each input comes with a corresponding correct output. The process is like a teacher guiding a studentâ€”hence the term "supervised" learning. I\\
\\
12 min read](https://www.geeksforgeeks.org/supervised-machine-learning/)

* * *

- [What is Unsupervised Learning?\\
\\
\\
Unsupervised learning is a branch of machine learning that deals with unlabeled data. Unlike supervised learning, where the data is labeled with a specific category or outcome, unsupervised learning algorithms are tasked with finding patterns and relationships within the data without any prior knowl\\
\\
8 min read](https://www.geeksforgeeks.org/unsupervised-learning/)

* * *

- [Semi-Supervised Learning in ML\\
\\
\\
Today's Machine Learning algorithms can be broadly classified into three categories, Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Casting Reinforced Learning aside, the primary two categories of Machine Learning problems are Supervised and Unsupervised Learning. The basic\\
\\
4 min read](https://www.geeksforgeeks.org/ml-semi-supervised-learning/)

* * *

- [Reinforcement Learning\\
\\
\\
Reinforcement Learning (RL) is a branch of machine learning that focuses on how agents can learn to make decisions through trial and error to maximize cumulative rewards. RL allows machines to learn by interacting with an environment and receiving feedback based on their actions. This feedback comes\\
\\
6 min read](https://www.geeksforgeeks.org/what-is-reinforcement-learning/)

* * *

- [Self-Supervised Learning (SSL)\\
\\
\\
In this article, we will learn a major type of machine learning model which is Self-Supervised Learning Algorithms. Usage of these algorithms has increased widely in the past times as the sizes of the model have increased up to billions of parameters and hence require a huge corpus of data to train\\
\\
8 min read](https://www.geeksforgeeks.org/self-supervised-learning-ssl/)

* * *

- [Introduction to Deep Learning\\
\\
\\
Deep Learning is transforming the way machines understand, learn, and interact with complex data. Deep learning mimics neural networks of the human brain, it enables computers to autonomously uncover patterns and make informed decisions from vast amounts of unstructured data. Deep Learning leverages\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-deep-learning/)

* * *

- [Natural Language Processing (NLP) - Overview\\
\\
\\
Natural Language Processing (NLP) is a field that combines computer science, artificial intelligence and language studies. It helps computers understand, process and create human language in a way that makes sense and is useful. With the growing amount of text data from social media, websites and ot\\
\\
9 min read](https://www.geeksforgeeks.org/natural-language-processing-overview/)

* * *

- [Computer Vision Tutorial\\
\\
\\
Computer Vision is a branch of Artificial Intelligence (AI) that enables computers to interpret and extract information from images and videos, similar to human perception. It involves developing algorithms to process visual data and derive meaningful insights. Why Learn Computer Vision?High Demand\\
\\
8 min read](https://www.geeksforgeeks.org/computer-vision/)

* * *

- [Artificial Intelligence in Robotics\\
\\
\\
Artificial Intelligence (AI) in robotics is one of the most groundbreaking technological advancements, revolutionizing how robots perform tasks. What was once a futuristic concept from space operas, the idea of "artificial intelligence robots" is now a reality, shaping industries globally. Unlike ea\\
\\
10 min read](https://www.geeksforgeeks.org/artificial-intelligence-in-robotics/)

* * *


## Generative AI

- [Generative Adversarial Network (GAN)\\
\\
\\
Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow and his colleagues in 2014. GANs are a class of neural networks that autonomously learn patterns in the input data to generate new examples resembling the original dataset. GAN's architecture consists of two neural networks: Ge\\
\\
12 min read](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)

* * *

- [Variational AutoEncoders\\
\\
\\
Variational Autoencoders (VAEs) are generative models in machine learning (ML) that create new data similar to the input they are trained on. Along with data generation they also perform common autoencoder tasks like denoising. Like all autoencoders VAEs consist of: Encoder: Learns important pattern\\
\\
8 min read](https://www.geeksforgeeks.org/variational-autoencoders/)

* * *

- [What are Diffusion Models?\\
\\
\\
Diffusion models are a powerful class of generative models that have gained prominence in the field of machine learning and artificial intelligence. They offer a unique approach to generating data by simulating the diffusion process, which is inspired by physical processes such as heat diffusion. Th\\
\\
6 min read](https://www.geeksforgeeks.org/what-are-diffusion-models/)

* * *

- [Transformers in Machine Learning\\
\\
\\
Transformer is a neural network architecture used for performing machine learning tasks particularly in natural language processing (NLP) and computer vision. In 2017 Vaswani et al. published a paper " Attention is All You Need" in which the transformers architecture was introduced. The article expl\\
\\
4 min read](https://www.geeksforgeeks.org/getting-started-with-transformers/)

* * *


Like125

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/supervised-machine-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1048404266.1745055437&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=1405381295)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=za6y7pfimd1k)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=icqhid16eqrq)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=invisible&cb=aucc4gsu05kh)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)