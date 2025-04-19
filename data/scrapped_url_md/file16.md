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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/machine-learning-models/?type%3Darticle%26id%3D1215030&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Flowchart for basic Machine Learning models\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/)

# Machine Learning Models

Last Updated : 08 Aug, 2024

Comments

Improve

Suggest changes

5 Likes

Like

Report

Machine Learning models are very powerful resources that automate multiple tasks and make them more accurate and efficient. ML handles new data and scales the growing demand for technology with valuable insight. It improves the performance over time. This cutting-edge technology has various benefits such as faster processing or response, enhancement of decision-making, and specialized services. In this article, we will discuss _**Machine Learning Models, their types, How Machine Learning works, Real-world examples of ML Models, and the Future of Machine Learning Models.**_

![Machine-Learning-Model](https://media.geeksforgeeks.org/wp-content/uploads/20240410181640/Machine-Learning-Model.webp)Machine Leraning Models

A model of machine learning is a set of programs that can be used to find the pattern and make a decision from an unseen dataset. These days [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/) (Natural language Processing) uses the machine learning model to recognize the unstructured text into usable data and insights. You may have heard about image recognition which is used to identify objects such as boy, girl, mirror, car, dog, etc. A model always requires a dataset to perform various tasks during training. In training duration, we use a [machine learning algorithm](https://www.geeksforgeeks.org/machine-learning-algorithms/) for the optimization process to find certain patterns or outputs from the [dataset](https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/) based upon tasks.

Table of Content

- [Types of Machine Learning Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#types-of-machine-learning-models)
- [1\. Supervised Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#1-supervised-models)

  - [1.1 Classification](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#11-classification)
  - [1.2 Regression](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#12-regression)

- [2\. Unsupervised Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#2-unsupervised-models)

  - [2.1 Clustering](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#21-clustering)
  - [2.2 Dimensionality Reduction](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#22-dimensionality-reduction)
  - [2.3 Anomaly Detection](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#23-anomaly-detection)

- [3\. Semi-Supervised Model](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#3-semisupervised-model)

  - [3.1 Generative Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#31-generative-semisupervised-learning)
  - [3.2 Graph-based Semi-Supervised Learning](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#32-graphbased-semisupervised-learning)

- [4\. Reinforcement learning Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#4-reinforcement-learning-models)

  - [4.1 Value-based learning:](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#41-valuebased-learning)
  - [4.2 Policy-based learning:](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#42-policybased-learning)

- [Deep Learning](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#deep-learning)
- [How Machine Learning Works?](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#how-machine-learning-works)
- [Advanced Machine Learning Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#advanced-machine-learning-models)
- [Real-world examples of ML Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#realworld-examples-of-ml-models)
- [Future of Machine Learning Models](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#future-of-machine-learning-models)
- [Conclusion](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp#conclusion)

## Types of Machine Learning Models

Machine learning models can be broadly categorized into four main paradigms based on the type of data and learning goals:

## **1\. Supervised Models**

Supervised learning is the study of algorithms that use labeled data in which each data instance has a known category or value to which it belongs. This results in the model to discover the relationship between the input features and the target outcome.

### 1.1 Classification

The classifier algorithms are designed to indicate whether a new data point belongs to one or another among several predefined classes. Imagine when you are organising emails into spam or inbox, categorising images as cat or dog, or predicting whether a loan applicant is a credible borrower. In the classification models, there is a learning process by the use of labeled examples from each category. In this process, they discover the correlations and relations within the data that help to distinguish class one from the other classes. After learning these patterns, the model is then capable of assigning these class labels to unseen data points.

**Common Classification Algorithms:**

- **Logistic Regression:** A very efficient technique for the classification problems of binary nature (two types, for example, spam/not spam).
- [**Support Vector Machine (SVM)**](https://www.geeksforgeeks.org/support-vector-machine-algorithm/) **:** Good for tasks like classification, especially when the data has a large number of features.
- **Decision Tree:** Constructs a decision tree having branches and proceeds to the class predictions through features.
- [**Random Forest**](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/) **:** The model generates an "ensemble" of decision trees that ultimately raise the accuracy and avoid overfitting (meaning that the model performs great on the training data but lousily on unseen data).
- [**K-Nearest Neighbors (KNN):**](https://www.geeksforgeeks.org/k-nearest-neighbours/) Assigns a label of the nearest neighbors for a given data point.

### 1.2 Regression

Regression algorithms are about forecasting of a continuous output variable using the input features as their basis. This value could be anything such as predicting real estate prices or stock market trends to anticipating customer churn (how likely customers stay) and sales forecasting. Regression models make the use of features to understand the relationship among the continuous features and the output variable. That is, they use the pattern that is learned to determine the value of the new data points.

**Common Regression Algorithms**

- [**Linear Regression:**](https://www.geeksforgeeks.org/ml-linear-regression/) Fits depth of a line to the data to model for the relationship between features and the continuous output.
- **Polynomial Regression:** Similiar to linear regression but uses more complex polynomial functions such as quadratic, cubic, etc, for accommodating non-linear relationships of the data.
- [**Decision Tree Regression:**](https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/) Implements a decision tree-based algorithm that predicts a continuous output variable from a number of branching decisions.
- [**Random Forest Regression**](https://www.geeksforgeeks.org/random-forest-regression-in-python/) **:** Creates one from several decision trees to guarantee error-free and robust regression prediction results.
- [**Support Vector Regression (SVR):**](https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/) Adjusts the Support Vector Machine ideas for regression tasks, where we are trying to find one hyperplane that most closely reflects continuous output data.

## **2\. Unsupervised Model** s

Unsupervised learning involves a difficult task of working with data which is not provided with pre-defined categories or label.

### 2.1 [Clustering](https://www.geeksforgeeks.org/clustering-in-machine-learning/)

Visualize being given a basket of fruits with no labels on them. The fruits clustering algorithms are to group them according to the inbuilt similarities. Techniques like K-means clustering are defined by exact number of clusters ("red fruits" and "green fruits") and then each data point (fruit) is assigned to the cluster with the highest similarity within based on features (color, size, texture). Contrary to this, hierarchical clustering features construction of hierarchy of clusters which makes it more easy to study the system of groups. Spatial clustering algorithm Density-Based Spatial Clustering of Applications with Noise (DBSCAN) detects groups of high-density data points, even in those areas where there is a lack of data or outliers.

### 2.2 [Dimensionality Reduction](https://www.geeksforgeeks.org/dimensionality-reduction/)

Sometimes it is difficult to both visualize and analyze the data when you have a large feature space (dimensions). The purpose of dimensionality reduction methods is to decrease the dimensions needed to maintain the key features. Dimensions of greatest importance are identified by [principal component analysis (PCA)](https://www.geeksforgeeks.org/principal-component-analysis-pca/), which is the reason why data is concentrated in fewer dimensions with the highest variations. This speeds up model training as well as offers a chance for more efficient visualization. LDA (Linear Discriminant Analysis) also resembles PCA but it is made for classification tasks where it concentrates on dimensions that can differentiate the present classes in the dataset.

### 2.3 [Anomaly Detection](https://www.geeksforgeeks.org/machine-learning-for-anomaly-detection/)

Unsupervised learning can also be applied to find those data points which greatly differ than the majorities. The statistics model may identify these outliers, or anomalies as signaling of errors, fraud or even something unusual. Local Outlier Factor (LOF) makes a comparison of a given data point's local density with those surrounding it. It then flags out the data points with significantly lower densities as outliers or potential anomalies. Isolation Forest is the one which uses different approach, which is to recursively isolate data points according to their features. Anomalies usually are simple to contemplate as they often necessitate fewer steps than an average normal point.

## **3\. Semi-Supervised** Model

Besides, supervised learning is such a kind of learning with labeled data that unsupervised learning, on the other hand, solves the task where there is no labeled data. Lastly, semi-supervised learning fills the gap between the two. It reveals the strengths of both approaches by training using data sets labeled along with unlabeled one. This is especially the case when labeled data might be sparse or prohibitively expensive to acquire, while unlabeled data is undoubtedly available in abundance.

### 3.1 Generative Semi-Supervised Learning

Envision having a few pictures of cats with labels and a universe of unlabeled photos. The big advantage of generative semi-supervised learning is its utilization of such a scenario. It exploits a generative model to investigate the unlabeled pictures and discover the orchestrating factors that characterize the data. This technique can then be used to generate the new synthetic data points that have the same features with the unlabeled data. The synthetic data is then labeled with the pseudo-labels that the generative model has interpreted from the data. This approach combines the existing labeled data with the newly generated labeled data to train the final model which is likely to perform better than the previous model that was trained with only the limited amount of the original labeled data.

### 3.2 Graph-based Semi-Supervised Learning

This process makes use of the relationships between data points and propagates labels to unmarked ones via labeled ones. Picture a social network platform where some of the users have been marked as fans of sports (labeled data). Cluster-based methods can analyze the links between users (friendships) and even apply this information to infer that if a user is connected to someone with a "sports" label then this user might also be interested in sports (unbiased labels with propagated label). While links and the entire structure of the network are also important for the distribution of labels. This method is beneficial when the data points are themselves connected to each other and this connection can be exploiting during labelling of new data.

## 4\. Reinforcement learning Models

Reinforcement learning takes a dissimilar approach from [supervised learning](https://www.geeksforgeeks.org/supervised-machine-learning/) and unsupervised learning. Different from supervised learning or just plain discovery of hidden patterns, reinforcement learning adopt an agent as it interacts with the surrounding and learns. This agent is a learning one which develops via experiment and error, getting rewarded for the desired actions and punished for the undesired ones. The main purpose is to help players play the game that can result in the highest rewards.

### 4.1 Value-based learning:

Visualize a robot trying to find its way through a maze. It has neither a map nor instructions, but it gets points for consuming the cheese at the end and fails with deduction of time when it runs into a wall. Value learning is an offshoot of predicting the anticipated future reward of taking a step in a particular state. For example, the algorithm Q-learning will learn a Q-value for each state-action combination. This Q-value is the expected reward for that action at that specific state. Through a repetitive process of assessing the state, gaining rewards, and updating the Q-values the agent manages to determine that which actions are most valuable in each state and eventually guides it to the most rewarding path. In contrast, [SARSA (State-Action-Reward-State-Action)](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/) looks at the value of the succeeding state-action pair that influences the exploration strategy.

### 4.2 Policy-based learning:

In contrast to the value-based learning, where we are learning a specific value for each state-action pair, in policy-based learning we are trying to directly learn a policy which maps states to actions. This policy in essence commands the agent to act in different situations as specified by the way it is written. Actor-Critic is a common approach that combines two models: an actor that retrains the policy and a critic that retrains the value function (just like value-based methods). The actor witnesses the critic's feedback which updates the policy that the actor uses for better decision making. Proximal Policy Optimization (PPO) is a specific policy-based method which focuses on high variance issues that complicate early policy-based learning methods.

## **Deep Learning**

Deep learning is a subfield of machine learning that utilizes artificial neural networks with multiple layers to achieve complex pattern recognition. These networks are particularly effective for tasks involving large amounts of data, such as image recognition and natural language processing.

1. **Artificial Neural Networks (ANNs)**\- This is a popular model that refers to the structure and function of the human brain. It consists of interconnected nodes based on various layers and is used for various ML tasks.
2. **Convolutional Neural Networks (CNNs) -** A [CNN](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/) is a [deep learning](https://www.geeksforgeeks.org/introduction-deep-learning/) model that automates the spatial hierarchies of features from input data. This model is commonly used in image recognition and classification.
3. **Recurrent Neural Networks (RNNs) -** This model is designed for the processing of sequential data. It enables the memory input which is known for [Neural network architectures](https://www.geeksforgeeks.org/how-to-decide-neural-network-architecture/).
4. **Long Short-Term Memory Networks (LSTMs) -** This model is comparatively similar to [Recurrent Neural Networks](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) and allows learners to learn the long-term dependencies from sequential data.

## How Machine Learning Works?

1. **Model Represntation:** Machine Learning Models are represented by mathematical functions that map input data to output predictions. These functions can take various forms, such as linear equations, decision trees , or complex neural networks.
2. **Learning Algorithm:** The learning algorithm is the main part of behind the model's ability to learn from data. It adjusts the parameters of the model's mathematical function iteratively during the training phase to minimize the difference between the model's prediction and the actual outcomes in the training data .
3. **Training Data:** Training data is used to teach the model to make accurate predictions. It consists of input features(e.g variables, attributes) and corresponding output labels(in supervised learning) or is unalabeled(in supervised learning). During training , the model analyzes the patterns in the training data to update its parameters accordingly.
4. **Objective Function:** The objective function, also known as the [loss function](https://www.geeksforgeeks.org/ml-common-loss-functions/), measures the difference between the model's predictions and the actual outcomes in the training data. The goal during training is to minimize this function, effectively reducing the errors in the model's predictions.
5. **Optimization Process:** Optimization is the process of finding the set of model parameters that minimize the objective function. This is typically achieved using optimization algorithms such as gradient descent, which iteratively adjusts the model's parameters in the direction that reduces the objective function.
6. **Generalization:** Once the model is trained, it is evaluated on a separate set of data called the validation or test set to assess its performance on new, unseen data. The model's ability to perform well on data it hasn't seen before is known as generalization.
7. **Final Output:** After training and validation, the model can be used to make predictions or decisions on new, unseen data. This process, known as inference, involves applying the trained model to new input data to generate predictions or classifications.

## Advanced Machine Learning Models

- **Neural Networks**: You must have heard about deep neural network which helps solve complex problems of data. It is made up of interconnected nodes of multiple layers which we also call neurons. Many things have been successful from this model such as image recognition, [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/), and [speech recognition](https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/).
- **Convolutional Neural Networks (CNNs)**: This is a type of model that is built in the framework of a neural network and it is made to handle data that are of symbolic type, like images. From this model, the hierarchy of spatial features can be determined.
- **Recurrent Neural Networks (RNNs)**: These can be used to process data that is sequentially ordered, such as reading categories or critical language. These networks are built with loops in their architectures that allow them to store information over time.
- **Long Short-Term Memory Networks (LSTMs)**: [LSTMs](https://www.geeksforgeeks.org/understanding-of-lstm-networks/), which are a type of RNNs, recognize long-term correlation objects. These models do a good job of incorporating information organized into long categories.
- **Generative Adversarial Networks (GANs)**: [GAN](https://www.geeksforgeeks.org/generative-adversarial-network-gan/) s are a type of neural networks that generate data by studying two networks over time. A product generates network data, while a determination attempts to distinguish between real and fake samples.
- **Transformer Models**: This model become popular in [natural language processing](https://www.geeksforgeeks.org/natural-language-processing-overview/). These models process input data over time and capture long-range dependencies.

## Real-world examples of ML Models

The ML model uses predictive analysis to maintain the growth of various Industries-

- **Financial Services**: Banks and financial institutions are using machine learning models to provide better services to their customers. Using intelligent algorithms, they understand customers' investment preferences, speed up the loan approval process, and receive alerts for non-ordinary transactions.
- **Healthcare**: In medicine, ML models are helpful in disease prediction, treatment recommendations, and prognosis. For example, physicians can use a machine learning model to predict the right cold medicine for a patient.
- **Manufacturing Industry**: In the manufacturing sector, ML has made the production process more smooth and optimized. For example, Machine Learning is being used in automated production lines to increase production efficiency and ensure manufacturing quality.
- **Commercial Sector**: In the marketing and marketing sector, ML models analyze huge data and predict production trends. This helps in understanding the marketing system and the products can be customized for their target customers.

## Future of Machine Learning Models

There are several important aspects to consider when considering the challenges and future of machine learning models. One challenge is that there are not enough resources and tools available to contextualize large data sets. Additionally, [machine learning](https://www.geeksforgeeks.org/machine-learning/) models need to be updated and restarted to understand new data patterns.

In the future, another challenge for machine learning may be to collect and aggregate collections of data between different existing technology versions. This can be important for scientific development along with promoting the discovery of new possibilities. Finally, good strategy, proper resources, and technological advancement are important concepts for success in developing machine learning models. To address all these challenges, appropriate time and attention is required to further expand machine learning capabilities.

## Conclusion

We first saw the introduction of machine learning in which we know what a model is and what is the benefit of implementing it in our system. Then look at the history and evolution of machine learning along with the selection criteria to decide which model to use specifically. Next, we read [data preparation](https://www.geeksforgeeks.org/what-is-data-preparation/) where you can read all the steps. Then we researched advanced model that has future benefits but some challenges can also be faced but the ML model is a demand for the future.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/)

[Flowchart for basic Machine Learning models](https://www.geeksforgeeks.org/flowchart-for-basic-machine-learning-models/)

[T](https://www.geeksforgeeks.org/user/tapasghotana/)

[tapasghotana](https://www.geeksforgeeks.org/user/tapasghotana/)

Follow

5

Improve

Article Tags :

- [AI-ML-DS Blogs](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science-blogs/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)

### Similar Reads

[Machine Learning Tutorial\\
\\
\\
Machine learning is a subset of Artificial Intelligence (AI) that enables computers to learn from data and make predictions without being explicitly programmed. If you're new to this field, this tutorial will provide a comprehensive understanding of machine learning, its types, algorithms, tools, an\\
\\
8 min read](https://www.geeksforgeeks.org/machine-learning/)

## Prerequisites for Machine Learning

[Python for Machine Learning\\
\\
\\
Welcome to "Python for Machine Learning," a comprehensive guide to mastering one of the most powerful tools in the data science toolkit. Python is widely recognized for its simplicity, versatility, and extensive ecosystem of libraries, making it the go-to programming language for machine learning. I\\
\\
6 min read](https://www.geeksforgeeks.org/python-for-machine-learning/)
[SQL for Machine Learning\\
\\
\\
Integrating SQL with machine learning can provide a powerful framework for managing and analyzing data, especially in scenarios where large datasets are involved. By combining the structured querying capabilities of SQL with the analytical and predictive capabilities of machine learning algorithms,\\
\\
6 min read](https://www.geeksforgeeks.org/sql-for-machine-learning/)

## Getting Started with Machine Learning

[Advantages and Disadvantages of Machine Learning\\
\\
\\
Machine learning (ML) has revolutionized industries, reshaped decision-making processes, and transformed how we interact with technology. As a subset of artificial intelligence ML enables systems to learn from data, identify patterns, and make decisions with minimal human intervention. While its pot\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-machine-learning/)
[Why ML is Important ?\\
\\
\\
Machine learning (ML) has become a cornerstone of modern technology, revolutionizing industries and reshaping the way we interact with the world. As a subset of artificial intelligence (AI), ML enables systems to learn and improve from experience without being explicitly programmed. Its importance s\\
\\
4 min read](https://www.geeksforgeeks.org/why-ml-is-important/)
[Real- Life Examples of Machine Learning\\
\\
\\
Machine learning plays an important role in real life, as it provides us with countless possibilities and solutions to problems. It is used in various fields, such as health care, financial services, regulation, and more. Importance of Machine Learning in Real-Life ScenariosThe importance of machine\\
\\
13 min read](https://www.geeksforgeeks.org/real-life-applications-of-machine-learning/)
[What is the Role of Machine Learning in Data Science\\
\\
\\
In today's world, the collaboration between machine learning and data science plays an important role in maximizing the potential of large datasets. Despite the complexity, these concepts are integral in unraveling insights from vast data pools. Let's delve into the role of machine learning in data\\
\\
9 min read](https://www.geeksforgeeks.org/role-of-machine-learning-in-data-science/)
[Top Machine Learning Careers/Jobs\\
\\
\\
Machine Learning (ML) is one of the fastest-growing fields in technology, driving innovations across healthcare, finance, e-commerce, and more. As companies increasingly adopt AI-based solutions, the demand for skilled ML professionals is Soaring. This article delves into the Type of Machine Learnin\\
\\
10 min read](https://www.geeksforgeeks.org/top-career-paths-in-machine-learning/)

Like5

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/machine-learning-models/?ref=lbp)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1096965869.1745055233&gtm=45je54h0h2v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1593712160)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745055233395&cv=11&fst=1745055233395&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54h0h2v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fmachine-learning-models%2F%3Fref%3Dlbp&hn=www.googleadservices.com&frm=0&tiba=Machine%20Learning%20Models%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=523044939.1745055233&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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

[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)