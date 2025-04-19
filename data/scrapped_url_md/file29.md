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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/unsupervised-learning/?type%3Darticle%26id%3D196856&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Semi-Supervised Learning in ML\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/ml-semi-supervised-learning/)

# What is Unsupervised Learning?

Last Updated : 15 Jan, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

Unsupervised learning is a branch of [machine learning](https://www.geeksforgeeks.org/machine-learning/) that deals with unlabeled data. Unlike supervised learning, where the data is labeled with a specific category or outcome, unsupervised learning algorithms **are tasked with finding patterns and relationships within the data without any prior knowledge of the data’s meaning**. Unsupervised machine learning algorithms **find hidden patterns and data without any human intervention, i.e., we don’t give output to our model. The training model has only input parameter values and discovers the groups or patterns on its own.**

![Unsupervised-learning](https://media.geeksforgeeks.org/wp-content/uploads/20231124111325/Unsupervised-learning.png)

Unsupervised Learning

**The image shows set of animals:** elephants, camels, and cows that represents raw data that the unsupervised learning algorithm will process.

- The “Interpretation” stage signifies that the algorithm doesn’t have predefined labels or categories for the data. It needs to figure out how to group or organize the data based on inherent patterns.
- **Algorithm** represents the core of unsupervised learning process using techniques like clustering, dimensionality reduction, or anomaly detection to identify patterns and structures in the data.
- **Processing** stage shows the algorithm working on the data.

The output shows the results of the unsupervised learning process. In this case, the algorithm might have grouped the animals into clusters based on their species (elephants, camels, cows).

## How does unsupervised learning work?

Unsupervised learning works by analyzing unlabeled data to identify patterns and relationships. The data is not labeled with any predefined categories or outcomes, so the algorithm must find these patterns and relationships on its own. This can be a challenging task, but it can also be very rewarding, as it can reveal insights into the data that would not be apparent from a labeled dataset.

Data-set in Figure A is Mall data that contains information about its clients that subscribe to them. Once subscribed they are provided a membership card and the mall has complete information about the customer and his/her every purchase. Now using this data and unsupervised learning techniques, the mall can easily group clients based on the parameters we are feeding in.

![](https://media.geeksforgeeks.org/wp-content/uploads/CLuster.png)

The input to the unsupervised learning models is as follows:

- **Unstructured data**: May contain noisy(meaningless) data, missing values, or unknown data
- **Unlabeled data**: Data only contains a value for input parameters, there is no targeted value(output). It is easy to collect as compared to the labeled one in the Supervised approach.

## Unsupervised Learning Algorithms

There are mainly 3 types of Algorithms which are used for Unsupervised dataset.

- **Clustering**
- **Association Rule Learning**
- **Dimensionality Reduction**

### **1\. Clustering Algorithms**

[Clustering](https://www.geeksforgeeks.org/clustering-in-machine-learning/) in unsupervised machine learning is the process of grouping unlabeled data into clusters based on their similarities. The goal of clustering is to identify patterns and relationships in the data without any prior knowledge of the data’s meaning.

Broadly this technique is applied to group data based on different patterns, such as similarities or differences, our machine model finds. These algorithms are used to process raw, unclassified data objects into groups. For example, in the above figure, we have not given output parameter values, so this technique will be used to group clients based on the input parameters provided by our data.

> **Some common clustering algorithms:**
>
> - [**K-means Clustering**](https://www.geeksforgeeks.org/k-means-clustering-introduction/) **:** Groups data into K clusters based on how close the points are to each other.
> - [**Hierarchical Clustering**](https://www.geeksforgeeks.org/ml-hierarchical-clustering-agglomerative-and-divisive-clustering/) **:** Creates clusters by building a tree step-by-step, either merging or splitting groups.
> - [**Density-Based Clustering (DBSCAN)**](https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/) **:** Finds clusters in dense areas and treats scattered points as noise.
> - [**Mean-Shift Clustering**](https://www.geeksforgeeks.org/ml-mean-shift-clustering/) **:** Discovers clusters by moving points toward the most crowded areas.
> - [**Spectral Clustering**](https://www.geeksforgeeks.org/ml-spectral-clustering/) **:** Groups data by analyzing connections between points using graphs.

### **2\. Association Rule Learning**

[Association rule learning](https://www.geeksforgeeks.org/association-rule/) is also known as association rule mining is a common technique used to discover associations in unsupervised machine learning. This technique is a rule-based ML technique that finds out some very useful relations between parameters of a large data set. This technique is basically used for market basket analysis that helps to better understand the relationship between different products.

For e.g. shopping stores use algorithms based on this technique to find out the relationship between the sale of one product w.r.t to another’s sales based on customer behavior. **Like if a customer buys milk, then he may also buy bread, eggs, or butter**. Once trained well, such models can be used to increase their sales by planning different offers.

> **Some common Association Rule Learning algorithms:**
>
> - [**Apriori Algorithm**](https://www.geeksforgeeks.org/apriori-algorithm/) **:** Finds patterns by exploring frequent item combinations step-by-step.
> - [**FP-Growth Algorithm**](https://www.geeksforgeeks.org/frequent-pattern-growth-algorithm/) **:** An Efficient Alternative to Apriori. It quickly identifies frequent patterns without generating candidate sets.
> - [**Eclat Algorithm**](https://www.geeksforgeeks.org/ml-eclat-algorithm/) **:** Uses intersections of itemsets to efficiently find frequent patterns.
> - [**Efficient Tree-based Algorithms**](https://www.geeksforgeeks.org/introduction-to-tree-data-structure-and-algorithm-tutorials/) **:** Scales to handle large datasets by organizing data in tree structures.

### **3\. Dimensionality Reduction**

Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much information as possible. This technique is useful for improving the performance of machine learning algorithms and for data visualization.

Imagine a dataset of 100 features about students (height, weight, grades, etc.). To focus on key traits, you reduce it to just 2 features: height and grades, making it easier to visualize or analyze the data.

> Here are some popular **Dimensionality Reduction algorithms**:
>
> - [**Principal Component Analysis (PCA)**](https://www.geeksforgeeks.org/principal-component-analysis-pca/) **:** Reduces dimensions by transforming data into uncorrelated principal components.
> - [**Linear Discriminant Analysis (LDA)**](https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/) **:** Reduces dimensions while maximizing class separability for classification tasks.
> - [**Non-negative Matrix Factorization (NMF**](https://www.geeksforgeeks.org/non-negative-matrix-factorization/) **):** Breaks data into non-negative parts to simplify representation.
> - [**Locally Linear Embedding (LLE)**](https://www.geeksforgeeks.org/locally-linear-embedding-in-machine-learning/) **:** Reduces dimensions while preserving the relationships between nearby points.
> - [**Isomap**](https://www.geeksforgeeks.org/isomap-a-non-linear-dimensionality-reduction-technique/) **:** Captures global data structure by preserving distances along a manifold.

## Challenges of **Unsupervised Learning**

Here are the key challenges of unsupervised learning:

- **Noisy Data**: Outliers and noise can distort patterns and reduce the effectiveness of algorithms.
- **Assumption Dependence**: Algorithms often rely on assumptions (e.g., cluster shapes), which may not match the actual data structure.
- **Overfitting Risk**: Overfitting can occur when models capture noise instead of meaningful patterns in the data.
- **Limited Guidance**: The absence of labels restricts the ability to guide the algorithm toward specific outcomes.
- **Cluster Interpretability**: Results, such as clusters, may lack clear meaning or alignment with real-world categories.
- **Sensitivity to Parameters**: Many algorithms require careful tuning of hyperparameters, such as the number of clusters in k-means.
- **Lack of Ground Truth**: Unsupervised learning lacks labeled data, making it difficult to evaluate the accuracy of results.

## Applications of Unsupervised learning

**Unsupervised learning has diverse applications across industries and domains. Key applications include:**

- **Customer Segmentation:** Algorithms cluster customers based on purchasing behavior or demographics, enabling targeted marketing strategies.
- **Anomaly Detection:** Identifies unusual patterns in data, aiding fraud detection, cybersecurity, and equipment failure prevention.
- **Recommendation Systems**: Suggests products, movies, or music by analyzing user behavior and preferences.
- **Image and Text Clustering**: Groups similar images or documents for tasks like organization, classification, or content recommendation.
- **Social Network Analysis**: Detects communities or trends in user interactions on social media platforms.
- **Astronomy and Climate Science:** Classifies galaxies or groups weather patterns to support scientific research

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/ml-semi-supervised-learning/)

[Semi-Supervised Learning in ML](https://www.geeksforgeeks.org/ml-semi-supervised-learning/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
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


Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/unsupervised-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1732390137.1745055441&gtm=45je54h0h2v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=318966735)

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