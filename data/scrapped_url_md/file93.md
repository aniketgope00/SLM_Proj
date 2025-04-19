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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/?type%3Darticle%26id%3D266999&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Check if absolute difference of consecutive nodes is 1 in Linked List\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/check-if-absolute-difference-of-consecutive-nodes-is-1-in-linked-list/)

# What is a Neural Network?

Last Updated : 03 Apr, 2025

Comments

Improve

Suggest changes

50 Likes

Like

Report

Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making.

In this article, we will explore the fundamentals of neural networks, their architecture, how they work, and their applications in various fields. Understanding neural networks is essential for anyone interested in the advancements of artificial intelligence.

## Understanding Neural Networks in Deep Learning

Neural networks are capable of learning and identifying patterns directly from data without pre-defined rules. These networks are built from several key components:

1. **Neurons**: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.
2. **Connections**: Links between neurons that carry information, regulated by weights and biases.
3. **Weights and Biases**: These parameters determine the strength and influence of connections.
4. **Propagation Functions**: Mechanisms that help process and transfer data across layers of neurons.
5. **Learning Rule**: The method that adjusts weights and biases over time to improve accuracy.

**Learning in neural networks follows a structured, three-stage process:**

1. **Input Computation**: Data is fed into the network.
2. **Output Generation**: Based on the current parameters, the network generates an output.
3. **Iterative Refinement**: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks.

**In an adaptive learning environment:**

- The neural network is exposed to a simulated scenario or dataset.
- Parameters such as weights and biases are updated in response to new data or conditions.
- With each adjustment, the network’s response evolves, allowing it to adapt effectively to different tasks or environments.

![Artificial-Neural-Networks](https://media.geeksforgeeks.org/wp-content/uploads/20241106171024318092/Artificial-Neural-Networks.webp)

The image illustrates the analogy between a biological neuron and an artificial neuron, showing how inputs are received and processed to produce outputs in both systems.

## **Importance of Neural Networks**

Neural networks are pivotal in identifying complex patterns, solving intricate challenges, and adapting to dynamic environments. Their ability to learn from vast amounts of data is transformative, impacting technologies like **natural language processing**, **self-driving vehicles**, and **automated decision-making**.

Neural networks streamline processes, increase efficiency, and support decision-making across various industries. As a backbone of artificial intelligence, they continue to drive innovation, shaping the future of technology.

## **Evolution of Neural Networks**

Neural networks have undergone significant evolution since their inception in the mid-20th century. Here’s a concise timeline of the major developments in the field:

- **1940s-1950s**: The concept of neural networks began with McCulloch and Pitts’ introduction of the first mathematical model for **artificial neurons**. However, the lack of computational power during that time posed significant challenges to further advancements.

- **1960s-1970s**: Frank Rosenblatt’s worked on perceptrons. [**Perceptrons**](https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/) are simple single-layer networks that can solve linearly separable problems, but can not perform complex tasks.

- **1980s:** The development of **backpropagation** by Rumelhart, Hinton, and Williams revolutionized neural networks by enabling the training of multi-layer networks. This period also saw the rise of connectionism, emphasizing learning through interconnected nodes.

- **1990s:** Neural networks experienced a surge in popularity with applications across image recognition, finance, and more. However, this growth was tempered by a period known as the **“AI winter,”** during which high computational costs and unrealistic expectations dampened progress.

- **2000s:** A resurgence was triggered by the availability of larger datasets, advances in computational power, and innovative network architectures. Deep learning, utilizing multiple layers, proved highly effective across various domains.

- **2010s:** The landscape of machine learning has been dominated by deep learning with **CNNs**(Convolutional Neural Networks) excelling in image classification and **RNNs**(Recurrent Neural Networks) , **LSTMs**, and **GRUs** gaining traction in sequence-based tasks like language modeling and speech recognition.

- **2017: Transformer models**, introduced by Vaswani et al. in “Attention is All You Need,” revolutionized NLP by using a self-attention mechanism for parallel processing, improving efficiency. Models like **BERT**, **GPT**, and **T5** set new benchmarks in machine translation and text generation.

## Layers in Neural Network Architecture

1. **Input Layer:** This is where the network receives its input data. Each input neuron in the layer corresponds to a feature in the input data.
2. **Hidden Layers:** These layers perform most of the computational heavy lifting. A neural network can have one or multiple hidden layers. Each layer consists of units (neurons) that transform the inputs into something that the output layer can use.
3. **Output Layer:** The final layer produces the output of the model. The format of these outputs varies depending on the specific task (e.g., classification, regression).

![nn-ar-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20231204175521/nn-ar.jpg)

## Working of Neural Networks

### Forward Propagation

When data is input into the network, it passes through the network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. Here’s what happens during this phase:

1. **Linear Transformation:** Each neuron in a layer receives inputs, which are multiplied by the weights associated with the connections. These products are summed together, and a bias is added to the sum. This can be represented mathematically as: z=w1x1+w2x2+…+wnxn+bz = w\_1x\_1 + w\_2x\_2 + \\ldots + w\_nx\_n + bz=w1​x1​+w2​x2​+…+wn​xn​+b where www represents the weights, xxx represents the inputs, and bbb is the bias.
2. **Activation:** The result of the linear transformation (denoted as zzz) is then passed through an activation function. The activation function is crucial because it introduces non-linearity into the system, enabling the network to learn more complex patterns. Popular activation functions include ReLU, sigmoid, and tanh.

### Backpropagation

After forward propagation, the network evaluates its performance using a loss function, which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play:

1. **Loss Calculation:** The network calculates the loss, which provides a measure of error in the predictions. The loss function could vary; common choices are mean squared error for regression tasks or cross-entropy loss for classification.
2. **Gradient Calculation:** The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.
3. **Weight Update:** Once the gradients are calculated, the weights and biases are updated using an optimization algorithm like stochastic gradient descent (SGD). The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.

### Iteration

This process of forward propagation, loss calculation, backpropagation, and weight update is repeated for many iterations over the dataset. Over time, this iterative process reduces the loss, and the network’s predictions become more accurate.

Through these steps, neural networks can adapt their parameters to better approximate the relationships in the data, thereby improving their performance on tasks such as classification, regression, or any other predictive modeling.

### Example of Email Classification

Let’s consider a record of an email dataset:

| Email ID | Email Content | Sender | Subject Line | Label |
| --- | --- | --- | --- | --- |
| 1 | “Get free gift cards now!” | [spam@example.com](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) | “Exclusive Offer” | 1 |

To classify this email, we will create a feature vector based on the analysis of keywords such as “free,” “win,” and “offer.”

The feature vector of the record can be presented as:

- “free”: Present (1)
- “win”: Absent (0)
- “offer”: Present (1)

| Email ID | Email Content | Sender | Subject Line | Feature Vector | Label |
| --- | --- | --- | --- | --- | --- |
| 1 | “Get free gift cards now!” | [spam@example.com](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) | “Exclusive Offer” | \[1, 0, 1\] | 1 |

### How Neurons Process Data in a Neural Network

In a **neural network**, input data is passed through multiple layers, including one or more **hidden layers**. Each **neuron** in these hidden layers performs several operations, transforming the input into a usable output.

**1\. Input Layer:** The input layer contains 3 nodes that indicates the presence of each keyword.

**2\. Hidden Layer**

- The input data is passed through one or more hidden layers.
- Each neuron in the hidden layer performs the following operations



**Weighted Sum**: Each input is multiplied by a corresponding weight assigned to the connection. For example, if the weights from the input layer to the hidden layer neurons are as follows:
  - Weights for Neuron H1: `[0.5, -0.2, 0.3]`
  - Weights for Neuron H2: `[0.4, 0.1, -0.5]`



    **Calculate Weighted Input**:
    - For Neuron H1:
      - Calculation=(1×0.5)+(0×−0.2)+(1×0.3)=0.5+0+0.3=0.8\\text{Calculation}=(1 \\times 0.5) + (0 \\times -0.2) + (1 \\times 0.3) = 0.5 + 0 + 0.3 = 0.8Calculation=(1×0.5)+(0×−0.2)+(1×0.3)=0.5+0+0.3=0.8
    - For Neuron H2:
      - Calculation=(1×0.4)+(0×0.1)+(1×−0.5)=0.4+0−0.5=−0.1\\text{Calculation} = (1×0.4)+(0×0.1)+(1×−0.5)=0.4+0−0.5=−0.1Calculation=(1×0.4)+(0×0.1)+(1×−0.5)=0.4+0−0.5=−0.1
    - **Activation Function**: The result is passed through an activation function (e.g., ReLU or sigmoid) to introduce non-linearity.
      - For H1, applying ReLU: ReLU(0.8)=0.8\\text{ReLU}(0.8) = 0.8ReLU(0.8)=0.8
      - For H2, applying ReLU: ReLU(−0.1)=0\\text{ReLU}(-0.1) = 0ReLU(−0.1)=0

#### 3\. Output Layer

- The activated outputs from the hidden layer are passed to the output neuron.
- The output neuron receives the values from the hidden layer neurons and computes the final prediction using weights:
  - Suppose the output weights from hidden layer to output neuron are `[0.7, 0.2]`.
  - Calculation:
    - Input=(0.8×0.7)+(0×0.2)=0.56+0=0.56\\text{Input}=(0.8 \\times 0.7) + (0 \\times 0.2) = 0.56 + 0 = 0.56Input=(0.8×0.7)+(0×0.2)=0.56+0=0.56
  - **Final Activation**: The output is passed through a sigmoid activation function to obtain a probability:
    - σ(0.56)≈0.636\\sigma(0.56) \\approx 0.636σ(0.56)≈0.636

#### 4\. Final Classification

- The output value of approximately **0.636** indicates the probability of the email being spam.
- Since this value is greater than 0.5, the neural network classifies the email as spam (1).

![Neural-Network](https://media.geeksforgeeks.org/wp-content/uploads/20241106184728862313/Neural-Network.png)

Neural Network for Email Classification Example

## Learning of a Neural Network

### **1\. Learning with Supervised Learning**

In supervised learning, a neural network learns from labeled input-output pairs provided by a teacher. The network generates outputs based on inputs, and by comparing these outputs to the known desired outputs, an error signal is created. The network iteratively adjusts its parameters to minimize errors until it reaches an acceptable performance level.

### **2\. Learning with Unsupervised Learning**

Unsupervised learning involves data without labeled output variables. The primary goal is to understand the underlying structure of the input data (X). Unlike supervised learning, there is no instructor to guide the process. Instead, the focus is on modeling data patterns and relationships, with techniques like clustering and association commonly used.

### **3\. Learning with Reinforcement Learning**

Reinforcement learning enables a neural network to learn through interaction with its environment. The network receives feedback in the form of rewards or penalties, guiding it to find an optimal policy or strategy that maximizes cumulative rewards over time. This approach is widely used in applications like gaming and decision-making.

## Types of Neural Networks

There are _seven_ types of neural networks that can be used.

- [**Feedforward Networks**](https://www.geeksforgeeks.org/feedforward-neural-network/) **:** A feedforward neural network is a simple artificial neural network architecture in which data moves from input to output in a single direction.
- [**Singlelayer Perceptron:**](https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/) A **single-layer perceptron** consists of only one layer of neurons . It takes inputs, applies weights, sums them up, and uses an activation function to produce an output **.**
- [**Multilayer Perceptron (MLP)**](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/) **:** MLP is a type of feedforward neural network with three or more layers, including an input layer, one or more hidden layers, and an output layer. It uses nonlinear activation functions.
- [**Convolutional Neural Network (CNN)**](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) **:** A Convolutional Neural Network (CNN) is a specialized artificial neural network designed for image processing. It employs convolutional layers to automatically learn hierarchical features from input images, enabling effective image recognition and classification.
- [**Recurrent Neural Network (RNN)**](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/) **:** An artificial neural network type intended for sequential data processing is called a Recurrent Neural Network (RNN). It is appropriate for applications where contextual dependencies are critical, such as time series prediction and natural language processing, since it makes use of feedback loops, which enable information to survive within the network.
- [**Long Short-Term Memory (LSTM)**](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/) **:** LSTM is a type of RNN that is designed to overcome the vanishing gradient problem in training RNNs. It uses memory cells and gates to selectively read, write, and erase information.

## Implementation of Neural Network using TensorFlow

Here, we implement simple feedforward neural network that trains on a sample dataset and makes predictions using following steps:

### Step 1: Import Necessary Libraries

Import necessary libraries, primarily TensorFlow and Keras, along with other required packages such as NumPy and Pandas for data handling.

Python`
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
`

### Step 2: Create and Load Dataset

- Create or load a dataset. Convert the data into a format suitable for training (usually NumPy arrays).
- Define features (X) and labels (y).

Python`
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values
`

### Step 3: Create a Neural Network

Instantiate a Sequential model and add layers. The input layer and hidden layers are typically created using `Dense` layers, specifying the number of neurons and activation functions.

Python`
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer
`

### **Step 4: Compiling the Model**

Compile the model by specifying the loss function, optimizer, and metrics to evaluate during training.

Python`
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
`

### **Step 5: Train the Model**

Fit the model on the training data, specifying the number of epochs and batch size. This step trains the neural network to learn from the input data.

Python`
model.fit(X, y, epochs=100, batch_size=1, verbose=1)
`

### **Step 5: Make Predictions**

Use the trained model to make predictions on new data. Process the output to interpret the predictions (e.g., convert probabilities to binary outcomes).

Python`
test_data = np.array([[0.2, 0.4]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
`

### Complete Code for the Implementation

Let’s have a complete code for the implementation.

Python`
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1, verbose=1)
test_data = np.array([[0.2, 0.4]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
print(f"Predicted label: {predicted_label[0][0]}")
`

**Output:**

> **Predicted label: 1**

## Advantages of Neural Networks

Neural networks are widely used in many different applications because of their many benefits:

- **Adaptability:** Neural networks are useful for activities where the link between inputs and outputs is complex or not well defined because they can adapt to new situations and learn from data.
- **Pattern Recognition:** Their proficiency in pattern recognition renders them efficacious in tasks like as audio and image identification, natural language processing, and other intricate data patterns.
- **Parallel Processing:** Because neural networks are capable of parallel processing by nature, they can process numerous jobs at once, which speeds up and improves the efficiency of computations.
- **Non-Linearity:** Neural networks are able to model and comprehend complicated relationships in data by virtue of the non-linear activation functions found in neurons, which overcome the drawbacks of linear models.

## Disadvantages of Neural Networks

Neural networks, while powerful, are not without drawbacks and difficulties:

- **Computational Intensity:** Large neural network training can be a laborious and computationally demanding process that demands a lot of computing power.
- **Black box Nature:** As “black box” models, neural networks pose a problem in important applications since it is difficult to understand how they make decisions.
- **Overfitting:** Overfitting is a phenomenon in which neural networks commit training material to memory rather than identifying patterns in the data. Although regularization approaches help to alleviate this, the problem still exists.
- **Need for Large datasets:** For efficient training, neural networks frequently need sizable, labeled datasets; otherwise, their performance may suffer from incomplete or skewed data.

## Applications of Neural Networks

Neural networks have numerous applications across various fields:

1. **Image and Video Recognition**: CNNs are extensively used in applications such as facial recognition, autonomous driving, and medical image analysis.
2. **Natural Language Processing (NLP)**: RNNs and transformers power language translation, chatbots, and sentiment analysis.
3. **Finance**: Predicting stock prices, fraud detection, and risk management.
4. **Healthcare**: Neural networks assist in diagnosing diseases, analyzing medical images, and personalizing treatment plans.
5. **Gaming and Autonomous Systems**: Neural networks enable real-time decision-making, enhancing user experience in video games and enabling autonomous systems like self-driving cars.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/check-if-absolute-difference-of-consecutive-nodes-is-1-in-linked-list/)

[Check if absolute difference of consecutive nodes is 1 in Linked List](https://www.geeksforgeeks.org/check-if-absolute-difference-of-consecutive-nodes-is-1-in-linked-list/)

[V](https://www.geeksforgeeks.org/user/Veena%20Ghorakavi/)

[Veena Ghorakavi](https://www.geeksforgeeks.org/user/Veena%20Ghorakavi/)

Follow

50

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Articles](https://www.geeksforgeeks.org/category/articles/)
- [Computer Subject](https://www.geeksforgeeks.org/category/computer-subject/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Technical Scripter](https://www.geeksforgeeks.org/category/technical-scripter/)
- [Technical Scripter 2018](https://www.geeksforgeeks.org/tag/technical-scripter-2018/)

+2 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[What is Dynamic Neural Network?\\
\\
\\
Dynamic Neural Networks are the upgraded version of Static Neural Networks. They have better decision algorithms and can generate better-quality results. The decision algorithm refers to the improvements to the network. It is responsible for making the right decisions accurately and with the right a\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-dynamic-neural-network/?ref=ml_lbp)
[What are Graph Neural Networks?\\
\\
\\
Graph Neural Networks (GNNs) are a neural network specifically designed to work with data represented as graphs. Unlike traditional neural networks, which operate on grid-like data structures like images (2D grids) or text (sequential), GNNs can model complex, non-Euclidean relationships in data, su\\
\\
13 min read](https://www.geeksforgeeks.org/what-are-graph-neural-networks/?ref=ml_lbp)
[Weights and Bias in Neural Networks\\
\\
\\
Machine learning, with its ever-expanding applications in various domains, has revolutionized the way we approach complex problems and make data-driven decisions. At the heart of this transformative technology lies neural networks, computational models inspired by the human brain's architecture. Neu\\
\\
13 min read](https://www.geeksforgeeks.org/the-role-of-weights-and-bias-in-neural-networks/?ref=ml_lbp)
[Shallow Neural Networks\\
\\
\\
Neural networks represent the backbone of modern artificial intelligence, helping machines mimic human decision-making processes. While deep neural networks, with their multiple layers, are often in the spotlight for complex tasks, shallow neural networks play a crucial role, especially in scenarios\\
\\
7 min read](https://www.geeksforgeeks.org/shallow-neural-networks/?ref=ml_lbp)
[Neural Network Advances\\
\\
\\
In recent years, Neural networks are growing rapidly and creating new technology. These neural network mimic human brain and its thought process. Letâ€™s see some of the key advancements in neural networks and how they are impacting the world. 1. Kolmogorov-Arnold Networks (KANs)Kolmogorov-Arnold Netw\\
\\
4 min read](https://www.geeksforgeeks.org/neural-network-advances/?ref=ml_lbp)
[Applications of Neural Network\\
\\
\\
A neural network is a processing device, either an algorithm or genuine hardware, that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. The computing world has a ton to acquire from neural networks, also known as artific\\
\\
3 min read](https://www.geeksforgeeks.org/applications-of-neural-network/?ref=ml_lbp)
[Neural Network Node\\
\\
\\
In the realm of artificial intelligence and machine learning particularly within the neural networks the concept of a "node" is fundamental. Nodes, often referred to as neurons in the context of neural networks are the core computational units that drive the learning process. They play a crucial rol\\
\\
5 min read](https://www.geeksforgeeks.org/neural-network-node/?ref=ml_lbp)
[Deep Neural Network With L - Layers\\
\\
\\
This article aims to implement a deep neural network with an arbitrary number of hidden layers each containing different numbers of neurons. We will be implementing this neural net using a few helper functions and at last, we will combine these functions to make the L-layer neural network model.L -\\
\\
11 min read](https://www.geeksforgeeks.org/deep-neural-network-with-l-layers/?ref=ml_lbp)
[What is Forward Propagation in Neural Networks?\\
\\
\\
Forward propagation is the fundamental process in a neural network where input data passes through multiple layers to generate an output. It is the process by which input data passes through each layer of neural network to generate output. In this article, weâ€™ll more about forward propagation and se\\
\\
4 min read](https://www.geeksforgeeks.org/what-is-forward-propagation-in-neural-networks/?ref=ml_lbp)
[Feedback System in Neural Networks\\
\\
\\
A feedback system in neural networks is a mechanism where the output is fed back into the network to influence subsequent outputs, often used to enhance learning and stability. This article provides an overview of the working of the feedback loop in Neural Networks. Understanding Feedback SystemIn d\\
\\
6 min read](https://www.geeksforgeeks.org/feedback-system-in-neural-networks/?ref=ml_lbp)

Like50

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1393322472.1745056716&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=2076703475)

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

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)