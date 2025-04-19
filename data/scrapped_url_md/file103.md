- [Statistics with Python](https://www.geeksforgeeks.org/statistics-with-python/)
- [Data Analysis Tutorial](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Python â€“ Data visualization tutorial](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [R](https://www.geeksforgeeks.org/r-tutorial/)
- [Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning-projects/)_)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)_)
- [Machine Learning Mathematics](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Deep Learning Tutorial](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Deep Learning Project](https://www.geeksforgeeks.org/5-deep-learning-project-ideas-for-beginners/)_)
- [Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning-interview-questions/)_)
- [Computer Vision Tutorial](https://www.geeksforgeeks.org/computer-vision/)
- [Computer Vision Projects](https://www.geeksforgeeks.org/computer-vision-projects/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [NLP Project](https://www.geeksforgeeks.org/nlp-project-ideas-for-beginners/))
- [NLP Interview Questions](https://www.geeksforgeeks.org/nlp-interview-questions/))
- [Statistics with Python](https://www.geeksforgeeks.org/statistics-with-python/)
- [100 Days of Machine Learning](https://www.geeksforgeeks.org/100-days-of-machine-learning/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/backpropagation-in-neural-network/?type%3Darticle%26id%3D1169637&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
K-Nearest Neighbors and Curse of Dimensionality\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/)

# Backpropagation in Neural Network

Last Updated : 05 Apr, 2025

Comments

Improve

Suggest changes

19 Likes

Like

Report

_Backpropagation_ is also known as " _Backward Propagation of Errors_" and it is a method used to train [**neural network**](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) . Its goal is to reduce the difference between the model’s predicted output and the actual output by adjusting the weights and biases in the network. In this article we will explore what backpropagation is, why it is crucial in machine learning and how it works.

## What is Backpropagation?

**Backpropagation** is a technique used in deep learning to train artificial neural networks particularly [**feed-forward networks**](https://www.geeksforgeeks.org/feedforward-neural-network/). It works iteratively to adjust weights and bias to minimize the cost function.

In each epoch the model adapts these parameters reducing loss by following the error gradient. Backpropagation often uses optimization algorithms like [**gradient descent**](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/) or [**stochastic gradient descent**](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/). The algorithm computes the gradient using the chain rule from calculus allowing it to effectively navigate complex layers in the neural network to minimize the cost function.

![Frame-13](https://media.geeksforgeeks.org/wp-content/uploads/20240217152156/Frame-13.png)Fig(a) A simple illustration of how the backpropagation works by adjustments of weights

Backpropagation plays a critical role in how neural networks improve over time. Here's why:

1. **Efficient Weight Update**: It computes the gradient of the loss function with respect to each weight using the chain rule making it possible to update weights efficiently.
2. **Scalability**: The backpropagation algorithm scales well to networks with multiple layers and complex architectures making deep learning feasible.
3. **Automated Learning**: With backpropagation the learning process becomes automated and the model can adjust itself to optimize its performance.

## Working of Backpropagation Algorithm

The Backpropagation algorithm involves two main steps: the **Forward Pass** and the **Backward Pass**.

### How Does Forward Pass Work?

In **forward pass** the input data is fed into the input layer. These inputs combined with their respective weights are passed to hidden layers. For example in a network with two hidden layers (h1 and h2 as shown in Fig. (a)) the output from h1 serves as the input to h2. Before applying an activation function, a bias is added to the weighted inputs.

Each hidden layer computes the weighted sum (\`a\`) of the inputs then applies an activation function like [**ReLU (Rectified Linear Unit)**](https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/) to obtain the output (\`o\`). The output is passed to the next layer where an activation function such as [**softmax**](https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) converts the weighted outputs into probabilities for classification.

![forwards-pass](https://media.geeksforgeeks.org/wp-content/uploads/20240218115130/forwards-pass.png)The forward pass using weights and biases

### How Does the Backward Pass Work?

In the backward pass the error (the difference between the predicted and actual output) is propagated back through the network to adjust the weights and biases. One common method for error calculation is the [**Mean Squared Error (MSE)**](https://www.geeksforgeeks.org/mean-squared-error/) given by:

MSE=(Predicted Output−Actual Output)2MSE = (\\text{Predicted Output} - \\text{Actual Output})^2MSE=(Predicted Output−Actual Output)2

Once the error is calculated the network adjusts weights using **gradients** which are computed with the chain rule. These gradients indicate how much each weight and bias should be adjusted to minimize the error in the next iteration. The backward pass continues layer by layer ensuring that the network learns and improves its performance. The activation function through its derivative plays a crucial role in computing these gradients during backpropagation.

## Example of Backpropagation in Machine Learning

Let’s walk through an example of backpropagation in machine learning. Assume the neurons use the sigmoid activation function for the forward and backward pass. The target output is 0.5, and the learning rate is 1.

![example-1](https://media.geeksforgeeks.org/wp-content/uploads/20240220194947/example-1.png)Example (1) of backpropagation sum

### Forward Propagation

#### 1\. Initial Calculation

The weighted sum at each node is calculated using:

aj​=∑(wi​,j∗xi​)a j​ =∑(w i​ ,j∗x i​ )aj​=∑(wi​,j∗xi​)

Where,

- aja\_jaj​ is the weighted sum of all the inputs and weights at each node
- wi,jw\_{i,j}wi,j​ represents the weights between the ithi^{th}ithinput and the jthj^{th}jth neuron
- xix\_ixi​ represents the value of the ithi^{th}ith input

**`o`** **(output):** After applying the activation function to `a,` we get the output of the neuron:

ojo\_joj​ = activation function(aja\_j aj​)

#### 2\. Sigmoid Function

The sigmoid function returns a value between 0 and 1, introducing non-linearity into the model.

yj=11+e−ajy\_j = \\frac{1}{1+e^{-a\_j}}yj​=1+e−aj​1​

![example-2](https://media.geeksforgeeks.org/wp-content/uploads/20240220202427/example-2.png)To find the outputs of y3, y4 and y5

#### 3\. Computing Outputs

At h1 node

a1=(w1,1x1)+(w2,1x2)=(0.2∗0.35)+(0.2∗0.7)=0.21\\begin {aligned}a\_1 &= (w\_{1,1} x\_1) + (w\_{2,1} x\_2) \\\& = (0.2 \* 0.35) + (0.2\* 0.7)\\\&= 0.21\\end {aligned}a1​​=(w1,1​x1​)+(w2,1​x2​)=(0.2∗0.35)+(0.2∗0.7)=0.21​

Once we calculated the a1 value, we can now proceed to find the y3 value:

yj=F(aj)=11+e−a1y\_j= F(a\_j) = \\frac 1 {1+e^{-a\_1}}yj​=F(aj​)=1+e−a1​1​

y3=F(0.21)=11+e−0.21y\_3 = F(0.21) = \\frac 1 {1+e^{-0.21}}y3​=F(0.21)=1+e−0.211​

y3=0.56y\_3 = 0.56y3​=0.56

Similarly find the values of y4 at **h** ****2**** and y5 at O3

a2=(w1,2∗x1)+(w2,2∗x2)=(0.3∗0.35)+(0.3∗0.7)=0.315a\_2 = (w\_{1,2} \* x\_1) + (w\_{2,2} \* x\_2) = (0.3\*0.35)+(0.3\*0.7)=0.315a2​=(w1,2​∗x1​)+(w2,2​∗x2​)=(0.3∗0.35)+(0.3∗0.7)=0.315

y4=F(0.315)=11+e−0.315y\_4 = F(0.315) = \\frac 1{1+e^{-0.315}}y4​=F(0.315)=1+e−0.3151​

a3=(w1,3∗y3)+(w2,3∗y4)=(0.3∗0.57)+(0.9∗0.59)=0.702a3 = (w\_{1,3}\*y\_3)+(w\_{2,3}\*y\_4) =(0.3\*0.57)+(0.9\*0.59) =0.702a3=(w1,3​∗y3​)+(w2,3​∗y4​)=(0.3∗0.57)+(0.9∗0.59)=0.702

y5=F(0.702)=11+e−0.702=0.67y\_5 = F(0.702) = \\frac 1 {1+e^{-0.702} } = 0.67 y5​=F(0.702)=1+e−0.7021​=0.67

![example-3](https://media.geeksforgeeks.org/wp-content/uploads/20240220210122/example-3.png)Values of y3, y4 and y5

#### 4\. Error Calculation

Our actual output is 0.5 but we obtained 0.67 _._ To calculate the error we can use the below formula:

Errorj=ytarget−y5Error\_j= y\_{target} - y\_5Errorj​=ytarget​−y5​

Error=0.5−0.67=−0.17Error = 0.5 - 0.67 = -0.17Error=0.5−0.67=−0.17

Using this error value we will be backpropagating.

### **Backpropagation**

#### **1\. Calculating Gradients**

The change in each weight is calculated as:

Δwij=η×δj×Oj\\Delta w\_{ij} = \\eta \\times \\delta\_j \\times O\_jΔwij​=η×δj​×Oj​

Where:

- δj\\delta\_jδj​​ is the error term for each unit,
- η\\etaη is the learning rate.

#### **2\. Output Unit Error**

For O3:

δ5=y5(1−y5)(ytarget−y5)\\delta\_5 = y\_5(1-y\_5) (y\_{target} - y\_5) δ5​=y5​(1−y5​)(ytarget​−y5​)

=0.67(1−0.67)(−0.17)=−0.0376 = 0.67(1-0.67)(-0.17) = -0.0376=0.67(1−0.67)(−0.17)=−0.0376

#### **3\. Hidden Unit Error**

For h1:

δ3=y3(1−y3)(w1,3×δ5)\\delta\_3 = y\_3 (1-y\_3)(w\_{1,3} \\times \\delta\_5)δ3​=y3​(1−y3​)(w1,3​×δ5​)

=0.56(1−0.56)(0.3×−0.0376)=−0.0027= 0.56(1-0.56)(0.3 \\times -0.0376) = -0.0027=0.56(1−0.56)(0.3×−0.0376)=−0.0027

For h2:

δ4=y4(1−y4)(w2,3×δ5)\\delta\_4 = y\_4(1-y\_4)(w\_{2,3} \\times \\delta\_5) δ4​=y4​(1−y4​)(w2,3​×δ5​)

=0.59(1−0.59)(0.9×−0.0376)=−0.0819=0.59 (1-0.59)(0.9 \\times -0.0376) = -0.0819=0.59(1−0.59)(0.9×−0.0376)=−0.0819

#### 4\. Weight Updates

For the weights from hidden to output layer:

Δw2,3=1×(−0.0376)×0.59=−0.022184\\Delta w\_{2,3} = 1 \\times (-0.0376) \\times 0.59 = -0.022184Δw2,3​=1×(−0.0376)×0.59=−0.022184

New weight:

w2,3(new)=−0.022184+0.9=0.877816w\_{2,3}(\\text{new}) = -0.022184 + 0.9 = 0.877816w2,3​(new)=−0.022184+0.9=0.877816

For weights from input to hidden layer:

Δw1,1=1×(−0.0027)×0.35=0.000945\\Delta w\_{1,1} = 1 \\times (-0.0027) \\times 0.35 = 0.000945Δw1,1​=1×(−0.0027)×0.35=0.000945

New weight:

w1,1(new)=0.000945+0.2=0.200945w\_{1,1}(\\text{new}) = 0.000945 + 0.2 = 0.200945w1,1​(new)=0.000945+0.2=0.200945

Similarly other weights are updated:

- w1,2(new)=0.273225w\_{1,2}(\\text{new}) = 0.273225w1,2​(new)=0.273225
- w1,3(new)=0.086615w\_{1,3}(\\text{new}) = 0.086615w1,3​(new)=0.086615
- w2,1(new)=0.269445w\_{2,1}(\\text{new}) = 0.269445w2,1​(new)=0.269445
- w2,2(new)=0.18534w\_{2,2}(\\text{new}) = 0.18534w2,2​(new)=0.18534

The updated weights are illustrated below

![example-4-(1)](https://media.geeksforgeeks.org/wp-content/uploads/20240220210850/example-4-(1).png)Through backward pass the weights are updated

After updating the weights the forward pass is repeated yielding:

- y3=0.57y\_3 = 0.57y3​=0.57
- y4=0.56y\_4 = 0.56y4​=0.56
- y5=0.61y\_5 = 0.61y5​=0.61

Since y5=0.61y\_5 = 0.61y5​=0.61 is still not the target output the process of calculating the error and backpropagating continues until the desired output is reached.

This process demonstrates how backpropagation iteratively updates weights by minimizing errors until the network accurately predicts the output.

Error=ytarget−y5Error = y\_{target} - y\_5Error=ytarget​−y5​

=0.5−0.61=−0.11= 0.5 - 0.61 = -0.11=0.5−0.61=−0.11

This process is said to be continued until the actual output is gained by the neural network.

## Backpropagation Implementation in Python for XOR Problem

This code demonstrates how backpropagation is used in a neural network to solve the XOR problem. The neural network consists of:

### 1\. Defining Neural Network

- **Input layer** with 2 inputs
- **Hidden layer** with 4 neurons
- **Output layer** with 1 output neuron
- Using Sigmoid function as activation function

Python`
import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
`

- **`def __init__(self, input_size, hidden_size, output_size):`**: constructor to initialize the neural network
- **`self.input_size = input_size`**: stores the size of the input layer
- **`self.hidden_size = hidden_size`**: stores the size of the hidden layer
- **`self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)`**: initializes weights for input to hidden layer
- **`self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)`**: initializes weights for hidden to output layer
- **`self.bias_hidden = np.zeros((1, self.hidden_size))`**: initializes bias for hidden layer
- **`self.bias_output = np.zeros((1, self.output_size))`**: initializes bias for output layer

### 2\. Defining Feed Forward Network

In Forward pass inputs are passed through the network activating the hidden and output layers using the sigmoid function.

Python`
    def feedforward(self, X):
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)
        return self.predicted_output
`

- **`self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden`**: calculates activation for hidden layer
- **`self.hidden_output = self.sigmoid(self.hidden_activation)`**: applies activation function to hidden layer
- **`self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output`**: calculates activation for output layer
- **`self.predicted_output = self.sigmoid(self.output_activation)`**: applies activation function to output layer

### 3\. Defining Backward Network

In Backward pass (Backpropagation) the errors between the predicted and actual outputs are computed. The gradients are calculated using the derivative of the sigmoid function and weights and biases are updated accordingly.

Python`
    def backward(self, X, y, learning_rate):
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
`

- **`output_error = y - self.predicted_output`**: calculates the error at the output layer
- **`output_delta = output_error * self.sigmoid_derivative(self.predicted_output)`**: calculates the delta for the output layer
- **`hidden_error = np.dot(output_delta, self.weights_hidden_output.T)`**: calculates the error at the hidden layer
- **`hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)`**: calculates the delta for the hidden layer
- **`self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate`**: updates weights between hidden and output layers
- **`self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate`**: updates weights between input and hidden layers

### 4\. Training Network

The network is trained over 10,000 epochs using the backpropagation algorithm with a learning rate of 0.1 progressively reducing the error.

Python`
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")
`

- **`output = self.feedforward(X)`**: computes the output for the current inputs
- **`self.backward(X, y, learning_rate)`**: updates weights and biases using backpropagation
- **`loss = np.mean(np.square(y - output))`**: calculates the mean squared error (MSE) loss

### 5\. Testing Neural Network

Python`
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)
output = nn.feedforward(X)
print("Predictions after training:")
print(output)
`

- **`X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`**: defines the input data
- **`y = np.array([[0], [1], [1], [0]])`**: defines the target values
- **`nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)`**: initializes the neural network
- **`nn.train(X, y, epochs=10000, learning_rate=0.1)`**: trains the network
- **`output = nn.feedforward(X)`**: gets the final predictions after training

**Output:**

![Screenshot-2025-03-07-130223](https://media.geeksforgeeks.org/wp-content/uploads/20250307130234877256/Screenshot-2025-03-07-130223.png)Trained ModelFreFF

The output shows the training progress of a neural network over 10,000 epochs. Initially the loss was high (0.2713) but it gradually decreased as the network learned reaching a low value of 0.0066 by epoch 8000. The final predictions are close to the expected XOR outputs: approximately 0 for `[0, 0]` and `[1, 1]` and approximately 1 for `[0, 1]` and `[1, 0]` indicating that the network successfully learned to approximate the XOR function.

## Advantages of Backpropagation for Neural Network Training

The key benefits of using the backpropagation algorithm are:

1. **Ease of Implementation:** Backpropagation is beginner-friendly requiring no prior neural network knowledge and simplifies programming by adjusting weights with error derivatives.
2. **Simplicity and Flexibility:** Its straightforward design suits a range of tasks from basic feedforward to complex convolutional or recurrent networks.
3. **Efficiency:** Backpropagation accelerates learning by directly updating weights based on error especially in deep networks.
4. **Generalization:** It helps models generalize well to new data improving prediction accuracy on unseen examples.
5. **Scalability:** The algorithm scales efficiently with larger datasets and more complex networks making it ideal for large-scale tasks.

## Challenges with Backpropagation

While backpropagation is powerful it does face some challenges:

1. **Vanishing Gradient Problem**: In deep networks the gradients can become very small during backpropagation making it difficult for the network to learn. This is common when using activation functions like sigmoid or tanh.
2. **Exploding Gradients**: The gradients can also become excessively large causing the network to diverge during training.
3. **Overfitting**: If the network is too complex it might memorize the training data instead of learning general patterns.

Backpropagation is a technique that makes neural network learn. By propagating errors backward and adjusting the weights and biases neural networks can gradually improve their predictions. Though it has some limitations like vanishing gradients many techniques like ReLU activation or optimizing learning rates have been developed to address these issues.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/)

[K-Nearest Neighbors and Curse of Dimensionality](https://www.geeksforgeeks.org/k-nearest-neighbors-and-curse-of-dimensionality/)

[![author](https://media.geeksforgeeks.org/auth/profile/atq6bpb8l6ppj6778btz)](https://www.geeksforgeeks.org/user/tejashreeganesan/)

[tejashreeganesan](https://www.geeksforgeeks.org/user/tejashreeganesan/)

Follow

19

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Python-PyTorch](https://www.geeksforgeeks.org/tag/python-pytorch/)
- [ML-Statistics](https://www.geeksforgeeks.org/tag/ml-statistics/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Backpropagation in Convolutional Neural Networks\\
\\
\\
Convolutional Neural Networks (CNNs) have become the backbone of many modern image processing systems. Their ability to learn hierarchical representations of visual data makes them exceptionally powerful. A critical component of training CNNs is backpropagation, the algorithm used for effectively up\\
\\
4 min read](https://www.geeksforgeeks.org/backpropagation-in-convolutional-neural-networks/)
[Dropout in Neural Networks\\
\\
\\
The concept of Neural Networks is inspired by the neurons in the human brain and scientists wanted a machine to replicate the same process. This craved a path to one of the most important topics in Artificial Intelligence. A Neural Network (NN) is based on a collection of connected units or nodes ca\\
\\
3 min read](https://www.geeksforgeeks.org/dropout-in-neural-networks/)
[Applications of Neural Network\\
\\
\\
A neural network is a processing device, either an algorithm or genuine hardware, that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. The computing world has a ton to acquire from neural networks, also known as artific\\
\\
3 min read](https://www.geeksforgeeks.org/applications-of-neural-network/)
[Activation functions in Neural Networks\\
\\
\\
While building a neural network, one key decision is selecting the Activation Function for both the hidden layer and the output layer. Activation functions decide whether a neuron should be activated. Before diving into the activation function, you should have prior knowledge of the following topics\\
\\
8 min read](https://www.geeksforgeeks.org/activation-functions-neural-networks/)
[Batch Size in Neural Network\\
\\
\\
Batch size is a hyperparameter that determines the number of training records used in one forward and backward pass of the neural network. In this article, we will explore the concept of batch size, its impact on training, and how to choose the optimal batch size. Prerequisites: Neural Network, Grad\\
\\
5 min read](https://www.geeksforgeeks.org/batch-size-in-neural-network/)
[Effect of Bias in Neural Network\\
\\
\\
Neural Network is conceptually based on actual neuron of brain. Neurons are the basic units of a large neural network. A single neuron passes single forward based on input provided. In Neural network, some inputs are provided to an artificial neuron, and with each input a weight is associated. Weigh\\
\\
3 min read](https://www.geeksforgeeks.org/effect-of-bias-in-neural-network/)
[Backpropagation in Data Mining\\
\\
\\
Backpropagation is an algorithm that backpropagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors. It uses in the vast applications of neural networks in data mining like Character recognition, Signature verification, e\\
\\
6 min read](https://www.geeksforgeeks.org/backpropagation-in-data-mining/)
[Back Propagation through time - RNN\\
\\
\\
Introduction: Recurrent Neural Networks are those networks that deal with sequential data. They predict outputs using not only the current inputs but also by taking into consideration those that occurred before it. In other words, the current output depends on current output as well as a memory elem\\
\\
5 min read](https://www.geeksforgeeks.org/ml-back-propagation-through-time/)
[Feedback System in Neural Networks\\
\\
\\
A feedback system in neural networks is a mechanism where the output is fed back into the network to influence subsequent outputs, often used to enhance learning and stability. This article provides an overview of the working of the feedback loop in Neural Networks. Understanding Feedback SystemIn d\\
\\
6 min read](https://www.geeksforgeeks.org/feedback-system-in-neural-networks/)
[Auto-associative Neural Networks\\
\\
\\
Auto associative Neural networks are the types of neural networks whose input and output vectors are identical. These are special kinds of neural networks that are used to simulate and explore the associative process. Association in this architecture comes from the instruction of a set of simple pro\\
\\
3 min read](https://www.geeksforgeeks.org/auto-associative-neural-networks/)

Like19

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1858332077.1745056909&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1607124962)

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