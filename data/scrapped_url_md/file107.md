- [Python for Machine Learning](https://www.geeksforgeeks.org/python-for-machine-learning/)
- [Machine Learning with R](https://www.geeksforgeeks.org/introduction-to-machine-learning-in-r/)
- [Machine Learning Algorithms](https://www.geeksforgeeks.org/machine-learning-algorithms/)
- [EDA](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/)
- [Math for Machine Learning](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-projects/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Computer vision](https://www.geeksforgeeks.org/computer-vision/)
- [Data Science](https://www.geeksforgeeks.org/data-science-for-beginners/)
- [Artificial Intelligence](https://www.geeksforgeeks.org/artificial-intelligence/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/?type%3Darticle%26id%3D273757&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Q-Learning in Reinforcement Learning\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/q-learning-in-python/)

# Gradient Descent Algorithm in Machine Learning

Last Updated : 23 Jan, 2025

Comments

Improve

Suggest changes

59 Likes

Like

Report

**Gradient descent** is the backbone of the learning process for various algorithms, including linear regression, logistic regression, support vector machines, and neural networks which serves as a fundamental optimization technique to minimize the cost function of a model by **iteratively adjusting the model parameters to reduce the difference between predicted and actual values, improving the model’s performance**. Let’s see it’s role in machine learning:

**Prerequisites**: Understand the working and math of gradient descent.

### 1\. Training Machine Learning Models

[Neural networks](https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/) are trained using Gradient Descent (or its variants) in combination with [backpropagation](https://www.geeksforgeeks.org/backpropagation-in-neural-network/). Backpropagation computes the gradients of the **loss function with respect to each parameter (weights and biases) in the network by applying the** [**chain rule**](https://www.geeksforgeeks.org/chain-rule-derivative-in-machine-learning/) **.** The process involves:

- **Forward Propagation**: Computes the output for a given input by passing data through the layers.
- **Backward Propagation**: Uses the chain rule to calculate gradients of the loss with respect to each parameter (weights and biases) across all layers.

_**Gradients are then used by Gradient Descent to update the parameters layer-by-layer, moving toward minimizing the loss function.**_

> Neural networks often use advanced variants of Gradient Descent. If you want to read more about variants, please refer : [Gradient Descent Variants](https://www.geeksforgeeks.org/different-variants-of-gradient-descent/).

## 2\. Minimizing the Cost Function

The algorithm minimizes a cost function, which quantifies the error or loss of the model’s predictions compared to the true labels for:

### **1\. Linear Regression**

Gradient descent minimizes the [Mean Squared Error (MSE)](https://www.geeksforgeeks.org/mean-squared-error/) which serves as the loss function to find the best-fit line. Gradient Descent is used to iteratively update the weights (coefficients) and bias by computing the gradient of the MSE with respect to these parameters.

Since MSE is a convex function **gradient descent guarantees convergence to the global minimum if the learning rate is appropriately chosen.** For each iteration:

The algorithm computes the gradient of the MSE with respect to the weights and biases.

It updates the weights (w) and bias (b) using the formula:

- Calculating the gradient of the log-loss with respect to the weights.
- Updating weights and biases iteratively to maximize the likelihood of the correct classification:

w=w–α⋅∂J(w,b)∂w,b=b–α⋅∂J(w,b)∂bw = w – \\alpha \\cdot \\frac{\\partial J(w, b)}{\\partial w}, \\quad b = b – \\alpha \\cdot \\frac{\\partial J(w, b)}{\\partial b}w=w–α⋅∂w∂J(w,b)​,b=b–α⋅∂b∂J(w,b)​

The formula is the **parameter update rule for gradient descent**, which adjusts the weights w and biases b to minimize a cost function. This process iteratively adjusts the line’s slope and intercept to minimize the error.

### **2\. Logistic Regression**

In logistic regression, gradient descent minimizes the [**Log Loss (Cross-Entropy Loss)**](https://www.geeksforgeeks.org/binary-cross-entropy-log-loss-for-binary-classification/) to optimize the decision boundary for binary classification. Since the output is probabilistic (between 0 and 1), the sigmoid function is applied. The process involves:

- Calculating the gradient of the log-loss with respect to the weights.
- Updating weights and biases iteratively to maximize the likelihood of the correct classification:

w=w–α⋅∂J(w)∂ww = w – \\alpha \\cdot \\frac{\\partial J(w)}{\\partial w}w=w–α⋅∂w∂J(w)​

This adjustment shifts the decision boundary to separate classes more effectively.

### **3\. Support Vector Machines (SVMs)**

For SVMs, gradient descent optimizes the [**hinge loss**](https://www.geeksforgeeks.org/hinge-loss-relationship-with-support-vector-machines/), which ensures a maximum-margin hyperplane. The algorithm:

- Calculates gradients for the hinge loss and the regularization term (if used, such as L2 regularization).
- Updates the weights to maximize the margin between classes while minimizing misclassification penalties with same formula provided above.

Gradient descent ensures the **optimal placement of the hyperplane to separate classes with the largest possible margin.**

## Gradient Descent Python Implementation

Diving further into the concept, let’s understand in depth, with practical implementation.

#### Import the necessary libraries

Python`
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
`

#### Set the input and output data

Python`
# set random seed for reproducibility
torch.manual_seed(42)
# set number of samples
num_samples = 1000
# create random features with 2 dimensions
x = torch.randn(num_samples, 2)
# create random weights and bias for the linear regression model
true_weights = torch.tensor([1.3, -1])
true_bias    = torch.tensor([-3.5])
# Target variable
y = x @ true_weights.T + true_bias
# Plot the dataset
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(x[:,0],y)
ax[1].scatter(x[:,1],y)
ax[0].set_xlabel('X1')
ax[0].set_ylabel('Y')
ax[1].set_xlabel('X2')
ax[1].set_ylabel('Y')
plt.show()
`

**Output**:

![X vs Y - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230413102209/download-(6).png)

X vs Y

#### Let’s first try with a linear model:

```
yp=xWT+by_p = xW^T+byp​=xWT+b
```

Python`
# Define the model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.linear(x)
        return out
# Define the input and output dimensions
input_size = x.shape[1]
output_size = 1
# Instantiate the model
model = LinearRegression(input_size, output_size)
`

#### **Note:**

The number of weight values will be equal to the input size of the model, And the input size in deep Learning is the number of independent input features i.e we are putting inside the model

In our case, input features are two so, the input size will also be two, and the corresponding weight value will also be two.

#### We can manually set the model parameter

Python`
# create a random weight & bias tensor
weight = torch.randn(1, input_size)
bias   = torch.rand(1)
# create a nn.Parameter object from the weight & bias tensor
weight_param = nn.Parameter(weight)
bias_param   = nn.Parameter(bias)
# assign the weight & bias parameter to the linear layer
model.linear.weight = weight_param
model.linear.bias   = bias_param
weight, bias = model.parameters()
print('Weight :',weight)
print('bias :',bias)
`

**Output**:

```
Weight : Parameter containing:
tensor([[-0.3239,  0.5574]], requires_grad=True)
bias : Parameter containing:
tensor([0.5710], requires_grad=True)

```

#### Prediction

Python`
y_p = model(x)
y_p[:5]
`

**Output**:

```
tensor([[ 0.7760],\
        [-0.8944],\
        [-0.3369],\
        [-0.3095],\
        [ 1.7338]], grad_fn=<SliceBackward0>)

```

### Define the loss function

```
Loss function (J)=1n∑(actual−predicted)2\text{Loss function (J)} =\frac{1}{n} \sum{(actual-predicted)^{2}}Loss function (J)=n1​∑(actual−predicted)2
```

Here we are calculating the Mean Squared Error by taking the square of the difference between the actual and the predicted value and then dividing it by its length (i.e n = the Total number of output or target values) which is the mean of squared errors.

Python`
# Define the loss function
def Mean_Squared_Error(prediction, actual):
    error = (actual-prediction)**2
    return error.mean()
# Find the total mean squared error
loss = Mean_Squared_Error(y_p, y)
loss
`

**Output**:

```
tensor(19.9126, grad_fn=<MeanBackward0>)

```

As we can see from the above right now the Mean Squared Error is 30559.4473. All the steps which are done till now are known as forward propagation.

Now our task is to find the optimal value of weight w and bias b which can fit our model well by giving very less or minimum error as possible. i.e

```
minimize  1n∑(actual−predicted)2\text{minimize}\; \frac{1}{n} \sum{(actual-predicted)^{2}}minimizen1​∑(actual−predicted)2
```

Now to update the weight and bias value and find the optimal value of weight and bias we will do backpropagation. Here the Gradient Descent comes into the role to find the optimal value weight and bias.

## How the Gradient Descent Algorithm Works

For the sake of complexity, we can write our loss function for the single row as below

```
J(w,b)=1n(yp−y)2J(w, b) = \frac{1}{n} (y_p-y)^2J(w,b)=n1​(yp​−y)2
```

In the above function x and y are our input data i.e constant. To find the optimal value of weight w and bias b. we partially differentiate with respect to w and b. This is also said that we will find the gradient of loss function J(w,b) with respect to w and b to find the optimal value of w and b.

#### Gradient of J(w,b) with respect to w

J’w=∂J(w,b)∂w=∂∂w\[1n(yp−y)2\]=2(yp−y)n∂∂w\[(yp−y)\]=2(yp−y)n∂∂w\[((xWT+b)−y)\]=2(yp−y)n\[∂(xWT+b)∂w−∂(y)∂w\]=2(yp−y)n\[x–0\]=1n(yp−y)\[2x\]\\begin {aligned} {J}’\_w &=\\frac{\\partial J(w,b)}{\\partial w} \\\ &= \\frac{\\partial}{\\partial w} \\left\[\\frac{1}{n} (y\_p-y)^2 \\right\] \\\ &= \\frac{2(y\_p-y)}{n}\\frac{\\partial}{\\partial w}\\left \[(y\_p-y)  \\right \] \\\ &= \\frac{2(y\_p-y)}{n}\\frac{\\partial}{\\partial w}\\left \[((xW^T+b)-y)  \\right \] \\\ &= \\frac{2(y\_p-y)}{n}\\left\[\\frac{\\partial(xW^T+b)}{\\partial w}-\\frac{\\partial(y)}{\\partial w}\\right\] \\\ &= \\frac{2(y\_p-y)}{n}\\left \[ x – 0 \\right \] \\\ &= \\frac{1}{n}(y\_p-y)\[2x\] \\end {aligned}J’w​​=∂w∂J(w,b)​=∂w∂​\[n1​(yp​−y)2\]=n2(yp​−y)​∂w∂​\[(yp​−y)\]=n2(yp​−y)​∂w∂​\[((xWT+b)−y)\]=n2(yp​−y)​\[∂w∂(xWT+b)​−∂w∂(y)​\]=n2(yp​−y)​\[x–0\]=n1​(yp​−y)\[2x\]​

i.e

J’w=∂J(w,b)∂w=J(w,b)\[2x\]\\begin {aligned} {J}’\_w &= \\frac{\\partial J(w,b)}{\\partial w} \\\ &= J(w,b)\[2x\] \\end{aligned}J’w​​=∂w∂J(w,b)​=J(w,b)\[2x\]​

#### Gradient of J(w,b) with respect to b

J’b=∂J(w,b)∂b=∂∂b\[1n(yp−y)2\]=2(yp−y)n∂∂b\[(yp−y)\]=2(yp−y)n∂∂b\[((xWT+b)−y)\]=2(yp−y)n\[∂(xWT+b)∂b−∂(y)∂b\]=2(yp−y)n\[1–0\]=1n(yp−y)\[2\]\\begin {aligned} {J}’\_b &=\\frac{\\partial J(w,b)}{\\partial b} \\\ &= \\frac{\\partial}{\\partial b} \\left\[\\frac{1}{n} (y\_p-y)^2 \\right\] \\\ &= \\frac{2(y\_p-y)}{n}\\frac{\\partial}{\\partial b}\\left \[(y\_p-y)  \\right \] \\\ &= \\frac{2(y\_p-y)}{n}\\frac{\\partial}{\\partial b}\\left \[((xW^T+b)-y)  \\right \] \\\ &= \\frac{2(y\_p-y)}{n}\\left\[\\frac{\\partial(xW^T+b)}{\\partial b}-\\frac{\\partial(y)}{\\partial b}\\right\] \\\ &= \\frac{2(y\_p-y)}{n}\\left \[ 1 – 0 \\right \] \\\ &= \\frac{1}{n}(y\_p-y)\[2\] \\end {aligned}J’b​​=∂b∂J(w,b)​=∂b∂​\[n1​(yp​−y)2\]=n2(yp​−y)​∂b∂​\[(yp​−y)\]=n2(yp​−y)​∂b∂​\[((xWT+b)−y)\]=n2(yp​−y)​\[∂b∂(xWT+b)​−∂b∂(y)​\]=n2(yp​−y)​\[1–0\]=n1​(yp​−y)\[2\]​

i.e

J’b=∂J(w,b)∂b=J(w,b)\[2\]\\begin {aligned} {J}’\_b &= \\frac{\\partial J(w,b)}{\\partial b} \\\ &= J(w,b)\[2\] \\end{aligned}J’b​​=∂b∂J(w,b)​=J(w,b)\[2\]​

Here we have considered the linear regression. So that here the parameters are weight and bias only. But in a fully connected neural network model there can be multiple layers and multiple parameters.  but the concept will be the same everywhere. And the below-mentioned formula will work everywhere.

```
Param=Param−γ∇JParam = Param -\gamma \nabla JParam=Param−γ∇J
```

Here,

- γ\\gamma







γ = Learning rate
- J = Loss function
- ∇\\nabla







∇  = Gradient symbol denotes the derivative of loss function J
- Param = weight and bias     There can be multiple weight and bias values depending upon the complexity of the model and features in the dataset

In our case:

```
w=w−γ∇J(w,,b)b  =b−γ∇J(w,,b)w = w -\gamma \nabla J(w,,b)\\b \;= b -\gamma \nabla J(w,,b)w=w−γ∇J(w,,b)b=b−γ∇J(w,,b)
```

In the current problem, two input features, So, the weight will be two.

### Implementations of the Gradient Descent algorithm for the above model

Steps:

1.  Find the gradient using loss.backward()
2. Get the parameter using model.linear.weight and model.linear.bias
3. Update the parameter using the above-defined equation.
4. Again assign the model parameter to our model

```
# Find the gradient using
loss.backward()
# Learning Rate
learning_rate = 0.001
# Model Parameter
w = model.linear.weight
b = model.linear.bias
# Matually Update the model parameter
w = w - learning_rate * w.grad
b = b - learning_rate * b.grad
# assign the weight & bias parameter to the linear layer
model.linear.weight = nn.Parameter(w)
model.linear.bias   = nn.Parameter(b)

```

#### Now Repeat this process till 1000 epochs

Python`
# Number of epochs
num_epochs = 1000
# Learning Rate
learning_rate = 0.01
# SUBPLOT WEIGHT & BIAS VS lOSSES
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
for epoch in range(num_epochs):
    # Forward pass
    y_p = model(x)
    loss = Mean_Squared_Error(y_p, y)

    # Backproogation
    # Find the fradient using
    loss.backward()
    # Learning Rate
    learning_rate = 0.001
    # Model Parameter
    w = model.linear.weight
    b = model.linear.bias
    # Matually Update the model parameter
    w = w - learning_rate * w.grad
    b = b - learning_rate * b.grad
    # assign the weight & bias parameter to the linear layer
    model.linear.weight = nn.Parameter(w)
    model.linear.bias   = nn.Parameter(b)

    if (epoch+1) % 100 == 0:
        ax1.plot(w.detach().numpy(),loss.item(),'r*-')
        ax2.plot(b.detach().numpy(),loss.item(),'g+-')
        print('Epoch [{}/{}], weight:{}, bias:{} Loss: {:.4f}'.format(
            epoch+1,num_epochs,
            w.detach().numpy(),
            b.detach().numpy(),
            loss.item()))

ax1.set_xlabel('weight')
ax2.set_xlabel('bias')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Loss')
plt.show()
`

**Output**:

```
Epoch [100/1000], weight:[[-0.2618025   0.44433367]], bias:[-0.17722966] Loss: 14.1803
Epoch [200/1000], weight:[[-0.21144074  0.35393423]], bias:[-0.7892358] Loss: 10.3030
Epoch [300/1000], weight:[[-0.17063744  0.28172654]], bias:[-1.2897989] Loss: 7.7120
Epoch [400/1000], weight:[[-0.13759881  0.22408141]], bias:[-1.699218] Loss: 5.9806
Epoch [500/1000], weight:[[-0.11086453  0.17808875]], bias:[-2.0340943] Loss: 4.8235
Epoch [600/1000], weight:[[-0.08924612  0.14141548]], bias:[-2.3080034] Loss: 4.0502
Epoch [700/1000], weight:[[-0.0717768   0.11219224]], bias:[-2.5320508] Loss: 3.5333
Epoch [800/1000], weight:[[-0.0576706   0.08892148]], bias:[-2.7153134] Loss: 3.1878
Epoch [900/1000], weight:[[-0.04628877  0.07040432]], bias:[-2.8652208] Loss: 2.9569
Epoch [1000/1000], weight:[[-0.0371125   0.05568104]], bias:[-2.9878428] Loss: 2.8026

```

![Weight & Bias vs Losses - Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230413102015/download-(5).png)

Weight & Bias vs Losses – Geeksforgeeks

From the above graph and data, we can observe the Losses are decreasing as per the weight and bias variations.

Now we have found the optimal weight and bias values. Print the optimal weight and bias and

Python`
w = model.linear.weight
b = model.linear.bias
print('weight(W) = {} \n  bias(b) = {}'.format(
w.abs(),
b.abs()))
`

**Output:**

```
weight(W) = tensor([[0.0371, 0.0557]], grad_fn=<AbsBackward0>)
  bias(b) = tensor([2.9878], grad_fn=<AbsBackward0>)

```

#### Prediction

Python`
pred =  x @ w.T + b
pred[:5]
`

**Output:**

```
tensor([[-2.9765],\
        [-3.1385],\
        [-3.0818],\
        [-3.0756],\
        [-2.8681]], grad_fn=<SliceBackward0>)

```

## Gradient Descent Learning Rate

The [learning rate](https://www.geeksforgeeks.org/impact-of-learning-rate-on-a-model/) is a critical hyperparameter in the context of gradient descent, influencing the size of steps taken during the optimization process to update the model parameters. Choosing an appropriate learning rate is crucial for efficient and effective model training.

When the learning rate is **too small**, the optimization process progresses very slowly. The model makes tiny updates to its parameters in each iteration, leading to sluggish convergence and potentially getting stuck in local minima.

On the other hand, an **excessively large learning rate** can cause the optimization algorithm to overshoot the optimal parameter values, leading to divergence or oscillations that hinder convergence.

Achieving the right balance is essential. A small learning rate might result in vanishing gradients and slow convergence, while a large learning rate may lead to overshooting and instability.

## Vanishing and Exploding Gradients

[Vanishing and exploding gradients](https://www.geeksforgeeks.org/vanishing-and-exploding-gradients-problems-in-deep-learning/) are common problems that can occur during the training of deep neural networks. These problems can significantly slow down the training process or even prevent the network from learning altogether.

The vanishing gradient problem occurs when gradients become too small during backpropagation. The weights of the network are not considerably changed as a result, and the network is unable to discover the underlying patterns in the data. Many-layered deep neural networks are especially prone to this issue. The gradient values fall exponentially as they move backward through the layers, making it challenging to efficiently update the weights in the earlier layers.

The exploding gradient problem, on the other hand, occurs when gradients become too large during backpropagation. When this happens, the weights are updated by a large amount, which can cause the network to diverge or oscillate, making it difficult to converge to a good solution.

#### To address these problems the following technique can be used:

- **Weights Regularzations:** The initialization of weights can be adjusted to ensure that they are in an appropriate range. Using a different activation function, such as the Rectified Linear Unit (ReLU), can also help to mitigate the vanishing gradient problem.
- **Gradient clipping:** It involves limiting the maximum and minimum values of the gradient during backpropagation. This can prevent the gradients from becoming too large or too small and can help to stabilize the training process.
- **Batch normalization:** It can also help to address these problems by normalizing the input to each layer, which can prevent the activation function from saturating and help to reduce the vanishing and exploding gradient problems.

## Different Variants of Gradient Descent

There are several variants of gradient descent that differ in the way the step size or learning rate is chosen and the way the updates are made. Here are some popular variants:

### **Batch Gradient Descent**

In [batch gradient descent](https://www.geeksforgeeks.org/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/), To update the model parameter values like weight and bias, the entire training dataset is used to compute the gradient and update the parameters at each iteration. This can be slow for large datasets but may lead to a more accurate model. It is effective for convex or relatively smooth error manifolds because it moves directly toward an optimal solution by taking a large step in the direction of the negative gradient of the cost function. However, it can be slow for large datasets because it computes the gradient and updates the parameters using the entire training dataset at each iteration. This can result in longer training times and higher computational costs.

### **Stochastic Gradient Descent (SGD)**

In [SGD](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/), only one training example is used to compute the gradient and update the parameters at each iteration. This can be faster than batch gradient descent but may lead to more noise in the updates.

### **Mini-batch Gradient Descent**

**I** n [Mini-batch gradient descent](https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/) a small batch of training examples is used to compute the gradient and update the parameters at each iteration. This can be a good compromise between batch gradient descent and Stochastic Gradient Descent, as it can be faster than batch gradient descent and less noisy than Stochastic Gradient Descent.

### **Momentum-based Gradient Descent**

In [momentum-based gradient descent](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/), Momentum is a variant of gradient descent that incorporates information from the previous weight updates to help the algorithm converge more quickly to the optimal solution. Momentum adds a term to the weight update that is proportional to the running average of the past gradients, allowing the algorithm to move more quickly in the direction of the optimal solution. The updates to the parameters are based on the current gradient and the previous updates. This can help prevent the optimization process from getting stuck in local minima and reach the global minimum faster.

### **Nesterov Accelerated Gradient (NAG)**

Nesterov Accelerated Gradient (NAG) is an extension of Momentum Gradient Descent. It evaluates the gradient at a hypothetical position ahead of the current position based on the current momentum vector, instead of evaluating the gradient at the current position. This can result in faster convergence and better performance.

### **Adagrad**

In [Adagrad](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/), the learning rate is adaptively adjusted for each parameter based on the historical gradient information. This allows for larger updates for infrequent parameters and smaller updates for frequent parameters.

### **RMSprop**

In [RMSprop](https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/) the learning rate is adaptively adjusted for each parameter based on the moving average of the squared gradient. This helps the algorithm to converge faster in the presence of noisy gradients.

### **Adam**

[Adam](https://www.geeksforgeeks.org/intuition-of-adam-optimizer/) stands for adaptive moment estimation, it combines the benefits of Momentum-based Gradient Descent, Adagrad, and RMSprop the learning rate is adaptively adjusted for each parameter based on the moving average of the gradient and the squared gradient, which allows for faster convergence and better performance on non-convex optimization problems. It keeps track of two exponentially decaying averages the first-moment estimate, which is the exponentially decaying average of past gradients, and the second-moment estimate, which is the exponentially decaying average of past squared gradients. The first-moment estimate is used to calculate the momentum, and the second-moment estimate is used to scale the learning rate for each parameter. This is one of the most popular optimization algorithms for deep learning.

## Advantages & Disadvantages of gradient descent

### Advantages of Gradient Descent

1. **Widely used:** Gradient descent and its variants are widely used in machine learning and optimization problems because they are effective and easy to implement.
2. **Convergence**: Gradient descent and its variants can converge to a global minimum or a good local minimum of the cost function, depending on the problem and the variant used.
3. **Scalability**: Many variants of gradient descent can be parallelized and are scalable to large datasets and high-dimensional models.
4. **Flexibility**: Different variants of gradient descent offer a range of trade-offs between accuracy and speed, and can be adjusted to optimize the performance of a specific problem.

### Disadvantages of gradient descent:

1. **Choice of learning rate:** The choice of learning rate is crucial for the convergence of gradient descent and its variants. Choosing a learning rate that is too large can lead to oscillations or overshooting while choosing a learning rate that is too small can lead to slow convergence or getting stuck in local minima.
2. **Sensitivity to initialization:** Gradient descent and its variants can be sensitive to the initialization of the model’s parameters, which can affect the convergence and the quality of the solution.
3. **Time-consuming:** Gradient descent and its variants can be time-consuming, especially when dealing with large datasets and high-dimensional models. The convergence speed can also vary depending on the variant used and the specific problem.
4. **Local optima:** Gradient descent and its variants can converge to a local minimum instead of the global minimum of the cost function, especially in non-convex problems. This can affect the quality of the solution, and techniques like random initialization and multiple restarts may be used to mitigate this issue.

## Conclusion

In the intricate landscape of machine learning and deep learning, the journey of model optimization revolves around the foundational concept of gradient descent and its diverse variants. Through the lens of this powerful optimization algorithm, we explored the intricacies of minimizing the cost function, a pivotal task in training models.

[iframe](https://cdnads.geeksforgeeks.org/instream/video.html)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/q-learning-in-python/)

[Q-Learning in Reinforcement Learning](https://www.geeksforgeeks.org/q-learning-in-python/)

[C](https://www.geeksforgeeks.org/user/cs16011997/)

[cs16011997](https://www.geeksforgeeks.org/user/cs16011997/)

Follow

59

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Data Science](https://www.geeksforgeeks.org/category/ai-ml-ds/data-science/)
- [Deep Learning](https://www.geeksforgeeks.org/tag/deep-learning/)
- [Machine Learning](https://www.geeksforgeeks.org/tag/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Gradient Descent Algorithm in R\\
\\
\\
Gradient Descent is a fundamental optimization algorithm used in machine learning and statistics. It is designed to minimize a function by iteratively moving toward the direction of the steepest descent, as defined by the negative of the gradient. The goal is to find the set of parameters that resul\\
\\
7 min read](https://www.geeksforgeeks.org/gradient-descent-algorithm-in-r/)
[Mini-Batch Gradient Descent in Deep Learning\\
\\
\\
Mini-batch gradient descent is a variant of the traditional gradient descent algorithm used to optimize the parameters (weights and biases) of a neural network. It divides the training data into small subsets called mini-batches, allowing the model to update its parameters more frequently compared t\\
\\
7 min read](https://www.geeksforgeeks.org/mini-batch-gradient-descent-in-deep-learning/)
[First-Order algorithms in machine learning\\
\\
\\
First-order algorithms are a cornerstone of optimization in machine learning, particularly for training models and minimizing loss functions. These algorithms are essential for adjusting model parameters to improve performance and accuracy. This article delves into the technical aspects of first-ord\\
\\
7 min read](https://www.geeksforgeeks.org/first-order-algorithms-in-machine-learning/)
[Machine Learning Algorithms Cheat Sheet\\
\\
\\
This guide is a go to guide for machine learning algorithms for your interview and test preparation. This guide briefly describes key points and applications for each algorithm. This article provides an overview of key algorithms in each category, their purposes, and best use-cases. Types of Machine\\
\\
6 min read](https://www.geeksforgeeks.org/machine-learning-algorithms-cheat-sheet/)
[Tree Based Machine Learning Algorithms\\
\\
\\
Tree-based algorithms are a fundamental component of machine learning, offering intuitive decision-making processes akin to human reasoning. These algorithms construct decision trees, where each branch represents a decision based on features, ultimately leading to a prediction or classification. By\\
\\
14 min read](https://www.geeksforgeeks.org/tree-based-machine-learning-algorithms/)
[Optimization Algorithms in Machine Learning\\
\\
\\
Optimization algorithms are the backbone of machine learning models as they enable the modeling process to learn from a given data set. These algorithms are used in order to find the minimum or maximum of an objective function which in machine learning context stands for error or loss. In this artic\\
\\
15+ min read](https://www.geeksforgeeks.org/optimization-algorithms-in-machine-learning/)
[Chain Rule Derivative in Machine Learning\\
\\
\\
In machine learning, understanding the chain rule and its application in computing derivatives is essential. The chain rule allows us to find the derivative of composite functions, which frequently arise in machine learning models due to their layered architecture. These models often involve multipl\\
\\
5 min read](https://www.geeksforgeeks.org/chain-rule-derivative-in-machine-learning/)
[Exploitation and Exploration in Machine Learning\\
\\
\\
Exploration and Exploitation are methods for building effective learning algorithms that can adapt and perform optimally in different environments. This article focuses on exploitation and exploration in machine learning, and it elucidates various techniques involved. Table of Content Understanding\\
\\
8 min read](https://www.geeksforgeeks.org/exploitation-and-exploration-in-machine-learning/)
[Interpolation in Machine Learning\\
\\
\\
In machine learning, interpolation refers to the process of estimating unknown values that fall between known data points. This can be useful in various scenarios, such as filling in missing values in a dataset or generating new data points to smooth out a curve. In this article, we are going to exp\\
\\
7 min read](https://www.geeksforgeeks.org/interpolation-in-machine-learning/)
[Gradient Descent in Linear Regression\\
\\
\\
Gradient descent is a optimization algorithm used in linear regression to minimize the error in predictions. This article explores how gradient descent works in linear regression. Why Gradient Descent in Linear Regression?Linear regression involves finding the best-fit line for a dataset by minimizi\\
\\
4 min read](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/)

Like59

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2036106708.1745056920&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1681785009)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745056920140&cv=11&fst=1745056920140&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3h1v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fgradient-descent-algorithm-and-its-variants%2F&hn=www.googleadservices.com&frm=0&tiba=Gradient%20Descent%20Algorithm%20in%20Machine%20Learning%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1260175303.1745056920&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)