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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/rmsprop-optimizer-in-deep-learning/?type%3Darticle%26id%3D1292649&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Can AI replace Flutter developers ?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/can-ai-replace-flutter-developers/)

# RMSProp Optimizer in Deep Learning

Last Updated : 03 Mar, 2025

Comments

Improve

Suggest changes

2 Likes

Like

Report

**RMSProp (Root Mean Square Propagation)** is an adaptive learning rate optimization algorithm designed to improve the performance and speed of training deep learning models.

- It is a variant of the [**gradient descent**](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) algorithm, which adapts the learning rate for each parameter individually by considering the magnitude of recent gradients for those parameters.
- This adaptive nature helps in dealing with the challenges of non-stationary objectives and sparse gradients commonly encountered in deep learning tasks.

### Need of RMSProp Optimizer

RMSProp was developed to address the limitations of previous optimization methods such as [**SGD (Stochastic Gradient Descent)**](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/) and [**AdaGrad**](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/) **.**

While SGD uses a constant learning rate, which can be inefficient, and AdaGrad reduces the learning rate too aggressively, RMSProp strikes a balance by adapting the learning rates based on a moving average of squared gradients. This approach helps in maintaining a balance between efficient convergence and stability during the training process, making RMSProp a widely used optimization algorithm in modern deep learning.

## How RMSProp Works?

RMSProp keeps a moving average of the squared gradients to normalize the gradient updates. By doing so, RMSProp prevents the learning rate from becoming too small, which was a drawback in AdaGrad, and ensures that the updates are appropriately scaled for each parameter. This mechanism allows RMSProp to perform well even in the presence of non-stationary objectives, making it suitable for training deep learning models.

The mathematical formulation is as follows:

1. Compute the gradient gtg\_tgt​ at time step t: gt=∇θg\_t = \\nabla\_{\\theta} gt​=∇θ​
2. Update the moving average of squared gradients E\[g2\]t=γE\[g2\]t−1+(1−γ)E\[g^2\]\_t = \\gamma E\[g^2\]\_{t-1} + (1 - \\gamma) E\[g2\]t​=γE\[g2\]t−1​+(1−γ)​where γ\\gammaγ is the decay rate.
3. Update the parameter θ\\thetaθ using the adjusted learning rate: θt+1=θt−ηE\[g2\]t+ϵ\\theta\_{t+1} = \\theta\_t - \\frac{\\eta}{\\sqrt{E\[g^2\]\_t + \\epsilon}}θt+1​=θt​−E\[g2\]t​+ϵ​η​ ​where η\\etaη is the learning rate and ϵ\\epsilonϵ is a small constant added for numerical stability.

### Key Parameters Involved in RMSProp

- **Learning Rate (** η\\etaη **)**: Controls the step size during the parameter updates. RMSProp typically uses a default learning rate of 0.001, but it can be adjusted based on the specific problem.
- **Decay Rate (** γ\\gammaγ **)**: Determines how quickly the moving average of squared gradients decays. A common default value is 0.9, which balances the contribution of recent and past gradients.
- **Epsilon (** ϵ\\epsilonϵ **)**: A small constant added to the denominator to prevent division by zero and ensure numerical stability. A typical value for ϵ\\epsilonϵ is 1e-8.

By carefully adjusting these parameters, RMSProp effectively adapts the learning rates during training, leading to faster and more reliable convergence in deep learning models.

## RMSProp Optimization Algorithm on Linear Regression and Himmelblau's Function

In this section, we are going to demonstrate the application of RMSProp optimization algorithm to both [linear regression](https://www.geeksforgeeks.org/ml-linear-regression/) and non-convex optimization problem highlighting its versatility and effectiveness.

To illustrate RMSProp's performance in finding multiple global and local minima, we can create a visualization similar to the one you've provided. We'll use a two-dimensional function with multiple minima, such as the Himmelblau's function, to demonstrate the optimization process.

Himmelblau's function has several local minima and is defined as:

f(x,y)=(x2+y−11)2+(x+y2−7)2f(x,y) = (x^2 + y -11) ^ 2 + (x+y^2 -7)^2f(x,y)=(x2+y−11)2+(x+y2−7)2

Let's discuss the steps to visualize the RMSProp optimization on Himmelblau's function.

### Step 1: Import Necessary Libraries

First, we import the libraries needed for numerical computations and plotting.

Python`
import numpy as np
import matplotlib.pyplot as plt
`

### Step 2: Compute Gradient for Linear Regression

Define a function to compute the gradient of the cost function with respect to the parameters (theta) for linear regression.

Python`
# Function to compute gradient
def compute_gradient(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    return gradient
`

### Step 3: Implement RMSProp for Linear Regression

Define the RMSProp algorithm to update parameters for linear regression.

Python`
def rmsprop(X, y, theta, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, iterations=5000):
    E_g2 = np.zeros(theta.shape)
    cost_history = []

    for i in range(iterations):
        # Compute gradient
        g = compute_gradient(theta, X, y)

        # Update moving average of squared gradients
        E_g2 = decay_rate * E_g2 + (1 - decay_rate) * g**2

        # Update parameters
        theta = theta - learning_rate * g / (np.sqrt(E_g2) + epsilon)

        # Calculate and store the cost function value
        cost = (1/(2*len(y))) * np.sum((X.dot(theta) - y)**2)
        cost_history.append(cost)

        # Optional: Print the cost function
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history
`

### Step 4: Generate Example Data for Linear Regression

Create some synthetic data to test the linear regression model.

Python`
# Create example data for linear regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Add a bias column to X
X_b = np.c_[np.ones((100, 1)), X]
# Initialize parameters
theta = np.random.randn(2, 1)
`

### Step 5: Run RMSProp for Linear Regression

Run the RMSProp algorithm on the generated data and plot the cost function history.

Python`
# Run RMSProp
theta_final, cost_history = rmsprop(X_b, y, theta)
print("Final parameters:", theta_final)
# Plot the cost function history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence with RMSProp')
plt.grid(True)
plt.show()
`

### Step 6: Define Himmelblau's Function

Define Himmelblau's function, a common optimization test function, and its gradient.

Python`
# Define Himmelblau's function
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
# Compute the gradient of Himmelblau's function
def himmelblau_gradient(x):
    dfdx0 = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    dfdx1 = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return np.array([dfdx0, dfdx1])
`

### Step 7: Implement RMSProp for Himmelblau's Function

Define the RMSProp algorithm to minimize Himmelblau's function, tracking the optimization path.

Python`
def rmsprop_himmelblau(learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, iterations=5000):
    # Initialize variables
    x = np.random.rand(2) * 10 - 5
    E_g2 = np.zeros_like(x)
    path = [x.copy()]

    for i in range(iterations):
        # Compute gradient
        g = himmelblau_gradient(x)

        # Update moving average of squared gradients
        E_g2 = decay_rate * E_g2 + (1 - decay_rate) * g**2

        # Update parameters
        x = x - learning_rate * g / (np.sqrt(E_g2) + epsilon)

        # Store the path
        path.append(x.copy())

    return np.array(path)
`

### Step 8: Plot Optimization Path for Himmelblau's Function

Visualize the optimization path on Himmelblau's function.

Python`
def plot_himmelblau_rmsprop():
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50, cmap='jet')

    # Run RMSProp and plot the path
    path = rmsprop_himmelblau()
    plt.plot(path[:, 0], path[:, 1], 'o-', color='black', markersize=5, label='RMSProp')
    plt.scatter(path[-1, 0], path[-1, 1], color='red', marker='*', s=100, label='Final Point')

    plt.title("RMSProp Optimization Path on Himmelblau's Function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_himmelblau_rmsprop()
`

### Complete Code

Python`
import numpy as np
import matplotlib.pyplot as plt
# Function to compute gradient
def compute_gradient(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    return gradient
# RMSProp implementation with visualization
def rmsprop(X, y, theta, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, iterations=5000):
    E_g2 = np.zeros(theta.shape)
    cost_history = []

    for i in range(iterations):
        # Compute gradient
        g = compute_gradient(theta, X, y)

        # Update moving average of squared gradients
        E_g2 = decay_rate * E_g2 + (1 - decay_rate) * g**2

        # Update parameters
        theta = theta - learning_rate * g / (np.sqrt(E_g2) + epsilon)

        # Calculate and store the cost function value
        cost = (1/(2*len(y))) * np.sum((X.dot(theta) - y)**2)
        cost_history.append(cost)

        # Optional: Print the cost function value every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

    return theta, cost_history
# Example usage
# Create some example data for linear regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Add a bias column to X
X_b = np.c_[np.ones((100, 1)), X]
# Initialize parameters
theta = np.random.randn(2, 1)
# Run RMSProp
theta_final, cost_history = rmsprop(X_b, y, theta)
print("Final parameters:", theta_final)
# Plot the cost function history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence with RMSProp')
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
# Define Himmelblau's function
def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
# Compute the gradient of Himmelblau's function
def himmelblau_gradient(x):
    dfdx0 = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    dfdx1 = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return np.array([dfdx0, dfdx1])
# RMSProp implementation for Himmelblau's function
def rmsprop_himmelblau(learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, iterations=5000):
    # Initialize variables
    x = np.random.rand(2) * 10 - 5  # Random initialization in the range [-5, 5]
    E_g2 = np.zeros_like(x)
    path = [x.copy()]

    for i in range(iterations):
        # Compute gradient
        g = himmelblau_gradient(x)

        # Update moving average of squared gradients
        E_g2 = decay_rate * E_g2 + (1 - decay_rate) * g**2

        # Update parameters
        x = x - learning_rate * g / (np.sqrt(E_g2) + epsilon)

        # Store the path
        path.append(x.copy())

    return np.array(path)
# Plot the optimization path on Himmelblau's function
def plot_himmelblau_rmsprop():
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50, cmap='jet')

    # Run RMSProp and plot the path
    path = rmsprop_himmelblau()
    plt.plot(path[:, 0], path[:, 1], 'o-', color='black', markersize=5, label='RMSProp')
    plt.scatter(path[-1, 0], path[-1, 1], color='red', marker='*', s=100, label='Final Point')

    plt.title("RMSProp Optimization Path on Himmelblau's Function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_himmelblau_rmsprop()
`

**Output:**

> Iteration 0: Cost 15.46985518129688
>
> Iteration 100: Cost 14.31954930750629
>
> Iteration 200: Cost 13.287205984955156
>
> Iteration 300: Cost 12.29586798220484
>
> . . .
>
> **Final parameters:** \[\[4.21459616\]\
>\
> \[2.76961339\]\]

![Cost-Function-Convergence-with-RMSProp](https://media.geeksforgeeks.org/wp-content/uploads/20250303184548234418/Cost-Function-Convergence-with-RMSProp.png)

![download-(27)](https://media.geeksforgeeks.org/wp-content/uploads/20240724180219/download-(27).jpg)RMSProp Optimization Path on Himmelblau's Function

## Implementing RMSprop in Python using TensorFlow/Keras

The deep learning model is compiled with the RMSProp optimizer. We will use the following code line for initializing the RMSProp optimizer with hyperparameters:

> **tf.keras.optimizers.RMSprop(learning\_rate=0.001, rho=0.9)**

- **`learning_rate=0.001:`** ` Sets the step size for weight updates. Smaller learning rates result in smaller updates, helping to fine-tune weights and prevent overshooting the minimum loss.`
- **rho=0.9:** The discounting factor for the history of gradients, controlling the influence of past gradients on the current gradient computation.

Python`
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# Define the model
model = Sequential([\
    Flatten(input_shape=(28, 28)),\
    Dense(128, activation='relu'),\
    Dense(64, activation='relu'),\
    Dense(10, activation='softmax')\
])
# Compile the model with RMSprop optimizer
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model and store the history
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
# Plot the cost function graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cost Function Graph')
plt.legend()
plt.show()
`

**Output:**

```
Epoch 1/10
1500/1500 [==============================] - 15s 9ms/step - loss: 0.2709 - accuracy: 0.9210 - val_loss: 0.1407 - val_accuracy: 0.9598
.
.
.
Epoch 10/10
1500/1500 [==============================] - 13s 8ms/step - loss: 0.0245 - accuracy: 0.9933 - val_loss: 0.1297 - val_accuracy: 0.9761
313/313 [==============================] - 1s 3ms/step - loss: 0.1039 - accuracy: 0.9759
Test accuracy: 0.9759
```

![download-(28)](https://media.geeksforgeeks.org/wp-content/uploads/20240724183102/download-(28).png)Cost Function Graph

## Advantages of RMSProp

- **Adaptive Learning Rates:** Adjusts learning rates for each parameter individually, optimizing updates more effectively.
- **Handles Non-Stationary Objectives:** Efficiently adapts to changing optimal parameter values over time.
- **Prevents Learning Rate Decay Problem:** Maintains optimal learning rates by using a decay rate, unlike AdaGrad.
- **Improved Convergence Speed:** Faster convergence due to balanced and dynamic learning rates.

## Disadvantages of RMSProp

- **Sensitivity to Hyperparameters:** Performance is sensitive to settings like decay rate and epsilon, requiring careful tuning.
- **Poor Performance with Sparse Data:** May struggle with sparse data, leading to slower or inconsistent convergence.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/can-ai-replace-flutter-developers/)

[Can AI replace Flutter developers ?](https://www.geeksforgeeks.org/can-ai-replace-flutter-developers/)

[A](https://www.geeksforgeeks.org/user/alka1974/)

[alka1974](https://www.geeksforgeeks.org/user/alka1974/)

Follow

2

Improve

Article Tags :

- [Blogathon](https://www.geeksforgeeks.org/category/blogathon/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Data Science Blogathon 2024](https://www.geeksforgeeks.org/tag/data-science-blogathon-2024/)

+1 More

### Similar Reads

[Deep Learning in R Programming\\
\\
\\
Deep Learning is a type of Artificial Intelligence or AI function that tries to imitate or mimic the working principle of a human brain for data processing and pattern creation for decision-making purposes. It is a subset of ML or machine learning in an AI that owns or have networks that are capable\\
\\
5 min read](https://www.geeksforgeeks.org/deep-learning-in-r-programming/?ref=ml_lbp)
[Neural Network Pruning in Deep Learning\\
\\
\\
As deep learning models have grown larger and more complex, they have also become more resource-intensive in terms of computational power and memory. In many real-world applications, especially on edge devices like mobile phones or embedded systems, these resource-heavy models are not feasible to de\\
\\
8 min read](https://www.geeksforgeeks.org/neural-network-pruning-in-deep-learning/?ref=ml_lbp)
[How Deep Learning is Useful in Real-Time?\\
\\
\\
Deep learning has emerged as a powerful tool with immense potential across various fields, particularly in real-time applications. From image recognition to natural language processing, deep learning algorithms have shown remarkable capabilities in processing vast amounts of data and making accurate\\
\\
4 min read](https://www.geeksforgeeks.org/how-deep-learning-is-useful-in-real-time/?ref=ml_lbp)
[Bayesian Optimization in Machine Learning\\
\\
\\
Bayesian Optimization is a powerful optimization technique that leverages the principles of Bayesian inference to find the minimum (or maximum) of an objective function efficiently. Unlike traditional optimization methods that require extensive evaluations, Bayesian Optimization is particularly effe\\
\\
8 min read](https://www.geeksforgeeks.org/bayesian-optimization-in-machine-learning/?ref=ml_lbp)
[Adam Optimizer in Tensorflow\\
\\
\\
Adam (Adaptive Moment Estimation) is an optimizer that combines the best features of two well-known optimizers: Momentum and RMSprop. Adam is used in deep learning due to its efficiency and adaptive learning rate capabilities. To use Adam in TensorFlow, we can pass the string value 'adam' to the opt\\
\\
3 min read](https://www.geeksforgeeks.org/adam-optimizer-in-tensorflow/?ref=ml_lbp)
[How to Utilize Hebbian Learning\\
\\
\\
Hebbian learning is a fundamental theory in neuroscience that describes how neurons adapt during the learning process. The core principle is often summarized as "neurons that fire together, wire together." This concept can be applied to artificial neural networks to train them like how biological br\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-utilize-hebbian-learning/?ref=ml_lbp)
[Training and Validation Loss in Deep Learning\\
\\
\\
In deep learning, loss functions are crucial in guiding the optimization process. The loss represents the discrepancy between the predicted output of the model and the actual target value. During training, models attempt to minimize this loss by adjusting their weights. Training loss and validation\\
\\
6 min read](https://www.geeksforgeeks.org/training-and-validation-loss-in-deep-learning/?ref=ml_lbp)
[List of Deep Learning Layers\\
\\
\\
Deep learning (DL) is characterized by the use of neural networks with multiple layers to model and solve complex problems. Each layer in the neural network plays a unique role in the process of converting input data into meaningful and insightful outputs. The article explores the layers that are us\\
\\
7 min read](https://www.geeksforgeeks.org/ml-list-of-deep-learning-layers/?ref=ml_lbp)
[Deep Q-Learning\\
\\
\\
Deep Q-Learning integrates deep neural networks into the decision-making process. This combination allows agents to handle high-dimensional state spaces, making it possible to solve complex tasks such as playing video games or controlling robots. Before diving into Deep Q-Learning, itâ€™s important to\\
\\
4 min read](https://www.geeksforgeeks.org/deep-q-learning/?ref=ml_lbp)
[Deep Learning Roadmap: A Structured Roadmap for Mastery\\
\\
\\
A deep learning roadmap is a structured guide designed to help individuals progress through the study of deep learning, from basic concepts to advanced applications. It serves as a comprehensive plan that outlines key areas of learning and development in deep learning. In this article we will discus\\
\\
5 min read](https://www.geeksforgeeks.org/deep-learning-roadmap/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/rmsprop-optimizer-in-deep-learning/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1690237761.1745057056&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102015666~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=1828147584)

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