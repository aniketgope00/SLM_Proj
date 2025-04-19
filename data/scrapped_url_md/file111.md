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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/adam-optimizer/?type%3Darticle%26id%3D488094&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Implementation of Radius Neighbors from Scratch in Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/implementation-of-radius-neighbors-from-sratch-in-python/)

# What is Adam Optimizer?

Last Updated : 04 Mar, 2025

Comments

Improve

Suggest changes

31 Likes

Like

Report

**Adam (short for Adaptive Moment Estimation)** optimizer combines the strengths of two other well-known techniques— **Momentum** and **RMSprop**—to deliver a powerful method for adjusting the learning rates of parameters during training.

Adam is highly effective, especially when working with large datasets and complex models, because it is memory-efficient and adapts the learning rate dynamically for each parameter.

## How Does Adam Work?

Adam builds upon two key concepts in optimization:

### 1\. Momentum

Momentum is used to accelerate the gradient descent process by incorporating an exponentially weighted moving average of past gradients. This helps smooth out the trajectory of the optimization, allowing the algorithm to converge faster by reducing oscillations.

The update rule with momentum is:

wt+1=wt–αmtw\_{t+1} = w\_{t} – \\alpha m\_{t}wt+1​=wt​–αmt​

where:

- mtm\_tmt​ is the moving average of the gradients at time ttt,
- ααα is the learning rate,
- wtw\_twt​​ and wt+1w\_{t+1}wt+1​​ are the weights at time ttt and t+1t+1t+1, respectively.

The momentum term mtm\_tmt​ is updated recursively as:

mt=β1mt−1+(1–β1)∂L∂wtm\_{t} = \\beta\_1 m\_{t-1} + (1 – \\beta\_1) \\frac{\\partial L}{\\partial w\_t}mt​=β1​mt−1​+(1–β1​)∂wt​∂L​

where:

- β1\\beta\_1β1​​ is the momentum parameter (typically set to 0.9),
- ∂L∂wt\\frac{\\partial L}{\\partial w\_t}∂wt​∂L​​ is the gradient of the loss function with respect to the weights at time ttt.

### 2\. RMSprop (Root Mean Square Propagation)

[RMSprop](https://www.geeksforgeeks.org/rmsprop-optimizer-in-deep-learning/) is an adaptive learning rate method that improves upon AdaGrad. While [AdaGrad](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/) accumulates squared gradients, RMSprop uses an exponentially weighted moving average of squared gradients, which helps overcome the problem of diminishing learning rates.

The update rule for RMSprop is:

wt+1=wt–αtvt+ϵ∂L∂wtw\_{t+1} = w\_{t} – \\frac{\\alpha\_t}{\\sqrt{v\_t + \\epsilon}} \\frac{\\partial L}{\\partial w\_t}wt+1​=wt​–vt​+ϵ​αt​​∂wt​∂L​

where:

- vtv\_tvt​​ is the exponentially weighted average of squared gradients:

vt=β2vt−1+(1–β2)(∂L∂wt)2v\_t = \\beta\_2 v\_{t-1} + (1 – \\beta\_2) \\left( \\frac{\\partial L}{\\partial w\_t} \\right)^2vt​=β2​vt−1​+(1–β2​)(∂wt​∂L​)2

- ϵϵϵ is a small constant (e.g., 10−810^{-8}10−8) added to prevent division by zero.

## Combining Momentum and RMSprop: Adam Optimizer

Adam optimizer combines the momentum and RMSprop techniques to provide a more balanced and efficient optimization process. The key equations governing Adam are as follows:

- **First moment (mean) estimate**:

mt=β1mt−1+(1–β1)∂L∂wtm\_t = \\beta\_1 m\_{t-1} + (1 – \\beta\_1) \\frac{\\partial L}{\\partial w\_t}mt​=β1​mt−1​+(1–β1​)∂wt​∂L​

- **Second moment (variance) estimate**:

vt=β2vt−1+(1–β2)(∂L∂wt)2v\_t = \\beta\_2 v\_{t-1} + (1 – \\beta\_2) \\left( \\frac{\\partial L}{\\partial w\_t} \\right)^2vt​=β2​vt−1​+(1–β2​)(∂wt​∂L​)2

- **Bias correction**: Since both mtm\_tmt​​ and vtv\_tvt​ are initialized at zero, they tend to be biased toward zero, especially during the initial steps. To correct this bias, Adam computes the bias-corrected estimates:

mt^=mt1–β1t,vt^=vt1–β2t\\hat{m\_t} = \\frac{m\_t}{1 – \\beta\_1^t}, \\quad \\hat{v\_t} = \\frac{v\_t}{1 – \\beta\_2^t}mt​^​=1–β1t​mt​​,vt​^​=1–β2t​vt​​

- **Final weight update**: The weights are then updated as:

wt+1=wt–mt^vt^+ϵαw\_{t+1} = w\_t – \\frac{\\hat{m\_t}}{\\sqrt{\\hat{v\_t}} + \\epsilon} \\alphawt+1​=wt​–vt​^​​+ϵmt​^​​α

### Key Parameters in Adam

- ααα: The learning rate or step size (default is 0.001).
- β1\\beta\_1β1​​ and β2\\beta\_2β2​​: Decay rates for the moving averages of the gradient and squared gradient, typically set to β1=0.9\\beta\_1 = 0.9β1​=0.9 and β2=0.999\\beta\_2 = 0.999β2​=0.999.
- ϵϵϵ: A small positive constant (e.g., 10−810^{-8}10−8) used to avoid division by zero when computing the final update.

## Why Adam Works So Well?

Adam addresses several challenges of gradient descent optimization:

- **Dynamic learning rates**: Each parameter has its own adaptive learning rate based on past gradients and their magnitudes. This helps the optimizer avoid oscillations and get past local minima more effectively.
- **Bias correction**: By adjusting for the initial bias when the first and second moment estimates are close to zero, Adam helps prevent early-stage instability.
- **Efficient performance**: Adam typically requires fewer hyperparameter tuning adjustments compared to other optimization algorithms like SGD, making it a more convenient choice for most problems.

## Performance of Adam

In comparison to other optimizers like SGD (Stochastic Gradient Descent) and momentum-based SGD, Adam outperforms them significantly in terms of both training time and convergence accuracy. Its ability to adjust the learning rate per parameter, combined with the bias-correction mechanism, leads to faster convergence and more stable optimization. This makes Adam especially useful in complex models with large datasets, as it avoids slow convergence and instability while reaching the global minimum.

In practice, Adam often achieves superior results with minimal tuning, making it a go-to optimizer for deep learning tasks.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200909204946/performance-660x641.png)

Performance Comparison on Training cost

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/implementation-of-radius-neighbors-from-sratch-in-python/)

[Implementation of Radius Neighbors from Scratch in Python](https://www.geeksforgeeks.org/implementation-of-radius-neighbors-from-sratch-in-python/)

[P](https://www.geeksforgeeks.org/user/prakharr0y/)

[prakharr0y](https://www.geeksforgeeks.org/user/prakharr0y/)

Follow

31

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)
- [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/)

### Similar Reads

[Adam Optimizer in Tensorflow\\
\\
\\
Adam (Adaptive Moment Estimation) is an optimizer that combines the best features of two well-known optimizers: Momentum and RMSprop. Adam is used in deep learning due to its efficiency and adaptive learning rate capabilities. To use Adam in TensorFlow, we can pass the string value 'adam' to the opt\\
\\
3 min read](https://www.geeksforgeeks.org/adam-optimizer-in-tensorflow/?ref=ml_lbp)
[Custom Optimizers in Pytorch\\
\\
\\
In PyTorch, an optimizer is a specific implementation of the optimization algorithm that is used to update the parameters of a neural network. The optimizer updates the parameters in such a way that the loss of the neural network is minimized. PyTorch provides various built-in optimizers such as SGD\\
\\
11 min read](https://www.geeksforgeeks.org/custom-optimizers-in-pytorch/?ref=ml_lbp)
[Dragon fly Optimization\\
\\
\\
Due to its simplicity, easy operation, capacity to protect against local optima, and the problem of derivatives free, Metaheuristic was frequently employed throughout the previous three decades. Exploration and exploitation are two fundamental metaheuristic features. The first one shows how the algo\\
\\
4 min read](https://www.geeksforgeeks.org/dragon-fly-optimization/?ref=ml_lbp)
[Optimizers in Tensorflow\\
\\
\\
Optimizers adjust weights of the model based on the gradient of loss function, aiming to minimize the loss and improve model accuracy. In TensorFlow, optimizers are available through tf.keras.optimizers. You can use these optimizers in your models by specifying them when compiling the model. Here's\\
\\
3 min read](https://www.geeksforgeeks.org/optimizers-in-tensorflow/?ref=ml_lbp)
[What is Huggingface Trainer?\\
\\
\\
In the landscape of machine learning and natural language processing (NLP), Hugging Face has emerged as a key player with its tools and libraries that facilitate the development and deployment of state-of-the-art models. One of the most significant tools in its ecosystem is the Hugging Face Trainer.\\
\\
8 min read](https://www.geeksforgeeks.org/what-is-huggingface-trainer/?ref=ml_lbp)
[What is Keras?\\
\\
\\
Keras is a high-level deep learning API that simplifies the process of building deep neural networks. Initially developed as an independent library, Keras is now tightly integrated into TensorFlow as its official high-level API. It supports multiple backend engines, including TensorFlow, Theano, and\\
\\
5 min read](https://www.geeksforgeeks.org/what-is-keras/?ref=ml_lbp)
[What is Simulated Annealing\\
\\
\\
In the world of optimization, finding the best solution to complex problems can be challenging, especially when the solution space is vast and filled with local optima. One powerful method for overcoming this challenge is Simulated Annealing (SA). Inspired by the physical process of annealing in met\\
\\
5 min read](https://www.geeksforgeeks.org/what-is-simulated-annealing/?ref=ml_lbp)
[What is PyTorch Ignite?\\
\\
\\
PyTorch Ignite is a high-level library designed to simplify the process of training and evaluating neural networks using PyTorch. It provides a flexible and transparent framework that allows developers to focus on building models rather than dealing with the complexities of the training process. Thi\\
\\
7 min read](https://www.geeksforgeeks.org/what-is-pytorch-ignite/?ref=ml_lbp)
[CatBoost Bayesian optimization\\
\\
\\
Bayesian optimization is a powerful and efficient technique for hyperparameter tuning of machine learning models and CatBoost is a very popular gradient boosting library which is known for its robust performance in various tasks. When we combine both, Bayesian optimization for CatBoost can offer an\\
\\
10 min read](https://www.geeksforgeeks.org/catboost-bayesian-optimization/?ref=ml_lbp)
[Intuition behind Adagrad Optimizer\\
\\
\\
Adagrad is short for "Adaptive Gradient Algorithm". It is an adaptive learning rate optimization algorithm used for training deep learning models. It is particularly effective for sparse data or scenarios where features exhibit a large variation in magnitude. Adagrad adjusts the learning rate for ea\\
\\
7 min read](https://www.geeksforgeeks.org/intuition-behind-adagrad-optimizer/?ref=ml_lbp)

Like31

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/adam-optimizer/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=796407918.1745057060&gtm=45je54h0h2v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509156~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025~103130495~103130497&z=177080360)

Login Modal \| GeeksforGeeks

# Log in

New user ?Register Now

Continue with Google

or

Username or Email

Password

Remember me

Forgot Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=xf7wq42vjpy1)

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

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LexF0sUAAAAADiQjz9BMiSrqplrItl-tWYDSfWa&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=normal&cb=7el64ldorspw)

Sign Up

\*Please enter your email address or userHandle.

Back to Login

Reset Password

[iframe](https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LdMFNUZAAAAAIuRtzg0piOT-qXCbDF-iQiUi9KY&co=aHR0cHM6Ly93d3cuZ2Vla3Nmb3JnZWVrcy5vcmc6NDQz&hl=en&v=hbAq-YhJxOnlU-7cpgBoAJHb&size=invisible&cb=ouu01jhqs65t)

[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)