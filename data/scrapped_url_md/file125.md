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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/ml-back-propagation-through-time/?type%3Darticle%26id%3D402628&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
numpy.ma.masked\_values() function \| Python\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/numpy-ma-masked_values-function-python/)

# Back Propagation through time – RNN

Last Updated : 04 May, 2020

Comments

Improve

Suggest changes

21 Likes

Like

Report

**Introduction:**
Recurrent Neural Networks are those networks that deal with sequential data. They predict outputs using not only the current inputs but also by taking into consideration those that occurred before it. In other words, the current output depends on current output as well as a memory element (which takes into account the past inputs).
For training such networks, we use good old backpropagation but with a slight twist. We don’t independently train the system at a specific time _“t”_. We train it at a specific time _“t”_ as well as all that has happened before time _“t”_ like t-1, t-2, t-3.

Consider the following representation of a RNN:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200330110806/Rnn-full.png)

RNN Architecture

_S1, S2, S3_ are the hidden states or memory units at time _t1, t2, t3_ respectively, and _Ws_ is the weight matrix associated with it.
_X1, X2, X3_ are the inputs at time _t1, t2, t3_ respectively, and _Wx_ is the weight matrix associated with it.
_Y1, Y2, Y_ 3 are the outputs at time _t1, t2, t3_ respectively, and _Wy_ is the weight matrix associated with it.
For any time, t, we have the following two equations:

![ \begin{equation*} S_{t} = g_{1}(W_{x}x_{t} + W_{s}S_{t-1})                     \end{equation*} \begin{equation*}                     Y_{t} = g_{2}(W_{Y}S_{t})                         \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-654260fd22f089b77b25a377b659ec26_l3.svg)

where g1 and g2 are activation functions.
Let us now perform back propagation at time t = 3.
Let the error function be:

![ \begin{equation*} E_{t} = (d_{t} - Y_{t})^{2} \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-0d2a855cfdf9222b3729b9519da52b61_l3.svg)

, so at t =3,

![ \begin{equation*}  E_{3} = (d_{3} - Y_{3})^{2}                         \end{equation*}    ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-e1284d61cbdcfe498594f4c84e5422fd_l3.svg)

\*We are using the squared error here, where _d3_ is the desired output at time _t = 3_.
To perform back propagation, we have to adjust the weights associated with inputs, the memory units and the outputs.
**Adjusting Wy**
For better understanding, let us consider the following representation:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200330110810/wy.png)

Adjusting Wy

**Formula:**

![ \begin{equation*} \frac{\partial E_{3}}{\partial W_{y}} = \frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial W_{Y}} \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-58bcfd2399ee29a5446fe3d73451e6a8_l3.svg)

**Explanation:** _E3_ is a function of _Y3_. Hence, we differentiate _E3_ w.r.t _Y3_.
_Y3_ is a function of _WY_. Hence, we differentiate _Y3_ w.r.t _WY_.

**Adjusting Ws**
For better understanding, let us consider the following representation:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200330110807/ws.png)

Adjusting Ws

**Formula:**

![ \begin{equation*}      \frac{\partial E_{3}}{\partial W_{S}} = (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial W_{S}})     +   \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8db54bd7f392857c7d700d1ca346640f_l3.svg)

![ \begin{equation*}     (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial S_{2}} . \frac{\partial S_{2}}{\partial W_{S}})      +  \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-ebc6fe044536855d60640c4e28719e63_l3.svg)

![ \begin{equation*}      (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial S_{2}} . \frac{\partial S_{2}}{\partial S_{1}} . \frac{\partial S_{1}}{\partial W_{S}})   \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9fe00d21fb536673bf7de9f381143bc2_l3.svg)

**Explanation:** _E3_ is a function of _Y3_. Hence, we differentiate _E3_ w.r.t _Y3_.
_Y3_ is a function of _S3_. Hence, we differentiate _Y3_ w.r.t _S3_.
_S3_ is a function of _WS_. Hence, we differentiate _S3_ w.r.t _WS_.
But we can’t stop with this; we also have to take into consideration, the previous time steps. So, we differentiate (partially) the Error function with respect to memory units _S2_ as well as _S1_ taking into consideration the weight matrix _WS_.
We have to keep in mind that a memory unit, say St is a function of its previous memory unit St-1.
Hence, we differentiate _S3_ with _S2_ and _S2_ with _S1_.
Generally, we can express this formula as:

![ \begin{equation*}  \frac{\partial E_{N}}{\partial W_{S}} = \sum_{i=1}^{N} \frac{\partial E_{N}}{\partial Y_{N}} . \frac{\partial Y_{N}}{\partial S_{i}} . \frac{\partial S_{i}}{\partial W_{S}}  \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-fa1b09edaa69ac1442ab591bf9ca71ea_l3.svg)

**Adjusting WX:**
For better understanding, let us consider the following representation:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200330110809/wx.png)

Adjusting Wx

**Formula:**

![ \begin{equation*}      \frac{\partial E_{3}}{\partial W_{X}} = (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial W_{X}})     +   \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b7ce6982359e8937b4f1e7260be2a0f8_l3.svg)

![ \begin{equation*}     (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial S_{2}} . \frac{\partial S_{2}}{\partial W_{X}})      +   \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-d9ab8d16f42b5f09e992d50094fcb35e_l3.svg)

![ \begin{equation*}      (\frac{\partial E_{3}}{\partial Y_{3}} . \frac{\partial Y_{3}}{\partial S_{3}} . \frac{\partial S_{3}}{\partial S_{2}} . \frac{\partial S_{2}}{\partial S_{1}} . \frac{\partial S_{1}}{\partial W_{X}})   \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b94cb321da3f405759d69ef925d48b8d_l3.svg)

**Explanation:** _E3_ is a function of _Y3_. Hence, we differentiate _E3_ w.r.t _Y3_.
_Y3_ is a function of _S3_. Hence, we differentiate _Y3_ w.r.t _S3_.
_S3_ is a function of _WX_. Hence, we differentiate _S3_ w.r.t _WX_.
Again we can’t stop with this; we also have to take into consideration, the previous time steps. So, we differentiate (partially) the Error function with respect to memory units _S2_ as well as _S1_ taking into consideration the weight matrix WX.
Generally, we can express this formula as:

![ \begin{equation*}  \frac{\partial E_{N}}{\partial W_{S}} = \sum_{i=1}^{N} \frac{\partial E_{N}}{\partial Y_{N}} . \frac{\partial Y_{N}}{\partial S_{i}} . \frac{\partial S_{i}}{\partial W_{X}}  \end{equation*} ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-5b5367e5f9ba99bab9578915c6fdb4ff_l3.svg)

**Limitations:**
This method of Back Propagation through time (BPTT) can be used up to a limited number of time steps like 8 or 10. If we back propagate further, the gradient ![\delta](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9c653b918396658c274a1acb53e654f6_l3.svg) becomes too small. This problem is called the “Vanishing gradient” problem. The problem is that the contribution of information decays geometrically over time. So, if the number of time steps is >10 (Let’s say), that information will effectively be discarded.

**Going Beyond RNNs:**
One of the famous solutions to this problem is by using what is called Long Short-Term Memory (LSTM for short) cells instead of the traditional RNN cells. But there might arise yet another problem here, called the **exploding gradient** problem, where the gradient grows uncontrollably large.
**Solution:** A popular method called gradient clipping can be used where in each time step, we can check if the gradient ![\delta](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-9c653b918396658c274a1acb53e654f6_l3.svg) \> threshold. If yes, then normalize it.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/numpy-ma-masked_values-function-python/)

[numpy.ma.masked\_values() function \| Python](https://www.geeksforgeeks.org/numpy-ma-masked_values-function-python/)

[K](https://www.geeksforgeeks.org/user/KeshavBalachandar/)

[KeshavBalachandar](https://www.geeksforgeeks.org/user/KeshavBalachandar/)

Follow

21

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[LSTM - Derivation of Back propagation through time\\
\\
\\
Long Short-Term Memory (LSTM) are a type of neural network designed to handle long-term dependencies by handling the vanishing gradient problem. One of the fundamental techniques used to train LSTMs is Backpropagation Through Time (BPTT) where we have sequential data. In this article we summarize ho\\
\\
4 min read](https://www.geeksforgeeks.org/lstm-derivation-of-back-propagation-through-time/)
[Back Propagation with TensorFlow\\
\\
\\
Backpropagation is a key method used to train neural networks by improving model accuracy. This article explains backpropagation, its working process and implementation in TensorFlow. Understanding BackpropagationBackpropagation is an algorithm that helps neural networks learn by reducing the error\\
\\
5 min read](https://www.geeksforgeeks.org/back-propagation-with-tensorflow/)
[Backpropagation in Neural Network\\
\\
\\
Backpropagation is also known as "Backward Propagation of Errors" and it is a method used to train neural network . Its goal is to reduce the difference between the modelâ€™s predicted output and the actual output by adjusting the weights and biases in the network. In this article we will explore what\\
\\
10 min read](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)
[Q-learning Mathematical Background\\
\\
\\
Prerequisites: Q-Learning. In the following derivations, the symbols defined as in the prerequisite article will be used. The Q-learning technique is based on the Bellman Equation. \[Tex\]v(s) = E(R\_{t+1}+\\lambda v(S\_{t+1})\|S\_{t}=s)\[/Tex\] where, E : Expectation t+1 : next state \[Tex\]\\lambda\[/Tex\] : di\\
\\
2 min read](https://www.geeksforgeeks.org/q-learning-mathematical-background/)
[Computational Graph in PyTorch\\
\\
\\
PyTorch is a popular open-source machine learning library for developing deep learning models. It provides a wide range of functions for building complex neural networks. PyTorch defines a computational graph as a Directed Acyclic Graph (DAG) where nodes represent operations (e.g., addition, multipl\\
\\
4 min read](https://www.geeksforgeeks.org/computational-graph-in-pytorch/)
[RNN for Text Classifications in NLP\\
\\
\\
In this article, we will learn how we can use recurrent neural networks (RNNs) for text classification tasks in natural language processing (NLP). We would be performing sentiment analysis, one of the text classification techniques on the IMDB movie review dataset. We would implement the network fro\\
\\
12 min read](https://www.geeksforgeeks.org/rnn-for-text-classifications-in-nlp/)
[How to Utilize Hebbian Learning\\
\\
\\
Hebbian learning is a fundamental theory in neuroscience that describes how neurons adapt during the learning process. The core principle is often summarized as "neurons that fire together, wire together." This concept can be applied to artificial neural networks to train them like how biological br\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-utilize-hebbian-learning/)
[Deep Neural net with forward and back propagation from scratch - Python\\
\\
\\
This article aims to implement a deep neural network from scratch. We will implement a deep neural network containing two input layers, a hidden layer with four units and one output layer. The implementation will go from scratch and the following steps will be implemented. Algorithm:1. Loading and v\\
\\
6 min read](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)
[Neural Logic Reinforcement Learning - An Introduction\\
\\
\\
Neural Logic Reinforcement Learning is an algorithm that combines logic programming with deep reinforcement learning methods. Logic programming can be used to express knowledge in a way that does not depend on the implementation, making programs more flexible, compressed and understandable. It enabl\\
\\
3 min read](https://www.geeksforgeeks.org/neural-logic-reinforcement-learning-an-introduction/)
[NLP \| Backoff Tagging to combine taggers\\
\\
\\
What is Part-of-speech (POS) tagging ? It is a process of converting a sentence to forms â€“ list of words, list of tuples (where each tuple is having a form (word, tag)). The tag in case of is a part-of-speech tag, and signifies whether the word is a noun, adjective, verb, and so on. What is Backoff\\
\\
3 min read](https://www.geeksforgeeks.org/nlp-backoff-tagging-to-combine-taggers/)

Like21

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/ml-back-propagation-through-time/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=2145750470.1745057209&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=485805607)

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