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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/importance-of-convolutional-neural-network-ml/?type%3Darticle%26id%3D311745&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
NLP \| Distributed Tagging with Execnet - Part 1\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/nlp-distributed-tagging-with-execnet-part-1/)

# Importance of Convolutional Neural Network \| ML

Last Updated : 14 Feb, 2022

Comments

Improve

Suggest changes

Like Article

Like

Report

[**Convolutional Neural Network**](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) as the name suggests is a neural network that makes use of convolution operation to classify and predict.

Let’s analyze the use cases and advantages of a convolutional neural network over a simple deep learning network.

**Weight sharing:**

It makes use of Local Spatial coherence that provides same weights to some of the edges, In this way, this weight sharing minimizes the cost of computing. This is especially useful when GPU is low power or missing.

![](https://media.geeksforgeeks.org/wp-content/uploads/20190528195150/Screenshot-3051.png)

**Memory Saving:**

The reduced number of parameters helps in memory saving. For e.g. in case of MNIST dataset to recognize digits, if we use a CNN with single hidden layer and 10 nodes, it would require few hundred nodes but if we use a simple deep neural network, it would require around 19000 parameters.

**Independent of local variations in Image:**

Let’s consider if we are training our fully connected neural network for face recognition with head-shot images of people, Now if we test it on an image which is not a head-shot image but full body image then it may fail to recognize. Since the convolutional neural network makes use of convolution operation, they are independent of local variations in the image.

**Equivariance:**

Equivariance is the property of CNNs and one that can be seen as a specific type of parameter sharing. Conceptually, a function can be considered equivariance if, upon a change in the input, a similar change is reflected in the output. Mathematically, it can be represented as f(g(x)) = g(f(x)). It turns out that convolutions are equivariant to many data transformation operations which helps us to identify, how a particular change in input will affect the output. This helps us to identify any drastic change in the output and retain the reliability of the model.

**Independent of Transformations:**

CNNs are much more independent to geometrical transformations like Scaling, Rotation etc.

**Example of Translation independence –** CNN identifies object correctly

![](https://media.geeksforgeeks.org/wp-content/uploads/20190528200250/Screenshot-2992-1024x363.png)

**Example of Rotation independence –** CNN identifies object correctly

![](https://media.geeksforgeeks.org/wp-content/uploads/20190528200418/Screenshot-3061-1024x338.png)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/nlp-distributed-tagging-with-execnet-part-1/)

[NLP \| Distributed Tagging with Execnet - Part 1](https://www.geeksforgeeks.org/nlp-distributed-tagging-with-execnet-part-1/)

[S](https://www.geeksforgeeks.org/user/Sourabh_Sinha/)

[Sourabh\_Sinha](https://www.geeksforgeeks.org/user/Sourabh_Sinha/)

Follow

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)
- [Neural Network](https://www.geeksforgeeks.org/tag/neural-network/)

### Similar Reads

[Convolutional Neural Networks (CNNs) in R\\
\\
\\
Convolutional Neural Networks (CNNs) are a specialized type of neural network designed to process and analyze visual data. They are particularly effective for tasks involving image recognition and classification due to their ability to automatically and adaptively learn spatial hierarchies of featur\\
\\
11 min read](https://www.geeksforgeeks.org/convolutional-neural-networks-cnns-in-r/?ref=ml_lbp)
[Introduction to Convolution Neural Network\\
\\
\\
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs are widely us\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-convolution-neural-network/?ref=ml_lbp)
[Convolutional Neural Network (CNN) in Tensorflow\\
\\
\\
Convolutional Neural Networks (CNNs) have revolutionized the field of computer vision by automatically learning spatial hierarchies of features from images. In this article we will explore the basic building blocks of CNNs and show you how to implement a CNN model using TensorFlow. Building Blocks o\\
\\
5 min read](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-tensorflow/?ref=ml_lbp)
[Kernels (Filters) in convolutional neural network\\
\\
\\
Convolutional Neural Networks (CNNs) are a category of neural networks designed specifically for processing structured arrays of data such as images. Essential to the functionality of CNNs are components known as kernels or filters. These are small, square matrices that perform convolution operation\\
\\
8 min read](https://www.geeksforgeeks.org/kernels-filters-in-convolutional-neural-network/?ref=ml_lbp)
[Applying Convolutional Neural Network on mnist dataset\\
\\
\\
CNN is a model known to be a Convolutional Neural Network and in recent times it has gained a lot of popularity because of its usefulness. CNN uses multilayer perceptrons to do computational work. CNN uses relatively little pre-processing compared to other image classification algorithms. This means\\
\\
6 min read](https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/?ref=ml_lbp)
[How do convolutional neural networks (CNNs) work?\\
\\
\\
Convolutional Neural Networks (CNNs) have transformed computer vision by allowing machines to achieve unprecedented accuracy in tasks like image classification, object detection, and segmentation. CNNs, which originated with Yann LeCun's work in the late 1980s, are inspired by the human visual syste\\
\\
7 min read](https://www.geeksforgeeks.org/how-do-convolutional-neural-networks-cnns-work/?ref=ml_lbp)
[Math Behind Convolutional Neural Networks\\
\\
\\
Convolutional Neural Networks (CNNs) are designed to process data that has a known grid-like topology, such as images (which can be seen as 2D grids of pixels). The key components of a CNN include convolutional layers, pooling layers, activation functions, and fully connected layers. Each of these c\\
\\
8 min read](https://www.geeksforgeeks.org/math-behind-convolutional-neural-networks/?ref=ml_lbp)
[ML \| Transfer Learning with Convolutional Neural Networks\\
\\
\\
Transfer learning as a general term refers to reusing the knowledge learned from one task for another. Specifically for convolutional neural networks (CNNs), many image features are common to a variety of datasets (e.g. lines, edges are seen in almost every image). It is for this reason that, especi\\
\\
7 min read](https://www.geeksforgeeks.org/ml-transfer-learning-with-convolutional-neural-networks/?ref=ml_lbp)
[Convolutional Neural Network (CNN) in Machine Learning\\
\\
\\
Convolutional Neural Networks (CNNs) are a specialized class of neural networks designed to process grid-like data, such as images. They are particularly well-suited for image recognition and processing tasks. They are inspired by the visual processing mechanisms in the human brain, CNNs excel at ca\\
\\
8 min read](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/?ref=ml_lbp)
[Emotion Detection Using Convolutional Neural Networks (CNNs)\\
\\
\\
Emotion detection, also known as facial emotion recognition, is a fascinating field within the realm of artificial intelligence and computer vision. It involves the identification and interpretation of human emotions from facial expressions. Accurate emotion detection has numerous practical applicat\\
\\
15+ min read](https://www.geeksforgeeks.org/emotion-detection-using-convolutional-neural-networks-cnns/?ref=ml_lbp)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/importance-of-convolutional-neural-network-ml/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=821874556.1745057070&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=498765887)