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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/cnn-introduction-to-padding/?type%3Darticle%26id%3D324898&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is the difference between 'SAME' and 'VALID' padding in tf.nn.max\_pool of tensorflow?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max_pool-of-tensorflow/)

# CNN \| Introduction to Padding

Last Updated : 13 Dec, 2023

Comments

Improve

Suggest changes

Like Article

Like

Report

During convolution, the size of the output feature map is determined by the size of the input feature map, the size of the kernel, and the stride. if we simply apply the kernel on the input feature map, then the output feature map will be smaller than the input. This can result in the loss of information at the borders of the input feature map. In Order to preserve the border information we use padding.

## What Is Padding

padding is a technique used to preserve the spatial dimensions of the input image after convolution operations on a feature map. Padding involves adding extra pixels around the border of the input feature map before convolution.

**This can be done in two ways:**

- **Valid Padding**: In the valid padding, no padding is added to the input feature map, and the output feature map is smaller than the input feature map. This is useful when we want to reduce the spatial dimensions of the feature maps.
- **Same Padding**: In the same padding, padding is added to the input feature map such that the size of the output feature map is the same as the input feature map. This is useful when we want to preserve the spatial dimensions of the feature maps.

The number of pixels to be added for padding can be calculated based on the size of the kernel and the desired output of the feature map size. The most common padding value is zero-padding, which involves adding zeros to the borders of the input feature map.

Padding can help in reducing the loss of information at the borders of the input feature map and can improve the performance of the model. However, it also increases the computational cost of the convolution operation. Overall, padding is an important technique in CNNs that helps in preserving the spatial dimensions of the feature maps and can improve the performance of the model.

## **Problem With  Convolution Layers Without Padding**

- For a grayscale (n x n) image and (f x f) filter/kernel, the dimensions of the image resulting from a convolution operation is **(n – f + 1) x (n – f + 1)**.

For example, for an (8 x 8) image and (3 x 3) filter, the output resulting after the convolution operation would be of size (6 x 6). Thus, the image shrinks every time a convolution operation is performed. This places an upper limit to the number of times such an operation could be performed before the image reduces to nothing thereby precluding us from building deeper networks.
- Also, the pixels on the corners and the edges are used much less than those in the middle.

For example,

![Padding in convulational neural network ](https://media.geeksforgeeks.org/wp-content/uploads/20230510174423/Screenshot-2019-07-16-at-13520.webp)

Padding in convulational neural network

- Clearly, pixel A is touched in just one convolution operation and pixel B is touched in 3 convolution operations, while pixel C is touched in 9 convolution operations. In general, pixels in the middle are used more often than pixels on corners and edges. Consequently, the information on the borders of images is not preserved as well as the information in the middle.

## **Effect Of Padding On Input Images**

Padding is simply a process of adding layers of zeros to our input images so as to avoid the problems mentioned above through the following changes to the input image.

![padding in convolutional network ](https://media.geeksforgeeks.org/wp-content/uploads/20230512174050/padding-in-cnn-(1).webp)

padding in convolutional network

**Padding prevents the shrinking of the input image.**

> **p** = number of layers of zeros added to the border of the image,
>
> then **(n x n)** image **—>** **(n + 2p) x (n + 2p)** image after padding.
>
> **(n + 2p) x (n + 2p) \* (f x f)   —–>** **outputs (n + 2p – f + 1) x (n + 2p – f + 1)** images

For example, by adding one layer of padding to an (8 x 8) image and using a (3 x 3) filter we would get an (8 x 8) output after performing a convolution operation.

This increases the contribution of the pixels at the border of the original image by bringing them into the middle of the padded image. Thus, information on the borders is preserved as well as the information in the middle of the image.

### Types of Padding

**Valid Padding:** It implies no padding at all. The input image is left in its valid/unaltered shape. So

> (n×n)∗(f×f)⟶(n−f+1)×(n−f+1)(n×n)∗(f×f )⟶(n−f+1)×(n−f+1)
> (n×n)∗(f×f)⟶(n−f+1)×(n−f+1)
>
> where,     nxn is the dimension of input image
>
>                fxf is kernel size
>
>                n-f+1 is output image size
>
>               \\* represents a convolution operation.

**Same Padding:** In this case, we add ‘p’ padding layers such that the output image has the same dimensions as the input image.

So,

> **\[(n + 2p) x (n + 2p) image\] \* \[(f x f) filter\] —> \[(n x n) image\]**

which gives **p = (f – 1) / 2** (because n + 2p – f + 1 = n).

So, if we use a (3 x 3) filter on an input image to get the output with the same dimensions. the 1 layer of zeros must be added to the borders for the same padding. Similarly, if (5 x 5) filter is used 2 layers of zeros must be appended to the border of the image.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max_pool-of-tensorflow/)

[What is the difference between 'SAME' and 'VALID' padding in tf.nn.max\_pool of tensorflow?](https://www.geeksforgeeks.org/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max_pool-of-tensorflow/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Computer Vision Tutorial\\
\\
\\
Computer Vision is a branch of Artificial Intelligence (AI) that enables computers to interpret and extract information from images and videos, similar to human perception. It involves developing algorithms to process visual data and derive meaningful insights. Why Learn Computer Vision?High Demand\\
\\
8 min read](https://www.geeksforgeeks.org/computer-vision/)

## Introduction to Computer Vision

- [Computer Vision - Introduction\\
\\
\\
Ever wondered how are we able to understand the things we see? Like we see someone walking, whether we realize it or not, using the prerequisite knowledge, our brain understands what is happening and stores it as information. Imagine we look at something and go completely blank. Into oblivion. Scary\\
\\
3 min read](https://www.geeksforgeeks.org/computer-vision-introduction/)

* * *

- [A Quick Overview to Computer Vision\\
\\
\\
Computer vision means the extraction of information from images, text, videos, etc. Sometimes computer vision tries to mimic human vision. Itâ€™s a subset of computer-based intelligence or Artificial intelligence which collects information from digital images or videos and analyze them to define the a\\
\\
3 min read](https://www.geeksforgeeks.org/a-quick-overview-to-computer-vision/)

* * *

- [Applications of Computer Vision\\
\\
\\
Have you ever wondered how machines can "see" and understand the world around them, much like humans do? This is the magic of computer visionâ€”a branch of artificial intelligence that enables computers to interpret and analyze digital images, videos, and other visual inputs. From self-driving cars to\\
\\
6 min read](https://www.geeksforgeeks.org/applications-of-computer-vision/)

* * *

- [Fundamentals of Image Formation\\
\\
\\
Image formation is an analog to digital conversion of an image with the help of 2D Sampling and Quantization techniques that is done by the capturing devices like cameras. In general, we see a 2D view of the 3D world. In the same way, the formation of the analog image took place. It is basically a c\\
\\
7 min read](https://www.geeksforgeeks.org/fundamentals-of-image-formation/)

* * *

- [Satellite Image Processing\\
\\
\\
Satellite Image Processing is an important field in research and development and consists of the images of earth and satellites taken by the means of artificial satellites. Firstly, the photographs are taken in digital form and later are processed by the computers to extract the information. Statist\\
\\
2 min read](https://www.geeksforgeeks.org/satellite-image-processing/)

* * *

- [Image Formats\\
\\
\\
Image formats are different types of file types used for saving pictures, graphics, and photos. Choosing the right image format is important because it affects how your images look, load, and perform on websites, social media, or in print. Common formats include JPEG, PNG, GIF, and SVG, each with it\\
\\
5 min read](https://www.geeksforgeeks.org/image-formats/)

* * *


## Image Processing & Transformation

- [Digital Image Processing Basics\\
\\
\\
Digital Image Processing means processing digital image by means of a digital computer. We can also say that it is a use of computer algorithms, in order to get enhanced image either to extract some useful information. Digital image processing is the use of algorithms and mathematical models to proc\\
\\
7 min read](https://www.geeksforgeeks.org/digital-image-processing-basics/)

* * *

- [Difference Between RGB, CMYK, HSV, and YIQ Color Models\\
\\
\\
The colour spaces in image processing aim to facilitate the specifications of colours in some standard way. Different types of colour models are used in multiple fields like in hardware, in multiple applications of creating animation, etc. Letâ€™s see each colour model and its application. RGBCMYKHSV\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-rgb-cmyk-hsv-and-yiq-color-models/)

* * *

- [Image Enhancement Techniques using OpenCV - Python\\
\\
\\
Image enhancement is the process of improving the quality and appearance of an image. It can be used to correct flaws or defects in an image, or to simply make an image more visually appealing. Image enhancement techniques can be applied to a wide range of images, including photographs, scans, and d\\
\\
15+ min read](https://www.geeksforgeeks.org/image-enhancement-techniques-using-opencv-python/)

* * *

- [Image Transformations using OpenCV in Python\\
\\
\\
In this tutorial, we are going to learn Image Transformation using the OpenCV module in Python. What is Image Transformation? Image Transformation involves the transformation of image data in order to retrieve information from the image or preprocess the image for further usage. In this tutorial we\\
\\
5 min read](https://www.geeksforgeeks.org/image-transformations-using-opencv-in-python/)

* * *

- [How to find the Fourier Transform of an image using OpenCV Python?\\
\\
\\
The Fourier Transform is a mathematical tool used to decompose a signal into its frequency components. In the case of image processing, the Fourier Transform can be used to analyze the frequency content of an image, which can be useful for tasks such as image filtering and feature extraction. In thi\\
\\
5 min read](https://www.geeksforgeeks.org/how-to-find-the-fourier-transform-of-an-image-using-opencv-python/)

* * *

- [Python \| Intensity Transformation Operations on Images\\
\\
\\
Intensity transformations are applied on images for contrast manipulation or image thresholding. These are in the spatial domain, i.e. they are performed directly on the pixels of the image at hand, as opposed to being performed on the Fourier transform of the image. The following are commonly used\\
\\
5 min read](https://www.geeksforgeeks.org/python-intensity-transformation-operations-on-images/)

* * *

- [Histogram Equalization in Digital Image Processing\\
\\
\\
A digital image is a two-dimensional matrix of two spatial coordinates, with each cell specifying the intensity level of the image at that point. So, we have an N x N matrix with integer values ranging from a minimum intensity level of 0 to a maximum level of L-1, where L denotes the number of inten\\
\\
6 min read](https://www.geeksforgeeks.org/histogram-equalization-in-digital-image-processing/)

* * *

- [Python - Color Inversion using Pillow\\
\\
\\
Color Inversion (Image Negative) is the method of inverting pixel values of an image. Image inversion does not depend on the color mode of the image, i.e. inversion works on channel level. When inversion is used on a multi color image (RGB, CMYK etc) then each channel is treated separately, and the\\
\\
4 min read](https://www.geeksforgeeks.org/python-color-inversion-using-pillow/)

* * *

- [Image Sharpening Using Laplacian Filter and High Boost Filtering in MATLAB\\
\\
\\
Image sharpening is an effect applied to digital images to give them a sharper appearance. Sharpening enhances the definition of edges in an image. The dull images are those which are poor at the edges. There is not much difference in background and edges. On the contrary, the sharpened image is tha\\
\\
4 min read](https://www.geeksforgeeks.org/image-sharpening-using-laplacian-filter-and-high-boost-filtering-in-matlab/)

* * *

- [Wand sharpen() function - Python\\
\\
\\
The sharpen() function is an inbuilt function in the Python Wand ImageMagick library which is used to sharpen the image. Syntax: sharpen(radius, sigma) Parameters: This function accepts four parameters as mentioned above and defined below: radius: This parameter stores the radius value of the sharpn\\
\\
2 min read](https://www.geeksforgeeks.org/wand-sharpen-function-python/)

* * *

- [Python OpenCV - Smoothing and Blurring\\
\\
\\
In this article, we are going to learn about smoothing and blurring with python-OpenCV. When we are dealing with images at some points the images will be crisper and sharper which we need to smoothen or blur to get a clean image, or sometimes the image will be with a really bad edge which also we ne\\
\\
7 min read](https://www.geeksforgeeks.org/python-opencv-smoothing-and-blurring/)

* * *

- [Python PIL \| GaussianBlur() method\\
\\
\\
PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities. The ImageFilter module contains definitions for a pre-defined set of filters, which can be used with the Image.filter() method. PIL.ImageFilter.GaussianBlur() method create Gaussian blur filter.\\
\\
1 min read](https://www.geeksforgeeks.org/python-pil-gaussianblur-method/)

* * *

- [Apply a Gauss filter to an image with Python\\
\\
\\
A Gaussian Filter is a low pass filter used for reducing noise (high frequency components) and blurring regions of an image. The filter is implemented as an Odd sized Symmetric Kernel (DIP version of a Matrix) which is passed through each pixel of the Region of Interest to get the desired effect. Th\\
\\
4 min read](https://www.geeksforgeeks.org/apply-a-gauss-filter-to-an-image-with-python/)

* * *

- [Spatial Filtering and its Types\\
\\
\\
Spatial Filtering technique is used directly on pixels of an image. Mask is usually considered to be added in size so that it has specific center pixel. This mask is moved on the image such that the center of the mask traverses all image pixels. Classification on the basis of Linearity There are two\\
\\
3 min read](https://www.geeksforgeeks.org/spatial-filtering-and-its-types/)

* * *

- [Python PIL \| MedianFilter() and ModeFilter() method\\
\\
\\
PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities. The ImageFilter module contains definitions for a pre-defined set of filters, which can be used with the Image.filter() method. PIL.ImageFilter.MedianFilter() method creates a median filter. Pick\\
\\
1 min read](https://www.geeksforgeeks.org/python-pil-medianfilter-and-modefilter-method/)

* * *

- [Python \| Bilateral Filtering\\
\\
\\
A bilateral filter is used for smoothening images and reducing noise, while preserving edges. This article explains an approach using the averaging filter, while this article provides one using a median filter. However, these convolutions often result in a loss of important edge information, since t\\
\\
2 min read](https://www.geeksforgeeks.org/python-bilateral-filtering/)

* * *

- [Python OpenCV - Morphological Operations\\
\\
\\
Python OpenCV Morphological operations are one of the Image processing techniques that processes image based on shape. This processing strategy is usually performed on binary images. Morphological operations based on OpenCV are as follows: ErosionDilationOpeningClosingMorphological GradientTop hatBl\\
\\
7 min read](https://www.geeksforgeeks.org/python-opencv-morphological-operations/)

* * *

- [Erosion and Dilation of images using OpenCV in python\\
\\
\\
Morphological operations are a set of operations that process images based on shapes. They apply a structuring element to an input image and generate an output image. The most basic morphological operations are two: Erosion and Dilation Basics of Erosion: Erodes away the boundaries of the foreground\\
\\
2 min read](https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/)

* * *

- [Introduction to Resampling methods\\
\\
\\
While reading about Machine Learning and Data Science we often come across a term called Imbalanced Class Distribution, which generally happens when observations in one of the classes are much higher or lower than in other classes. As Machine Learning algorithms tend to increase accuracy by reducing\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-to-resampling-methods/)

* * *

- [Python \| Image Registration using OpenCV\\
\\
\\
Image registration is a digital image processing technique that helps us align different images of the same scene. For instance, one may click the picture of a book from various angles. Below are a few instances that show the diversity of camera angles.Now, we may want to "align" a particular image\\
\\
3 min read](https://www.geeksforgeeks.org/image-registration-using-opencv-python/)

* * *


## Feature Extraction and Description

- [Feature Extraction Techniques - NLP\\
\\
\\
Introduction : This article focuses on basic feature extraction techniques in NLP to analyse the similarities between pieces of text. Natural Language Processing (NLP) is a branch of computer science and machine learning that deals with training computers to process a large amount of human (natural)\\
\\
11 min read](https://www.geeksforgeeks.org/feature-extraction-techniques-nlp/)

* * *

- [SIFT Interest Point Detector Using Python - OpenCV\\
\\
\\
SIFT (Scale Invariant Feature Transform) Detector is used in the detection of interest points on an input image. It allows the identification of localized features in images which is essential in applications such as: Object Recognition in ImagesPath detection and obstacle avoidance algorithmsGestur\\
\\
4 min read](https://www.geeksforgeeks.org/sift-interest-point-detector-using-python-opencv/)

* * *

- [Feature Matching using Brute Force in OpenCV\\
\\
\\
In this article, we will do feature matching using Brute Force in Python by using OpenCV library. Prerequisites: OpenCV OpenCV is a python library which is used to solve the computer vision problems. OpenCV is an open source Computer Vision library. So computer vision is a way of teaching intelligen\\
\\
13 min read](https://www.geeksforgeeks.org/feature-matching-using-brute-force-in-opencv/)

* * *

- [Feature detection and matching with OpenCV-Python\\
\\
\\
In this article, we are going to see about feature detection in computer vision with OpenCV in Python. Feature detection is the process of checking the important features of the image in this case features of the image can be edges, corners, ridges, and blobs in the images. In OpenCV, there are a nu\\
\\
5 min read](https://www.geeksforgeeks.org/feature-detection-and-matching-with-opencv-python/)

* * *

- [Feature matching using ORB algorithm in Python-OpenCV\\
\\
\\
ORB is a fusion of FAST keypoint detector and BRIEF descriptor with some added features to improve the performance. FAST is Features from Accelerated Segment Test used to detect features from the provided image. It also uses a pyramid to produce multiscale-features. Now it doesnâ€™t compute the orient\\
\\
2 min read](https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/)

* * *

- [Mahotas - Speeded-Up Robust Features\\
\\
\\
In this article we will see how we can get the speeded up robust features of image in mahotas. In computer vision, speeded up robust features (SURF) is a patented local feature detector and descriptor. It can be used for tasks such as object recognition, image registration, classification, or 3D rec\\
\\
2 min read](https://www.geeksforgeeks.org/mahotas-speeded-up-robust-features/)

* * *

- [Create Local Binary Pattern of an image using OpenCV-Python\\
\\
\\
In this article, we will discuss the image and how to find a binary pattern using the pixel value of the image. As we all know, image is also known as a set of pixels. When we store an image in computers or digitally, itâ€™s corresponding pixel values are stored. So, when we read an image to a variabl\\
\\
5 min read](https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/)

* * *


## Deep Learning for Computer Vision

- [Image Classification using CNN\\
\\
\\
The article is about creating an Image classifier for identifying cat-vs-dogs using TFLearn in Python. Machine Learning is now one of the hottest topics around the world. Well, it can even be said of the new electricity in today's world. But to be precise what is Machine Learning, well it's just one\\
\\
7 min read](https://www.geeksforgeeks.org/image-classifier-using-cnn/)

* * *

- [What is Transfer Learning?\\
\\
\\
Transfer learning is a machine learning technique where a model trained on one task is repurposed as the foundation for a second task. This approach is beneficial when the second task is related to the first or when data for the second task is limited. Leveraging learned features from the initial ta\\
\\
11 min read](https://www.geeksforgeeks.org/ml-introduction-to-transfer-learning/)

* * *

- [Top 5 PreTrained Models in Natural Language Processing (NLP)\\
\\
\\
Pretrained models are deep learning models that have been trained on huge amounts of data before fine-tuning for a specific task. The pre-trained models have revolutionized the landscape of natural language processing as they allow the developer to transfer the learned knowledge to specific tasks, e\\
\\
7 min read](https://www.geeksforgeeks.org/top-5-pre-trained-models-in-natural-language-processing-nlp/)

* * *

- [ML \| Introduction to Strided Convolutions\\
\\
\\
Let us begin this article with a basic question - "Why padding and strided convolutions are required?" Assume we have an image with dimensions of n x n. If it is convoluted with an f x f filter, then the dimensions of the image obtained are (n−f+1)x(n−f+1)(n-f+1) x (n-f+1)(n−f+1)x(n−f+1). Example: Consider a 6 x 6 ima\\
\\
2 min read](https://www.geeksforgeeks.org/ml-introduction-to-strided-convolutions/)

* * *

- [Dilated Convolution\\
\\
\\
Prerequisite: Convolutional Neural Networks Dilated Convolution: It is a technique that expands the kernel (input) by inserting holes between its consecutive elements. In simpler terms, it is the same as convolution but it involves pixel skipping, so as to cover a larger area of the input. Dilated c\\
\\
5 min read](https://www.geeksforgeeks.org/dilated-convolution/)

* * *

- [Continuous Kernel Convolution\\
\\
\\
Continuous Kernel convolution was proposed by the researcher of Verije University Amsterdam in collaboration with the University of Amsterdam in a paper titled 'CKConv: Continuous Kernel Convolution For Sequential Data'. The motivation behind that is to propose a model that uses the properties of co\\
\\
6 min read](https://www.geeksforgeeks.org/continous-kernel-convolution/)

* * *

- [CNN \| Introduction to Pooling Layer\\
\\
\\
Pooling layer is used in CNNs to reduce the spatial dimensions (width and height) of the input feature maps while retaining the most important information. It involves sliding a two-dimensional filter over each channel of a feature map and summarizing the features within the region covered by the fi\\
\\
5 min read](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/)

* * *

- [CNN \| Introduction to Padding\\
\\
\\
During convolution, the size of the output feature map is determined by the size of the input feature map, the size of the kernel, and the stride. if we simply apply the kernel on the input feature map, then the output feature map will be smaller than the input. This can result in the loss of inform\\
\\
5 min read](https://www.geeksforgeeks.org/cnn-introduction-to-padding/)

* * *

- [What is the difference between 'SAME' and 'VALID' padding in tf.nn.max\_pool of tensorflow?\\
\\
\\
Padding is a technique used in convolutional neural networks (CNNs) to preserve the spatial dimensions of the input data and prevent the loss of information at the edges of the image. It involves adding additional rows and columns of pixels around the edges of the input data. There are several diffe\\
\\
14 min read](https://www.geeksforgeeks.org/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max_pool-of-tensorflow/)

* * *

- [Convolutional Neural Network (CNN) Architectures\\
\\
\\
Convolutional Neural Network(CNN) is a neural network architecture in Deep Learning, used to recognize the pattern from structured arrays. However, over many years, CNN architectures have evolved. Many variants of the fundamental CNN Architecture This been developed, leading to amazing advances in t\\
\\
11 min read](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-architectures/)

* * *

- [Deep Transfer Learning - Introduction\\
\\
\\
Deep transfer learning is a machine learning technique that utilizes the knowledge learned from one task to improve the performance of another related task. This technique is particularly useful when there is a shortage of labeled data for the target task, as it allows the model to leverage the know\\
\\
8 min read](https://www.geeksforgeeks.org/deep-transfer-learning-introduction/)

* * *

- [Introduction to Residual Networks\\
\\
\\
Recent years have seen tremendous progress in the field of Image Processing and Recognition. Deep Neural Networks are becoming deeper and more complex. It has been proved that adding more layers to a Neural Network can make it more robust for image-related tasks. But it can also cause them to lose a\\
\\
4 min read](https://www.geeksforgeeks.org/introduction-to-residual-networks/)

* * *

- [Residual Networks (ResNet) - Deep Learning\\
\\
\\
After the first CNN-based architecture (AlexNet) that win the ImageNet 2012 competition, Every subsequent winning architecture uses more layers in a deep neural network to reduce the error rate. This works for less number of layers, but when we increase the number of layers, there is a common proble\\
\\
9 min read](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/)

* * *

- [ML \| Inception Network V1\\
\\
\\
Inception net achieved a milestone in CNN classifiers when previous models were just going deeper to improve the performance and accuracy but compromising the computational cost. The Inception network, on the other hand, is heavily engineered. It uses a lot of tricks to push performance, both in ter\\
\\
4 min read](https://www.geeksforgeeks.org/ml-inception-network-v1/)

* * *

- [Understanding GoogLeNet Model - CNN Architecture\\
\\
\\
Google Net (or Inception V1) was proposed by research at Google (with the collaboration of various universities) in 2014 in the research paper titled "Going Deeper with Convolutions". This architecture was the winner at the ILSVRC 2014 image classification challenge. It has provided a significant de\\
\\
4 min read](https://www.geeksforgeeks.org/understanding-googlenet-model-cnn-architecture/)

* * *

- [Image Recognition with Mobilenet\\
\\
\\
Introduction: Image Recognition plays an important role in many fields like medical disease analysis, and many more. In this article, we will mainly focus on how to Recognize the given image, what is being displayed. We are assuming to have a pre-knowledge of Tensorflow, Keras, Python, MachineLearni\\
\\
5 min read](https://www.geeksforgeeks.org/image-recognition-with-mobilenet/)

* * *

- [VGG-16 \| CNN model\\
\\
\\
A Convolutional Neural Network (CNN) architecture is a deep learning model designed for processing structured grid-like data, such as images. It consists of multiple layers, including convolutional, pooling, and fully connected layers. CNNs are highly effective for tasks like image classification, o\\
\\
7 min read](https://www.geeksforgeeks.org/vgg-16-cnn-model/)

* * *

- [Autoencoders in Machine Learning\\
\\
\\
An autoencoder is a type of artificial neural network that learns to represent data in a compressed form and then reconstructs it as closely as possible to the original input. Autoencoders consists of two components: Encoder: This compresses the input into a compact representation and capture the mo\\
\\
9 min read](https://www.geeksforgeeks.org/auto-encoders/)

* * *

- [How Autoencoders works ?\\
\\
\\
Autoencoders is a type of neural network used for unsupervised learning particularly for tasks like dimensionality reduction, anomaly detection and feature extraction. It consists of two main parts: an encoder and a decoder. The goal of an autoencoder is to learn a more efficient representation of t\\
\\
7 min read](https://www.geeksforgeeks.org/how-autoencoders-works/)

* * *

- [Difference Between Encoder and Decoder\\
\\
\\
Combinational Logic is the concept in which two or more input states define one or more output states. The Encoder and Decoder are combinational logic circuits. In which we implement combinational logic with the help of boolean algebra. To encode something is to convert in piece of information into\\
\\
9 min read](https://www.geeksforgeeks.org/difference-between-encoder-and-decoder/)

* * *

- [Implementing an Autoencoder in PyTorch\\
\\
\\
Autoencoders are neural networks that learn to compress and reconstruct data. In this guide weâ€™ll walk you through building a simple autoencoder in PyTorch using the MNIST dataset. This approach is useful for image compression, denoising and feature extraction. Implementation of Autoencoder in PyTor\\
\\
4 min read](https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/)

* * *

- [Generative Adversarial Network (GAN)\\
\\
\\
Generative Adversarial Networks (GANs) were introduced by Ian Goodfellow and his colleagues in 2014. GANs are a class of neural networks that autonomously learn patterns in the input data to generate new examples resembling the original dataset. GAN's architecture consists of two neural networks: Ge\\
\\
12 min read](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)

* * *

- [Deep Convolutional GAN with Keras\\
\\
\\
Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision pro\\
\\
9 min read](https://www.geeksforgeeks.org/deep-convolutional-gan-with-keras/)

* * *

- [StyleGAN - Style Generative Adversarial Networks\\
\\
\\
Generative Adversarial Networks (GANs) are a type of neural network that consist two neural networks: a generator that creates images and a discriminator that evaluates them. The generator tries to produce realistic data while the discriminator tries to differentiate between real and generated data.\\
\\
6 min read](https://www.geeksforgeeks.org/stylegan-style-generative-adversarial-networks/)

* * *


## Object Detection and Recognition

- [Detect an object with OpenCV-Python\\
\\
\\
Object detection refers to identifying and locating objects within images or videos. OpenCV provides a simple way to implement object detection using Haar Cascades a classifier trained to detect objects based on positive and negative images. In this article we will focus on detecting objects using i\\
\\
4 min read](https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/)

* * *

- [Python \| Haar Cascades for Object Detection\\
\\
\\
Object Detection is a computer technology related to computer vision, image processing and deep learning that deals with detecting instances of objects in images and videos. We will do object detection in this article using something known as haar cascades. What are Haar Cascades? Haar Cascade class\\
\\
2 min read](https://www.geeksforgeeks.org/python-haar-cascades-for-object-detection/)

* * *

- [R-CNN - Region-Based Convolutional Neural Networks\\
\\
\\
R-CNN (Region-based Convolutional Neural Network) was introduced by Ross Girshick et al. in 2014. R-CNN revolutionized object detection by combining the strengths of region proposal algorithms and deep learning, leading to remarkable improvements in detection accuracy and efficiency. This article de\\
\\
9 min read](https://www.geeksforgeeks.org/r-cnn-region-based-cnns/)

* * *

- [YOLO v2 - Object Detection\\
\\
\\
In terms of speed, YOLO is one of the best models in object recognition, able to recognize objects and process frames at the rate up to 150 FPS for small networks. However, In terms of accuracy mAP, YOLO was not the state of the art model but has fairly good Mean average Precision (mAP) of 63% when\\
\\
6 min read](https://www.geeksforgeeks.org/yolo-v2-object-detection/)

* * *

- [Face recognition using Artificial Intelligence\\
\\
\\
The current technology amazes people with amazing innovations that not only make life simple but also bearable. Face recognition has over time proven to be the least intrusive and fastest form of biometric verification. The software uses deep learning algorithms to compare a live captured image to t\\
\\
15+ min read](https://www.geeksforgeeks.org/face-recognition-using-artificial-intelligence/)

* * *

- [Deep Face Recognition\\
\\
\\
DeepFace is the facial recognition system used by Facebook for tagging images. It was proposed by researchers at Facebook AI Research (FAIR) at the 2014 IEEE Computer Vision and Pattern Recognition Conference (CVPR). In modern face recognition there are 4 steps: DetectAlignRepresentClassify This app\\
\\
8 min read](https://www.geeksforgeeks.org/deep-face-recognition/)

* * *

- [ML \| Face Recognition Using Eigenfaces (PCA Algorithm)\\
\\
\\
In 1991, Turk and Pentland suggested an approach to face recognition that uses dimensionality reduction and linear algebra concepts to recognize faces. This approach is computationally less expensive and easy to implement and thus used in various applications at that time such as handwritten recogni\\
\\
4 min read](https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/)

* * *

- [Emojify using Face Recognition with Machine Learning\\
\\
\\
In this article, we will learn how to implement a modification app that will show an emoji of expression which resembles the expression on your face. This is a fun project based on computer vision in which we use an image classification model in reality to classify different expressions of a person.\\
\\
7 min read](https://www.geeksforgeeks.org/emojify-using-face-recognition-with-machine-learning/)

* * *

- [Object Detection with Detection Transformer (DETR) by Facebook\\
\\
\\
Facebook has just released its State of the art object detection Model on 27 May 2020. They are calling it DERT stands for Detection Transformer as it uses transformers to detect objects.This is the first time that transformer is used for such a task of Object detection along with a Convolutional Ne\\
\\
7 min read](https://www.geeksforgeeks.org/object-detection-with-detection-transformer-dert-by-facebook/)

* * *


## Image Segmentation

- [Image Segmentation Using TensorFlow\\
\\
\\
Image segmentation refers to the task of annotating a single class to different groups of pixels. While the input is an image, the output is a mask that draws the region of the shape in that image. Image segmentation has wide applications in domains such as medical image analysis, self-driving cars,\\
\\
7 min read](https://www.geeksforgeeks.org/image-segmentation-using-tensorflow/)

* * *

- [Thresholding-Based Image Segmentation\\
\\
\\
Image segmentation is the technique of subdividing an image into constituent sub-regions or distinct objects. The level of detail to which subdivision is carried out depends on the problem being solved. That is, segmentation should stop when the objects or the regions of interest in an application h\\
\\
7 min read](https://www.geeksforgeeks.org/thresholding-based-image-segmentation/)

* * *

- [Region and Edge Based Segmentation\\
\\
\\
Segmentation Segmentation is the separation of one or more regions or objects in an image based on a discontinuity or a similarity criterion. A region in an image can be defined by its border (edge) or its interior, and the two representations are equal. There are prominently three methods of perfor\\
\\
4 min read](https://www.geeksforgeeks.org/region-and-edge-based-segmentaion/)

* * *

- [Image Segmentation with Watershed Algorithm - OpenCV Python\\
\\
\\
Image segmentation is a fundamental computer vision task that involves partitioning an image into meaningful and semantically homogeneous regions. The goal is to simplify the representation of an image or make it more meaningful for further analysis. These segments typically correspond to objects or\\
\\
9 min read](https://www.geeksforgeeks.org/image-segmentation-with-watershed-algorithm-opencv-python/)

* * *

- [Mask R-CNN \| ML\\
\\
\\
The article provides a comprehensive understanding of the evolution from basic Convolutional Neural Networks (CNN) to the sophisticated Mask R-CNN, exploring the iterative improvements in object detection, instance segmentation, and the challenges and advantages associated with each model. What is R\\
\\
9 min read](https://www.geeksforgeeks.org/mask-r-cnn-ml/)

* * *


## 3D Reconstruction

- [Python OpenCV - Depth map from Stereo Images\\
\\
\\
OpenCV is the huge open-source library for the computer vision, machine learning, and image processing and now it plays a major role in real-time operation which is very important in todayâ€™s systems.Note: For more information, refer to Introduction to OpenCV Depth Map : A depth map is a picture wher\\
\\
2 min read](https://www.geeksforgeeks.org/python-opencv-depth-map-from-stereo-images/)

* * *

- [Top 7 Modern-Day Applications of Augmented Reality (AR)\\
\\
\\
Augmented Reality (or AR) in simpler terms means intensifying the reality of real-time objects which we see through our eyes or gadgets like smartphones. You may think how is it trending a lot? The answer is that it can impactfully offer an unforgettable experience either of learning, measuring the\\
\\
9 min read](https://www.geeksforgeeks.org/top-7-modern-day-applications-of-augmented-reality-ar/)

* * *

- [Virtual Reality, Augmented Reality, and Mixed Reality\\
\\
\\
Virtual Reality (VR): The word 'virtual' means something that is conceptual and does not exist physically and the word 'reality' means the state of being real. So the term 'virtual reality' is itself conflicting. It means something that is almost real. We will probably never be on the top of Mount E\\
\\
3 min read](https://www.geeksforgeeks.org/virtual-reality-augmented-reality-and-mixed-reality/)

* * *

- [Camera Calibration with Python - OpenCV\\
\\
\\
Prerequisites: OpenCV A camera is an integral part of several domains like robotics, space exploration, etc camera is playing a major role. It helps to capture each and every moment and helpful for many analyses. In order to use the camera as a visual sensor, we should know the parameters of the cam\\
\\
4 min read](https://www.geeksforgeeks.org/camera-calibration-with-python-opencv/)

* * *

- [Python OpenCV - Pose Estimation\\
\\
\\
What is Pose Estimation? Pose estimation is a computer vision technique that is used to predict the configuration of the body(POSE) from an image. The reason for its importance is the abundance of applications that can benefit from technology.Â  Human pose estimation localizes body key points to accu\\
\\
6 min read](https://www.geeksforgeeks.org/python-opencv-pose-estimation/)

* * *


[50+ Top Computer Vision Projects \[2025 Updated\]\\
\\
\\
Computer Vision is a field of Artificial Intelligence (AI) that focuses on interpreting and extracting information from images and videos using various techniques. It is an emerging and evolving field within AI. Computer Vision applications have become an integral part of our daily lives, permeating\\
\\
6 min read](https://www.geeksforgeeks.org/computer-vision-projects/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/cnn-introduction-to-padding/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=528582519.1745057075&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&_ng=1&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=1718258772)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057075443&cv=11&fst=1745057075443&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fcnn-introduction-to-padding%2F&_ng=1&hn=www.googleadservices.com&frm=0&tiba=CNN%20%7C%20Introduction%20to%20Padding%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1412692869.1745057075&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)