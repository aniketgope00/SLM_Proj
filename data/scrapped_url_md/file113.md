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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/digital-image-processing-basics/?type%3Darticle%26id%3D174514&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
What is a Pixel?\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/what-is-a-pixel/)

# Digital Image Processing Basics

Last Updated : 22 Feb, 2023

Comments

Improve

Suggest changes

85 Likes

Like

Report

Digital Image Processing means processing digital image by means of a digital computer. We can also say that it is a use of computer algorithms, in order to get enhanced image either to extract some useful information.

Digital image processing is the use of algorithms and mathematical models to process and analyze digital images. The goal of digital image processing is to enhance the quality of images, extract meaningful information from images, and automate image-based tasks.

### The basic steps involved in digital image processing are:

1. Image acquisition: This involves capturing an image using a digital camera or scanner, or importing an existing image into a computer.
2. Image enhancement: This involves improving the visual quality of an image, such as increasing contrast, reducing noise, and removing artifacts.
3. Image restoration: This involves removing degradation from an image, such as blurring, noise, and distortion.
4. Image segmentation: This involves dividing an image into regions or segments, each of which corresponds to a specific object or feature in the image.
5. Image representation and description: This involves representing an image in a way that can be analyzed and manipulated by a computer, and describing the features of an image in a compact and meaningful way.
6. Image analysis: This involves using algorithms and mathematical models to extract information from an image, such as recognizing objects, detecting patterns, and quantifying features.
7. Image synthesis and compression: This involves generating new images or compressing existing images to reduce storage and transmission requirements.
8. Digital image processing is widely used in a variety of applications, including medical imaging, remote sensing, computer vision, and multimedia.

## Image processing mainly include the following steps:

1.Importing the image via image acquisition tools;

2.Analysing and manipulating the image;

3.Output in which result can be altered image or a report which is based on analysing that image.

## What is an image?

An image is defined as a two-dimensional function, **F(x,y)**, where x and y are spatial coordinates, and the amplitude of **F** at any pair of coordinates (x,y) is called the **intensity** of that image at that point. When x,y, and amplitude values of **F** are finite, we call it a **digital image**.

In other words, an image can be defined by a two-dimensional array specifically arranged in rows and columns.

Digital Image is composed of a finite number of elements, each of which elements have a particular value at a particular location.These elements are referred to as _picture elements,image elements,and pixels_.A _Pixel_ is most widely used to denote the elements of a Digital Image.

## Types of an image

1. **BINARY IMAGE**– The binary image as its name suggests, contain only two pixel elements i.e 0 & 1,where 0 refers to black and 1 refers to white. This image is also known as Monochrome.
2. **BLACK AND WHITE IMAGE**– The image which consist of only black and white color is called BLACK AND WHITE IMAGE.
3. **8 bit COLOR FORMAT**– It is the most famous image format.It has 256 different shades of colors in it and commonly known as Grayscale Image. In this format, 0 stands for Black, and 255 stands for white, and 127 stands for gray.
4. **16 bit COLOR FORMAT**– It is a color image format. It has 65,536 different colors in it.It is also known as High Color Format. In this format the distribution of color is not as same as Grayscale image.

A 16 bit format is actually divided into three further formats which are Red, Green and Blue. That famous RGB format.

## Image as a Matrix

As we know, images are represented in rows and columns we have the following syntax in which images are represented:

![](https://media.geeksforgeeks.org/wp-content/uploads/gfg2-5.png)

The right side of this equation is digital image by definition. Every element of this matrix is called image element , picture element , or pixel.

## DIGITAL IMAGE REPRESENTATION IN MATLAB:

![](https://media.geeksforgeeks.org/wp-content/uploads/gfg3-6.png)

In MATLAB the start index is from 1 instead of 0. Therefore, f(1,1) = f(0,0).

henceforth the two representation of image are identical, except for the shift in origin.

In MATLAB, matrices are stored in a variable i.e X,x,input\_image , and so on. The variables must be a letter as same as other programming languages.

## PHASES OF IMAGE PROCESSING:

1. **ACQUISITION**– It could be as simple as being given an image which is in digital form. The main work involves:

a) Scaling

b) Color conversion(RGB to Gray or vice-versa)

2. **IMAGE ENHANCEMENT**– It is amongst the simplest and most appealing in areas of Image Processing it is also used to extract some hidden details from an image and is subjective.

3. **IMAGE RESTORATION**– It also deals with appealing of an image but it is objective(Restoration is based on mathematical or probabilistic model or image degradation).

4. **COLOR IMAGE PROCESSING**– It deals with pseudocolor and full color image processing color models are applicable to digital image processing.

5. **WAVELETS AND MULTI-RESOLUTION PROCESSING**– It is foundation of representing images in various degrees.

6. **IMAGE COMPRESSION**-It involves in developing some functions to perform this operation. It mainly deals with image size or resolution.

7. **MORPHOLOGICAL PROCESSING**-It deals with tools for extracting image components that are useful in the representation & description of shape.

8. **SEGMENTATION PROCEDURE**-It includes partitioning an image into its constituent parts or objects. Autonomous segmentation is the most difficult task in Image Processing.

9. **REPRESENTATION & DESCRIPTION**-It follows output of segmentation stage, choosing a representation is only the part of solution for transforming raw data into processed data.

10. **OBJECT DETECTION AND RECOGNITION**-It is a process that assigns a label to an object based on its descriptor.

## OVERLAPPING FIELDS WITH IMAGE PROCESSING

![](https://media.geeksforgeeks.org/wp-content/cdn-uploads/digital-image-processing.png)

**According to block 1**,if input is an image and we get out image as a output, then it is termed as Digital Image Processing.

**According to block 2**,if input is an image and we get some kind of information or description as a output, then it is termed as Computer Vision.

**According to block 3**,if input is some description or code and we get image as an output, then it is termed as Computer Graphics.

**According to block 4**,if input is description or some keywords or some code and we get description or some keywords as a output,then it is termed as Artificial Intelligence

### Advantages of Digital Image Processing:

1. Improved image quality: Digital image processing algorithms can improve the visual quality of images, making them clearer, sharper, and more informative.
2. Automated image-based tasks: Digital image processing can automate many image-based tasks, such as object recognition, pattern detection, and measurement.
3. Increased efficiency: Digital image processing algorithms can process images much faster than humans, making it possible to analyze large amounts of data in a short amount of time.
4. Increased accuracy: Digital image processing algorithms can provide more accurate results than humans, especially for tasks that require precise measurements or quantitative analysis.

### Disadvantages of Digital Image Processing:

1. High computational cost: Some digital image processing algorithms are computationally intensive and require significant computational resources.
2. Limited interpretability: Some digital image processing algorithms may produce results that are difficult for humans to interpret, especially for complex or sophisticated algorithms.
3. Dependence on quality of input: The quality of the output of digital image processing algorithms is highly dependent on the quality of the input images. Poor quality input images can result in poor quality output.
4. Limitations of algorithms: Digital image processing algorithms have limitations, such as the difficulty of recognizing objects in cluttered or poorly lit scenes, or the inability to recognize objects with significant deformations or occlusions.
5. Dependence on good training data: The performance of many digital image processing algorithms is dependent on the quality of the training data used to develop the algorithms. Poor quality training data can result in poor performance of the algorit


## REFERENCES

Digital Image Processing (Rafael c. gonzalez)

**Reference books:**

“Digital Image Processing” by Rafael C. Gonzalez and Richard E. Woods.

“Computer Vision: Algorithms and Applications” by Richard Szeliski.

“Digital Image Processing Using MATLAB” by Rafael C. Gonzalez, Richard E. Woods, and Steven L. Eddins.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/what-is-a-pixel/)

[What is a Pixel?](https://www.geeksforgeeks.org/what-is-a-pixel/)

[N](https://www.geeksforgeeks.org/user/Nishant_Kumar/)

[Nishant\_Kumar](https://www.geeksforgeeks.org/user/Nishant_Kumar/)

Follow

85

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Image-Processing](https://www.geeksforgeeks.org/tag/image-processing/)

### Similar Reads

[Digital Image Processing Tutorial\\
\\
\\
In this tutorial, we will learn all about Digital Image Processing or DIP which is a subcategory of signal processing that particularly deals with the manipulation of digital images by using a digital computer. It is based on the principle of the I-P-O cycle, where it will take a digital image as an\\
\\
13 min read](https://www.geeksforgeeks.org/digital-image-processing-tutorial/)

## Introduction to Digital Image Processing

- [Digital Image Processing Basics\\
\\
\\
Digital Image Processing means processing digital image by means of a digital computer. We can also say that it is a use of computer algorithms, in order to get enhanced image either to extract some useful information. Digital image processing is the use of algorithms and mathematical models to proc\\
\\
7 min read](https://www.geeksforgeeks.org/digital-image-processing-basics/)

* * *

- [What is a Pixel?\\
\\
\\
A pixel is the smallest unit of a digital image or display and stands for â€œpicture element.â€ It is a very small, isolated dot that stands for one color and plays the most basic part in digital images. Pixels when combined help to create the mosaic of colors and shapes contributing towards visual con\\
\\
10 min read](https://www.geeksforgeeks.org/what-is-a-pixel/)

* * *


## Image Conversion

- [MATLAB \| RGB image representation\\
\\
\\
RGB image can be viewed as three different images(a red scale image, a green scale image and a blue scale image) stacked on top of each other, and when fed into the red, green and blue inputs of a color monitor, it produces a color image on the screen. RGB color model is the model in which Red, Blue\\
\\
4 min read](https://www.geeksforgeeks.org/matlab-rgb-image-representation/)

* * *

- [How to Convert RGB Image to Binary Image Using MATLAB?\\
\\
\\
An Image, by definition, is essentially a visual representation of something that depicts or records visual perception. Images are classified in one of the three types. Binary ImagesGrayscale ImagesColor ImagesBinary Images: This is the most basic type of image that exists. The only permissible pixe\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-convert-rgb-image-to-binary-image-using-matlab/)

* * *

- [YIQ Color Model in Computer Graphics\\
\\
\\
During the early days of color television, black-and-white sets were still expected to display what were originally color images. YIQ model separated chrominance from luminance. Luminance information is contained on the Y-channel, whereas color information is carried on I and Q channels (in-phase an\\
\\
2 min read](https://www.geeksforgeeks.org/yiq-color-model-in-computer-graphics/)

* * *

- [How to Convert YIQ Image to RGB Image Using MATLAB?\\
\\
\\
Converting images from one color space to another is a handy process in image processing that provides convenience and eases the overall process by choosing the most accurate color space to work the image on. The YIQ color space, as the name suggests comprises of three components namely Luma (Y), In\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-convert-yiq-image-to-rgb-image-using-matlab/)

* * *

- [How to Convert RGB Image to YIQ Image using MATLAB?\\
\\
\\
Image Processing in MATLAB use functions from the Image Processing Toolbox. This toolbox generally represents colors as RGB numeric values. Different models also exist for representing colors numerically. The official term for these models is "color spaces" and is coined from the definition of a vec\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-convert-rgb-image-to-yiq-image-using-matlab/)

* * *

- [MATLAB \| RGB image to grayscale image conversion\\
\\
\\
An RGB image can be viewed as three images( a red scale image, a green scale image and a blue scale image) stacked on top of each other. In MATLAB, an RGB image is basically a M\*N\*3 array of colour pixel, where each colour pixel is a triplet which corresponds to red, blue and green colour component\\
\\
4 min read](https://www.geeksforgeeks.org/matlab-rgb-image-to-grayscale-image-conversion/)

* * *

- [MATLAB \| Change the color of background pixels by OTSU Thresholding\\
\\
\\
MATLAB also called Matrix Laboratory is a numerical computing environment and a platform for programming language. it was designed and developed by MathWorks. MATLAB is a framework that allows you to perform matrix manipulations, implementing algorithms, plotting functions and data, creating user-in\\
\\
2 min read](https://www.geeksforgeeks.org/matlab-change-the-color-of-background-pixels-by-otsu-thresholding/)

* * *

- [How to Converting RGB Image to HSI Image in MATLAB?\\
\\
\\
Converting the color space of an image is one of the most commonly performed operations in Image Processing. It is used so much as a lot of transformation/filters are performed on a specific color mode of an image. i.e. Thresholding is performed in Grayscale, Color slicing in HSV etc. A color space\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-converting-rgb-image-to-hsi-image-in-matlab/)

* * *

- [How to Convert HSI Image to RGB Image in MATLAB?\\
\\
\\
HSI stands for Hue Saturation Intensity. HSI is color space, it provides a way of numeric to readout the image that is corresponding to the color name contained. RGB it's basically called the RGB triplet. in MATLAB the RGB image is a kind of MxNx3 M cross N cross 3 arrays of colors pixel which is us\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-convert-hsi-image-to-rgb-image-in-matlab/)

* * *

- [How to Partially Colored Gray Image in MATLAB?\\
\\
\\
Partially colored images are a common way of enhancing the objects within the image. It is sometimes used as a tool to emphasize the presence of certain objects within the scene. And the processing required to create one is negligible, in contrast to the effect it produces. In this article you will\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-partially-colored-gray-image-in-matlab/)

* * *

- [HSV Color Model in Computer Graphics\\
\\
\\
A color model is a multidimensional representation of the color spectrum. The most relevant color spectrums are RGB, HSV, HSL and CMYK. A color model can be represented as a 3D surface (e.g. for RGB) or go into much higher dimensions (such as CMYK). By adjusting the parameters of these surfaces, we\\
\\
2 min read](https://www.geeksforgeeks.org/hsv-color-model-in-computer-graphics/)

* * *

- [How to Color Slicing Using HSV Color Space in MATLAB?\\
\\
\\
Color slicing is a technique in image processing, which lets us separate certain objects from their surroundings. Furthermore, it works by filtering through only a certain band of the color spectrum, essentially erasing all other colors beyond it. In this article, you will learn how to perform color\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-color-slicing-using-hsv-color-space-in-matlab/)

* * *


## Image Filtering Techniques

- [Spatial Filtering and its Types\\
\\
\\
Spatial Filtering technique is used directly on pixels of an image. Mask is usually considered to be added in size so that it has specific center pixel. This mask is moved on the image such that the center of the mask traverses all image pixels. Classification on the basis of Linearity There are two\\
\\
3 min read](https://www.geeksforgeeks.org/spatial-filtering-and-its-types/)

* * *

- [Frequency Domain Filters and its Types\\
\\
\\
Frequency Domain Filters are used for smoothing and sharpening of image by removal of high or low frequency components. Sometimes it is possible of removal of very high and very low frequency. Frequency domain filters are different from spatial domain filters as it basically focuses on the frequency\\
\\
2 min read](https://www.geeksforgeeks.org/frequency-domain-filters-and-its-types/)

* * *

- [How to Remove Salt and Pepper Noise from Image Using MATLAB?\\
\\
\\
Impulse noise is a unique form of noise that can have many different origins. Images are frequently corrupted through impulse noise due to transmission errors, defective memory places, or timing mistakes in analog-to-digital conversion. Salt-and-pepper noise is one form of impulse noise that can cor\\
\\
4 min read](https://www.geeksforgeeks.org/how-to-remove-salt-and-pepper-noise-from-image-using-matlab/)

* * *

- [How to Decide Window Size for a Moving Average Filter in MATLAB?\\
\\
\\
A moving average filter is a widely used technique for smoothing data in signal processing. It is used to reduce the amount of noise in a given signal and to identify trends in the data. In MATLAB, the window size of a moving average filter is an important parameter that determines how much data is\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-decide-window-size-for-a-moving-average-filter-in-matlab/)

* * *

- [Noise Models in Digital Image Processing\\
\\
\\
The principal source of noise in digital images arises during image acquisition and transmission. The performance of imaging sensors is affected by a variety of environmental and mechanical factors of the instrument, resulting in the addition of undesirable noise in the image. Images are also corrup\\
\\
3 min read](https://www.geeksforgeeks.org/noise-models-in-digital-image-processing/)

* * *

- [How to Apply Median Filter For RGB Image in MATLAB?\\
\\
\\
Filtering by definition is the process of enhancing or modifying an image by applying some method or algorithm over a small area and extending that algorithm over all the areas of the image. Since we apply the methods over an area, filtering is classified as a neighboring operation. Filters are esse\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-apply-median-filter-for-rgb-image-in-matlab/)

* * *

- [How to Linear Filtering Without Using Imfilter Function in MATLAB?\\
\\
\\
Edges can be sharpened, random noise can be reduced, and uneven illuminations can be corrected using a linear filtering technique. Correlating the image with the proper filter kernel completes the process. The imfilter function calculates the value of each output pixel using double-precision floatin\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-linear-filtering-without-using-imfilter-function-in-matlab/)

* * *

- [Noise addition using in-built Matlab function\\
\\
\\
Noise in an image: Digital images are prone to various types of noise that makes the quality of the images worst. Image noise is random variation of brightness or color information in the captured image. Noise is basically the degradation in image signal caused by external sources such as camera. Im\\
\\
3 min read](https://www.geeksforgeeks.org/noise-addition-using-in-built-matlab-function/)

* * *

- [Adaptive Filtering - Local Noise Filter in MATLAB\\
\\
\\
On the degraded image, which contains both the original image and noise, an adaptive filter is applied. With a predetermined mxn window region, the mean and variance are the two statistical measures on which a locally adaptive filter depends. Adaptive Filters:adaptive filters are also Digital filter\\
\\
4 min read](https://www.geeksforgeeks.org/adaptive-filtering-local-noise-filter-in-matlab/)

* * *

- [Difference between Low pass filter and High pass filter\\
\\
\\
IntrWhen it comes to processing signals, filtering is a key aspect that helps in shaping the characteristics of the signal. Low-pass and high-pass filters are two commonly used types of filters that work in opposite ways to filter signals. Low-pass filters, as the name suggests, allow low-frequency\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-low-pass-filter-and-high-pass-filter/)

* * *

- [MATLAB - Butterworth Lowpass Filter in Image Processing\\
\\
\\
In the field of Image Processing, Butterworth Lowpass Filter (BLPF) is used for image smoothing in the frequency domain. It removes high-frequency noise from a digital image and preserves low-frequency components. The transfer function of BLPF of order \[Tex\]n\[/Tex\] is defined as- \[Tex\]H(u, v)=\\frac{\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-butterworth-lowpass-filter-in-image-processing/)

* * *

- [MATLAB - Ideal Lowpass Filter in Image Processing\\
\\
\\
In the field of Image Processing, Ideal Lowpass Filter (ILPF) is used for image smoothing in the frequency domain. It removes high-frequency noise from a digital image and preserves low-frequency components. It can be specified by the function- \[Tex\] $H(u, v)=\\left\\{\\begin{array}{ll}1 & D(u, v)\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-ideal-lowpass-filter-in-image-processing/)

* * *

- [MATLAB \| Converting a Grayscale Image to Binary Image using Thresholding\\
\\
\\
Thresholding is the simplest method of image segmentation and the most common way to convert a grayscale image to a binary image.In thresholding, we select a threshold value and then all the gray level value which is below the selected threshold value is classified as 0(black i.e background ) and al\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-converting-a-grayscale-image-to-binary-image-using-thresholding/)

* * *

- [Laplacian of Gaussian Filter in MATLAB\\
\\
\\
The Laplacian filter is used to detect the edges in the images. But it has a disadvantage over the noisy images. It amplifies the noise in the image. Hence, first, we use a Gaussian filter on the noisy image to smoothen it and then subsequently use the Laplacian filter for edge detection. Dealing wi\\
\\
4 min read](https://www.geeksforgeeks.org/laplacian-of-gaussian-filter-in-matlab/)

* * *

- [What is Upsampling in MATLAB?\\
\\
\\
In this article, we will see Upsampling in MATLAB. As we know that upsampling is the process of increasing the sampling rate, i.e, increasing the number of samples. When an upsampling functions on a series of samples of a signal or other continued function, it has an estimation of the row that would\\
\\
4 min read](https://www.geeksforgeeks.org/what-is-upsampling-in-matlab/)

* * *

- [Upsampling in Frequency Domain in MATLAB\\
\\
\\
Upsampling" is the process of inserting zero-valued samples between original samples to increase the sampling rate. (This is called "zero-stuffing".) Upsampling adds to the original signal undesired spectral images which are centered on multiples of the original sampling rate. In Frequency domain ,\\
\\
4 min read](https://www.geeksforgeeks.org/upsampling-in-frequency-domain-in-matlab/)

* * *

- [Convolution Shape (full/same/valid) in MATLAB\\
\\
\\
Convolution is a mathematical operation. It is used in Image processing in MatLab. A mask/filter is used to convolve an image for image detection purposes. But MatLab offers three types of convolution. Here we shall explain the simple convolution. The filter slides over the image matrix from left to\\
\\
6 min read](https://www.geeksforgeeks.org/convolution-shape-full-same-valid-in-matlab/)

* * *

- [Linear Convolution using C and MATLAB\\
\\
\\
A key concept often introduced to those pursuing electronics engineering is Linear Convolution. This is a crucial component of Digital Signal Processing and Signals and Systems. Keeping general interest and academic implications in mind, this article introduces the concept and its applications and i\\
\\
8 min read](https://www.geeksforgeeks.org/linear-convolution-using-c-and-matlab/)

* * *


## Histogram Equalization

- [Histogram Equalization in Digital Image Processing\\
\\
\\
A digital image is a two-dimensional matrix of two spatial coordinates, with each cell specifying the intensity level of the image at that point. So, we have an N x N matrix with integer values ranging from a minimum intensity level of 0 to a maximum level of L-1, where L denotes the number of inten\\
\\
6 min read](https://www.geeksforgeeks.org/histogram-equalization-in-digital-image-processing/)

* * *

- [Histogram Equalization Without Using histeq() Function in MATLAB\\
\\
\\
Histogram Equalization is the most famous contrast management technique for digital image processing with different image data intensity level values. or we can say it's a Pixels brightness transformations technique. The histogram is basically a graph-based representation method that clarifies the n\\
\\
4 min read](https://www.geeksforgeeks.org/histogram-equalization-without-using-histeq-function-in-matlab/)

* * *

- [MATLAB \| Display histogram of a grayscale Image\\
\\
\\
An image histogram is chart representation of the distribution of intensities in an Indexed image or grayscale image. It shows how many times each intensity value in image occurs. Code #1: Display histogram of an image using MATLAB library function. % Read an Image in MATLAB Environment img=imread('\\
\\
2 min read](https://www.geeksforgeeks.org/matlab-display-histogram-of-a-grayscale-image/)

* * *

- [What Color Histogram Equalization in MATLAB?\\
\\
\\
A Histogram is a graph-based representation technique between a number of pixels and intensity values. it is a plot of the frequency of occurrence of an event. So in this article, we will understand how we generate the and Equalize Histogram of a color image. Histogram EqualizationHistogram Equaliza\\
\\
5 min read](https://www.geeksforgeeks.org/what-color-histogram-equalization-in-matlab/)

* * *

- [Histogram of an Image\\
\\
\\
The histogram of a digital image with gray levels in the range \[0, L-1\] is a discrete function.Â Histogram Function:Â Â  Points about Histogram:Â Â  Histogram of an image provides a global description of the appearance of an image.Information obtained from histogram is very large in quality.Histogram of\\
\\
2 min read](https://www.geeksforgeeks.org/histogram-of-an-image/)

* * *


## Object Identification and Edge Detection

- [Functions in MATLAB\\
\\
\\
Methods are also popularly known as functions. The main aim of the methods is to reuse the code. A method is a block of code which is invoked and executed when it is called by the user. It contains local workspace and independent of base workspace which belongs to command prompt. Let's take a glance\\
\\
5 min read](https://www.geeksforgeeks.org/functions-in-matlab/)

* * *

- [Program to determine the quadrant of the cartesian plane\\
\\
\\
Given co-ordinates (x, y), determine the quadrant of the cartesian plane. Image\_source : wikipedia.org Examples : Input : x = 1, y = 1 Output : lies in 1st quadrant Input : x = 0, y = 0 Output : lies at origin There are 9 conditions that needs to be checked to determine where does the points lies -\\
\\
6 min read](https://www.geeksforgeeks.org/program-determine-quadrant-cartesian-plane/)

* * *

- [How To Identifying Objects Based On Label in MATLAB?\\
\\
\\
labeling means identifying and placing labels on each program of an object project in the image. in Binary image, it's classified as 4-connected and 8-connected. for performing the labeling in MATLAB we use the Built-in function bwlabel() used to label the object in the binary image. So, In this art\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-identifying-objects-based-on-label-in-matlab/)

* * *

- [What is Image shading in MATLAB?\\
\\
\\
As the name suggests, Image shading means modifying the original image into its sharpened form by detecting the edges of the image. Here two words are combined to define a process one is "image" which means picture and another word is "Shading" which is defined as a process of redrawing or modifying\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-image-shading-in-matlab/)

* * *

- [Edge detection using in-built function in MATLAB\\
\\
\\
Edge detection: In an image, an edge is a curve that follows a path of rapid change in intensity of that image. Edges are often associated with the boundaries of the object in a scene environment. Edge detection is used to identify the edges in an image to make image processing easy. Edge detection\\
\\
2 min read](https://www.geeksforgeeks.org/edge-detection-using-in-built-function-in-matlab/)

* * *

- [Digital Image Processing Algorithms using MATLAB\\
\\
\\
Like it is said, "One picture is worth more than ten thousand words "A digital image is composed of thousands and thousands of pixels. An image could also be defined as a two-dimensional function, f(x, y), where x and y are spatial (plane) coordinates and therefore the amplitude of f at any pair of\\
\\
8 min read](https://www.geeksforgeeks.org/digital-image-processing-algorithms-using-matlab/)

* * *

- [MATLAB - Image Edge Detection using Sobel Operator from Scratch\\
\\
\\
Sobel Operator: It is a discrete differentiation gradient-based operator. It computes the gradient approximation of image intensity function for image edge detection. At the pixels of an image, the Sobel operator produces either the normal to a vector or the corresponding gradient vector. It uses tw\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-image-edge-detection-using-sobel-operator-from-scratch/)

* * *

- [Image Complement in Matlab\\
\\
\\
Prerequisite: RGB image representationMATLAB stores most images as two-dimensional matrices, in which each element of the matrix corresponds to a single discrete pixel in the displayed image. Some images, such as truecolor images, represent images using a three-dimensional array. In truecolor images\\
\\
2 min read](https://www.geeksforgeeks.org/image-complement-in-matlab/)

* * *

- [Image Sharpening Using Laplacian Filter and High Boost Filtering in MATLAB\\
\\
\\
Image sharpening is an effect applied to digital images to give them a sharper appearance. Sharpening enhances the definition of edges in an image. The dull images are those which are poor at the edges. There is not much difference in background and edges. On the contrary, the sharpened image is tha\\
\\
4 min read](https://www.geeksforgeeks.org/image-sharpening-using-laplacian-filter-and-high-boost-filtering-in-matlab/)

* * *

- [Edge detection using in-built function in MATLAB\\
\\
\\
Edge detection: In an image, an edge is a curve that follows a path of rapid change in intensity of that image. Edges are often associated with the boundaries of the object in a scene environment. Edge detection is used to identify the edges in an image to make image processing easy. Edge detection\\
\\
2 min read](https://www.geeksforgeeks.org/edge-detection-using-in-built-function-in-matlab/)

* * *


## PhotoShop Effects in MATLAB

- [What is Swirl Effect in MATLAB?\\
\\
\\
In Matlab, the Swirl Effect is a type of Photoshop effect. Image processing has made extensive use of the swirl effect. It is advantageous for those with expertise in image processing because both the image and the fundamental component of the swirl effect are matrices. The swirl effect simplifies i\\
\\
2 min read](https://www.geeksforgeeks.org/what-is-swirl-effect-in-matlab/)

* * *

- [What is Oil Painting in MATLAB?\\
\\
\\
In this era of information and intelligence, it seems that painting graphics have been seriously marginalized, which also produced how to better develop painting technology in modern society. Nowadays, in this field, digital image processing plays important role in the expansion of oil painting crea\\
\\
3 min read](https://www.geeksforgeeks.org/what-is-oil-painting-in-matlab/)

* * *

- [Cone Effect in MATLAB\\
\\
\\
MATLAB is a high-performance language that is used for matrix manipulation, performing technical computations, graph plottings, etc. It stands for Matrix Laboratory. With the help of this software, we can also give the cone effect to an image. It is done by using the image's midpoint. After this, we\\
\\
2 min read](https://www.geeksforgeeks.org/cone-effect-in-matlab/)

* * *

- [What is Glassy Effect in MATLAB?\\
\\
\\
MATLAB is a high-performance language that is used for matrix manipulation, performing technical computations, graph plottings, etc. It stands for Matrix Laboratory. With the help of this software, we can also give the glassy effect to an image. It is done by replacing each pixel value in the image\\
\\
2 min read](https://www.geeksforgeeks.org/what-is-glassy-effect-in-matlab/)

* * *

- [What is Tiling Effect in MATLAB?\\
\\
\\
MATLAB is a high-performance language that is used for matrix manipulation, performing technical computations, graph plottings, etc. It stands for Matrix Laboratory. With the help of this software, we can provide a tiling effect to an image. It is done by breaking the original image and then allocat\\
\\
2 min read](https://www.geeksforgeeks.org/what-is-tiling-effect-in-matlab/)

* * *


## Image Geometry, Optical Illusion and Image Transformation

- [Matlab program to rotate an image 180 degrees clockwise without using function\\
\\
\\
An image is defined as two-dimensional function, f(x, y), where x and y are spatial (plane) coordinates, and the amplitude of f at any pair of coordinates(x, y) is called the Intensity or the Gray level of the image at that point. When x, y and the intensity values of f are all finite, discrete quan\\
\\
2 min read](https://www.geeksforgeeks.org/matlab-program-to-rotate-an-image-180-degrees-clockwise-without-using-function/)

* * *

- [Image Resizing in Matlab\\
\\
\\
Prerequisite : RGB image representation MATLAB stores most images as two-dimensional matrices, in which each element of the matrix corresponds to a single discrete pixel in the displayed image. Some images, such as truecolor images, represent images using a three-dimensional array. In truecolor imag\\
\\
2 min read](https://www.geeksforgeeks.org/image-resizing-in-matlab/)

* * *

- [Matlab program to rotate an image 180 degrees clockwise without using function\\
\\
\\
An image is defined as two-dimensional function, f(x, y), where x and y are spatial (plane) coordinates, and the amplitude of f at any pair of coordinates(x, y) is called the Intensity or the Gray level of the image at that point. When x, y and the intensity values of f are all finite, discrete quan\\
\\
2 min read](https://www.geeksforgeeks.org/matlab-program-to-rotate-an-image-180-degrees-clockwise-without-using-function/)

* * *

- [Nearest-Neighbor Interpolation Algorithm in MATLAB\\
\\
\\
Nearest neighbor interpolation is a type of interpolation. This method simply determines the "nearest" neighboring pixel and assumes its intensity value, as opposed to calculating an average value using some weighting criteria or producing an intermediate value based on intricate rules. Interpolatio\\
\\
3 min read](https://www.geeksforgeeks.org/nearest-neighbor-interpolation-algorithm-in-matlab/)

* * *

- [Black and White Optical illusion in MATLAB\\
\\
\\
MATLAB provides many toolboxes for different applications. The Image Processing Toolbox is one of the important toolboxes in MATLAB. The Black and White optical illusion in MATLAB happens when we compliment the pixel values in a black and white image and after staring at the center of the resultant\\
\\
2 min read](https://www.geeksforgeeks.org/black-and-white-optical-illusion-in-matlab/)

* * *

- [MATLAB \| Complement colors in a Binary image\\
\\
\\
Binary image is a digital image that has only two possible value for each pixel - either 1 or 0, where 0 represents white and 1 represents black. In the complement of a binary image, the image pixel having value zeros become ones and the image pixel having value ones become zeros; i.e white and blac\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-complement-colors-in-a-binary-image/)

* * *

- [Discrete Cosine Transform (Algorithm and Program)\\
\\
\\
Image Compression : Image is stored or transmitted with having pixel value. It can be compressed by reducing the value its every pixel contains. Image compression is basically of two types : Lossless compression : In this type of compression, after recovering image is exactly become same as that was\\
\\
11 min read](https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/)

* * *

- [2-D Inverse Cosine Transform in MATLAB\\
\\
\\
The 2-D inverse cosine transform is used to decode an image into the spatial domain, which is a more suitable data representation for compression (ICT). ICT-based decoding is the foundation for standards for image and video decompression. or, to put it another way, we can say that the inverse cosine\\
\\
2 min read](https://www.geeksforgeeks.org/2-d-inverse-cosine-transform-in-matlab/)

* * *

- [MATLAB - Intensity Transformation Operations on Images\\
\\
\\
Intensity transformations are among the simplest of all image processing techniques. Approaches whose results depend only on the intensity at a point are called point processing techniques or Intensity transformation techniques. Although intensity transformation and spatial filtering methods span a\\
\\
4 min read](https://www.geeksforgeeks.org/matlab-intensity-transformation-operations-on-images/)

* * *

- [Fast Fourier Transformation for polynomial multiplication\\
\\
\\
Given two polynomial A(x) and B(x), find the product C(x) = A(x)\*B(x). There is already an O(\[Tex\]n^2 \[/Tex\]) naive approach to solve this problem. here. This approach uses the coefficient form of the polynomial to calculate the product.A coefficient representation of a polynomial \[Tex\]A(x)=\\sum\_{j=\\
\\
13 min read](https://www.geeksforgeeks.org/fast-fourier-transformation-poynomial-multiplication/)

* * *

- [Gray Scale to Pseudo Color Transformation in MATLAB\\
\\
\\
The principle behind the pseudo color transformation is to map the intensity value in the image to the result of three distinct transformationsâ€”RED, BLUE, and GREENâ€”on a grayscale or intensity image. Now we can see an example of this procedure using Matlab. Example 1: C/C++ Code % READ A IMAGE INSTA\\
\\
1 min read](https://www.geeksforgeeks.org/gray-scale-to-pseudo-color-transformation-in-matlab/)

* * *

- [Piece-wise Linear Transformation\\
\\
\\
Piece-wise Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application. Some commonly used piece-wise linear tran\\
\\
2 min read](https://www.geeksforgeeks.org/piece-wise-linear-transformation/)

* * *

- [Balance Contrast Enhancement Technique in MATLAB\\
\\
\\
With this technique, biassed color (Red Green Blue) composition can be fixed. The histogram pattern of the input image is unaffected by changes to the contrast of the image (A). The solution is based on the parabolic function of the input image. Equation: \[Tex\] y = a (x - b)2 + c \[/Tex\] The three in\\
\\
1 min read](https://www.geeksforgeeks.org/balance-contrast-enhancement-technique-in-matlab/)

* * *


## Morphologiocal Image Processing, Compression and Files

- [Boundary Extraction of image using MATLAB\\
\\
\\
The boundary of the image is different from the edges in the image. Edges represent the abrupt change in pixel intensity values while the boundary of the image is the contour. As the name boundary suggests that something whose ownership changes, in the image when pixel ownership changes from one sur\\
\\
3 min read](https://www.geeksforgeeks.org/boundary-extraction-of-image-using-matlab/)

* * *

- [MATLAB: Connected Component Labeling without Using bwlabel or bwconncomp Functions\\
\\
\\
A connected component or object in a binary image is a set of adjacent pixels. Determining which pixels are adjacent depends on how pixel connectivity is defined. There are two standard classification connections for 2D images. Connected-4 - Pixels are connected when the edges touch. Two adjacent pi\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-connected-component-labeling-without-using-bwlabel-or-bwconncomp-functions/)

* * *

- [Morphological operations in MATLAB\\
\\
\\
Morphological Operations is a broad set of image processing operations that process digital images based on their shapes. In a morphological operation, each image pixel is corresponding to the value of other pixel in its neighborhood. By choosing the shape and size of the neighborhood pixel, you can\\
\\
2 min read](https://www.geeksforgeeks.org/morphological-operations-in-matlab/)

* * *

- [Matlab \| Erosion of an Image\\
\\
\\
Morphology is known as the broad set of image processing operations that process images based on shapes. It is also known as a tool used for extracting image components that are useful in the representation and description of region shape. The basic morphological operations are: Erosion DilationIn t\\
\\
3 min read](https://www.geeksforgeeks.org/matlab-erosion-of-an-image/)

* * *

- [Auto Cropping- Based on Labeling the Connected Components using MATLAB\\
\\
\\
Auto Cropping Based on labeling the connected components is related to digital image processing. So, In this article, we will discuss the Auto Cropping-Based on labeling the connected components. Before that look at the basic terms which are required in this topic. Auto CroppingIn terms of MATLAB, t\\
\\
3 min read](https://www.geeksforgeeks.org/auto-cropping-based-on-labeling-the-connected-components-using-matlab/)

* * *

- [Run Length Encoding & Decoding in MATLAB\\
\\
\\
Run-length encoding, or RLE, is a straightforward method of lossless data compression in which runs of data, or sequences of data with the same value in many consecutive elements, are stored as a single value and count rather than as the original run. Â In other words, RLE ( Run Length coding) is a s\\
\\
2 min read](https://www.geeksforgeeks.org/run-length-encoding-decoding-in-matlab/)

* * *

- [Lossless Predictive Coding in MATLAB\\
\\
\\
In Lossless Predictive Coding A new pixel value is obtained by finding the difference between the predicted pixel value and the current pixel. In other words, this is The new information of a pixel is defined as the difference between the actual and predicted value of that pixel. The approach common\\
\\
3 min read](https://www.geeksforgeeks.org/lossless-predictive-coding-in-matlab/)

* * *

- [Extract bit planes from an Image in Matlab\\
\\
\\
Image is basically combination of individual pixel (dots) information. When we write that image is of 620 X 480 size, it means that image has 620 pixel in horizontal direction and 480 pixel in vertical direction. So, altogether there is 620 X 480 pixels and each pixels contains some information abou\\
\\
3 min read](https://www.geeksforgeeks.org/extract-bit-planes-image-matlab/)

* * *

- [How to Read Text File Backwards Using MATLAB?\\
\\
\\
Prerequisites: Write Data to Text Files in MATLAB Sometimes for some specific use case, it is required for us to read the file backward. i.e. The file should be read from EOF (End of file Marker) to the beginning of the file in reverse order. In this article we would learn how to read a file in back\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-read-text-file-backwards-using-matlab/)

* * *

- [MATLAB - Read Words in a File in Reverse Order\\
\\
\\
MATLAB stands for Matrix Laboratory. It is a high-performance language that is used for technical computing. It allows matrix manipulations, plotting of functions, implementation of algorithms, and creation of user interfaces Suppose we are given a text file containing the following words "I STUDY F\\
\\
2 min read](https://www.geeksforgeeks.org/matlab-read-words-in-a-file-in-reverse-order/)

* * *

- [How to Read Image File or Complex Image File in MATLAB?\\
\\
\\
MATLAB is a programming and numeric computing platform used by millions of engineers and scientists to analyze data, develop algorithms, and create models. For Image Reading in MATLAB, we use the image processing toolbox. In this ToolBox, there are many methods such as imread(), imshow() etc. imshow\\
\\
2 min read](https://www.geeksforgeeks.org/how-to-read-image-file-or-complex-image-file-in-matlab/)

* * *


## Image Coding, Comparison and Texture Features

- [Digital Watermarking and its Types\\
\\
\\
Digital Watermarking is use of a kind of marker covertly embedded in a digital media such as audio, video or image which enables us to know the source or owner of the copyright. This technique is used for tracing copyright infringement in social media and knowing the genuineness of the notes in the\\
\\
3 min read](https://www.geeksforgeeks.org/digital-watermarking-and-its-types/)

* * *

- [How To Hide Message or Image Inside An Image In MATLAB?\\
\\
\\
MATLAB is a high-performance language that is used for matrix manipulation, performing technical computations, graph plottings, etc. It stands for Matrix Laboratory. With the help of this software, we can also hide a message or image inside an image. Following are the steps to hide an image, which c\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-hide-message-or-image-inside-an-image-in-matlab/)

* * *

- [How to Match a Template in MATLAB?\\
\\
\\
Moving the template across the entire image and comparing it to the image's covered window is a process known as "template matching." The implementation of template matching involves two-dimensional convolution. Template matching has different applications and is utilized in such fields as face ackn\\
\\
3 min read](https://www.geeksforgeeks.org/how-to-match-a-template-in-matlab/)

* * *

- [Grey Level Co-occurrence Matrix in MATLAB\\
\\
\\
The use of texture to identify regions of interest in an image is a crucial characteristic. One of Haralick et al.'s earliest approaches to texture feature extraction was Grey Level Co-occurrence Matrices (GLCM) in the year 1973. Since then, it has been used extensively in a number of texture analys\\
\\
4 min read](https://www.geeksforgeeks.org/grey-level-co-occurrence-matrix-in-matlab/)

* * *

- [MATLAB - Texture Measures from GLCM\\
\\
\\
GLCM stands for Gray Level Co-occurrence Matrix. In image processing, The GLCM function computes how often pairs of pixels with a particular value and in a particular spatial relationship occur in an image, constructs a GLCM, and extracts statistical measures from this matrix to determine the textur\\
\\
4 min read](https://www.geeksforgeeks.org/matlab-texture-measures-from-glcm/)

* * *


Like85

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/digital-image-processing-basics/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=1785639266.1745057065&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&z=145217252)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057065428&cv=11&fst=1745057065428&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54g3v877914216za200zb884918195&gcd=13l3l3R3l5l1&dma=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116025&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fdigital-image-processing-basics%2F&hn=www.googleadservices.com&frm=0&tiba=Digital%20Image%20Processing%20Basics%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=1592093534.1745057065&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

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