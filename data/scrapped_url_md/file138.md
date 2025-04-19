- [Python](https://www.geeksforgeeks.org/python-programming-language/)
- [R Language](https://www.geeksforgeeks.org/r-tutorial/)
- [Python for Data Science](https://www.geeksforgeeks.org/data-science-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [Data Analysis](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [ML Math](https://www.geeksforgeeks.org/machine-learning-mathematics/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [NLP](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)
- [Deep Learning](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning-interview-questions/)
- [Machine Learning](https://www.geeksforgeeks.org/machine-learning/)
- [ML Projects](https://www.geeksforgeeks.org/machine-learning-project-with-source-code/)
- [ML Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/super-resolution-gan-srgan/?type%3Darticle%26id%3D437385&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Difference between Backward and Forward chaining\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/difference-between-backward-and-forward-chaining-2/)

# Super Resolution GAN (SRGAN)

Last Updated : 04 Jul, 2022

Comments

Improve

Suggest changes

Like Article

Like

Report

SRGAN was proposed by researchers at Twitter. The motive of this architecture is to recover finer textures from the image when we upscale it so that it’s quality cannot be compromised. There are other methods such as Bilinear Interpolation that can be used to perform this task but they suffer from image information loss and smoothing. In this paper, the authors proposed two architectures the one without GAN (SRResNet) and one with GAN (SRGAN). It is concluded that SRGAN has better accuracy and generate image more pleasing to eyes as compared to SRGAN.

**Architecture:** Similar to GAN architectures, the Super Resolution GAN also contains two parts Generator and Discriminator where generator produces some data based on the probability distribution and discriminator tries to guess weather data coming from input dataset or generator.  Generator than tries to optimize the generated data so that it can fool the discriminator. Below are the generator and discriminator architectural details:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200619230513/SRGAN-660x442.jpg)

SR-GAN architecture

**Generator Architecture:**

The generator architecture contains residual network instead of deep convolution networks because residual networks are easy to train and allows them to be substantially deeper in order to generate better results. This is because the residual network used a type of connections called skip connections.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200612003056/generator-660x191.PNG)

There are B residual blocks (16), originated by ResNet. Within the residual block, two convolutional layers are used, with small 3×3 kernels and 64 feature maps followed by batch-normalization layers and ParametricReLU as the activation function.

The resolution of the input image is increased with two trained sub-pixel convolution layers.

This generator architecture also uses parametric ReLU as an activation function which instead of using a fixed value for a parameter of the rectifier (alpha) like LeakyReLU. It adaptively learns the parameters of rectifier and   improves the accuracy at negligible extra computational cost

During the training, A high-resolution image (HR) is downsampled to a low-resolution image (LR). The generator architecture than tries to upsample the image from low resolution to super-resolution. After then the image is passed into the discriminator, the discriminator and tries to distinguish between a super-resolution and High-Resolution image and generate the adversarial loss which then backpropagated into the generator architecture.

**Discriminator Architecture:**

The task of the discriminator is to discriminate between real HR images and generated SR images.   The discriminator architecture used in this paper is similar to DC- GAN architecture with LeakyReLU as activation. The network contains eight convolutional layers with of 3×3 filter kernels, increasing by a factor of 2 from 64 to 512 kernels. Strided convolutions are used to reduce the image resolution each time the number of features is doubled. The resulting 512 feature maps are followed by two dense layers and a leakyReLU applied between and a final sigmoid activation function to obtain a probability for sample classification.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200618222236/discriminator.PNG)

**Loss Function:**

The SRGAN uses perpetual loss function (LSR)  which is the weighted sum of two loss components : content loss and adversarial loss. This loss is very important for the performance of the generator architecture:

- **Content Loss:** We use two types of content loss in this paper : pixelwise MSE loss for the SRResnet architecture, which is most common MSE loss for image Super Resolution. However MSE loss does not able to deal with high frequency content in the image that resulted in producing overly smooth images. Therefore the authors of the paper decided to  use loss of different VGG layers. This VGG loss is based on the ReLU activation layers of the pre-trained 19 layer VGG network. This loss is defined as follows:

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611204717/simplecontentloss.PNG)

Simple Content Loss

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611204826/vggcontentloss.PNG)

VGG content loss

- **Adversarial Loss**: The Adversarial loss is the loss function that forces the generator to image more similar to high resolution image by using a discriminator that is trained to differentiate between high resolution and super resolution images.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611204925/adversarialloss.PNG)

- Therefore total content loss of this architecture will be :

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611205623/totalpereptualloss-300x77.PNG)

**Results:**

The authors performed experiments on three widely used benchmarks datasets known as Set 5, Set 14, and BSD 100. These experiments performed on 4x up sampling of both rows and columns.

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611214612/ssnetandsrgan.PNG)

In the above layer MSE means we take simple  mean squared pixelwise error as content loss, VGG22 indicate the feature map obtained by the 2nd convolution (after activation) before the 2nd maxpooling layer within the VGG19 network and we calculate the VGG loss using formula described above. This loss is  thus loss on the low-level features. Similarly VGG 54 uses loss calculated on  the feature map obtained by the 4th convolution (after activation) before the 5th maxpooling layer within the VGG19 network. This represents loss on  higher level features from deeper network layers with more potential to focus on the content of the images

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611232501/mosscore-660x300.PNG)

The above image shows MOS scores on BSD100 dat

![](https://media.geeksforgeeks.org/wp-content/uploads/20200611234520/comparison-660x259.PNG)

aset. For each method 2600 samples (100 images ×26 raters) were assessed. Mean shown as red marker, where the bins are centered around value i.

The main contributions of this paper is:

- This paper generates state-of-the-art results on upsampling (4x) as measured by PNSR (Peak Signal-to-Noise Ratio) and SSIM(Structural Similarity) with 16 block deep SRResNet network optimize for MSE.
- The authors propose a new  Super Resolution GAN in which the authors replace the MSE based content loss with the  loss calculated on VGG layer
- SRGAN was able to generate state-of-the-art results which the author validated with extensive Mean Opinion Score (MOS) test on three public benchmark datasets.

**References**:

- [SRGAN paper](https://arxiv.org/pdf/1609.04802.pdf)

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/difference-between-backward-and-forward-chaining-2/)

[Difference between Backward and Forward chaining](https://www.geeksforgeeks.org/difference-between-backward-and-forward-chaining-2/)

[P](https://www.geeksforgeeks.org/user/pawangfg/)

[pawangfg](https://www.geeksforgeeks.org/user/pawangfg/)

Follow

Improve

Article Tags :

- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [data-science](https://www.geeksforgeeks.org/tag/data-science/)
- [Image-Processing](https://www.geeksforgeeks.org/tag/image-processing/)

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)

### Similar Reads

[Deep Transfer Learning - Introduction\\
\\
\\
Deep transfer learning is a machine learning technique that utilizes the knowledge learned from one task to improve the performance of another related task. This technique is particularly useful when there is a shortage of labeled data for the target task, as it allows the model to leverage the know\\
\\
8 min read](https://www.geeksforgeeks.org/deep-transfer-learning-introduction/)
[Deep Convolutional GAN with Keras\\
\\
\\
Deep Convolutional GAN (DCGAN) was proposed by a researcher from MIT and Facebook AI research. It is widely used in many convolution-based generation-based techniques. The focus of this paper was to make training GANs stable. Hence, they proposed some architectural changes in the computer vision pro\\
\\
9 min read](https://www.geeksforgeeks.org/deep-convolutional-gan-with-keras/)
[DLSS - Deep Learning Super Sampling\\
\\
\\
Super Sampling is also known as Super Sampling Anti Aliasing(SSAA) is a spatial anti-aliasing method i.e. a method to remove aliasing (jagged and pixelated edges also known as "jaggies") from a video, rendered images or another software that produces computer graphics. Aliasing is not often dealt-wi\\
\\
4 min read](https://www.geeksforgeeks.org/dlss-deep-learning-super-sampling/)
[Python OpenCV - Super resolution with deep learning\\
\\
\\
Super-resolution (SR) implies the conversion of an image from a lower resolution (LR) to images with a higher resolution (HR). It makes wide use of augmentation. It forms the basis of most computer vision and image processing models. However, with the advancements in deep learning technologies, deep\\
\\
9 min read](https://www.geeksforgeeks.org/python-opencv-super-resolution-with-deep-learning/)
[Simple Genetic Algorithm (SGA)\\
\\
\\
Prerequisite - Genetic Algorithm Introduction : Simple Genetic Algorithm (SGA) is one of the three types of strategies followed in Genetic algorithm. SGA starts with the creation of an initial population of size N.Then, we evaluate the goodness/fitness of each of the solutions/individuals. After tha\\
\\
1 min read](https://www.geeksforgeeks.org/simple-genetic-algorithm-sga/)
[GPT-3 : Next AI Revolution\\
\\
\\
In recent years AI revolution is happening around the world but in recent months if you're a tech enthusiast you've heard about GPT-3. Generative Pre-trained Transformer 3 (GPT-3) is a language model that uses the Transformer technique to do various tasks. It is the third-generation language predict\\
\\
4 min read](https://www.geeksforgeeks.org/gpt-3-next-ai-revolution/)
[Introduction to ANN \| Set 4 (Network Architectures)\\
\\
\\
Prerequisites: Introduction to ANN \| Set-1, Set-2, Set-3 An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the brain. ANNs, like people, learn by examples. An ANN is configured for a specific application, such as pattern recognition or data classification,\\
\\
5 min read](https://www.geeksforgeeks.org/introduction-to-ann-set-4-network-architectures/)
[Self - attention in NLP\\
\\
\\
Self-attention was proposed by researchers at Google Research and Google Brain. It was proposed due to challenges faced by the encoder-decoder in dealing with long sequences. The authors also provide two variants of attention and transformer architecture. This transformer architecture generates stat\\
\\
7 min read](https://www.geeksforgeeks.org/self-attention-in-nlp/)
[Self -attention in NLP\\
\\
\\
Self-attention was proposed by researchers at Google Research and Google Brain. It was proposed due to challenges faced by encoder-decoder in dealing with long sequences. The authors also provide two variants of attention and transformer architecture. This transformer architecture generates the stat\\
\\
5 min read](https://www.geeksforgeeks.org/self-attention-in-nlp-2/)
[Computational Graph in PyTorch\\
\\
\\
PyTorch is a popular open-source machine learning library for developing deep learning models. It provides a wide range of functions for building complex neural networks. PyTorch defines a computational graph as a Directed Acyclic Graph (DAG) where nodes represent operations (e.g., addition, multipl\\
\\
4 min read](https://www.geeksforgeeks.org/computational-graph-in-pytorch/)

Like

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/super-resolution-gan-srgan/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=676917791.1745057279&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=101509157~102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&z=1904714222)