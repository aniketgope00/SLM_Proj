- [Deep Learning Tutorial](https://www.geeksforgeeks.org/deep-learning-tutorial/)
- [Data Analysis Tutorial](https://www.geeksforgeeks.org/data-analysis-tutorial/)
- [Python â€“ Data visualization tutorial](https://www.geeksforgeeks.org/python-data-visualization-tutorial/)
- [NumPy](https://www.geeksforgeeks.org/numpy-tutorial/)
- [Pandas](https://www.geeksforgeeks.org/pandas-tutorial/)
- [OpenCV](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- [R](https://www.geeksforgeeks.org/r-tutorial/)
- [Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/)
- [Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning-projects/)_)
- [Machine Learning Interview Questions](https://www.geeksforgeeks.org/machine-learning-interview-questions/)_)
- [Machine Learning Mathematics](https://www.geeksforgeeks.org/machine-learning-mathematics/)
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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/?type%3Darticle%26id%3D1004790&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Named Entity Recognition in NLP\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/named-entity-recognition-in-nlp/)

# Implement Convolutional Autoencoder in PyTorch with CUDA

Last Updated : 31 Jul, 2023

Comments

Improve

Suggest changes

2 Likes

Like

Report

Autoencoders are a type of neural network architecture used for unsupervised learning tasks such as data compression, dimensionality reduction, and data denoising. The architecture consists of two main components: an encoder and a decoder. The encoder portion of the network compresses the input data into a lower-dimensional representation, while the decoder portion of the network reconstructs the original input data from this lower-dimensional representation.

A Convolutional Autoencoder (CAE) is an [autoencoder](https://www.geeksforgeeks.org/ml-auto-encoders/) a type of [deep learning neural network](https://www.geeksforgeeks.org/deep-learning-tutorial/) architecture that is commonly used for [unsupervised learning](https://www.geeksforgeeks.org/supervised-unsupervised-learning/) tasks, such as image compression and denoising. It is an extension of the traditional autoencoder architecture that incorporates convolutional layers into both the encoder and decoder portions of the network.

Same like the Autoencoder, the Convolutional Autoencoder architecture also consists of two main components: an encoder and a decoder. The encoder portion of the network processes the input image using convolutional layers and pooling operations to produce a lower-dimensional feature representation of the image. The decoder portion of the network takes this lower-dimensional feature representation and upsamples it back to the original input image size using deconvolutional layers. The final output of the network is a reconstructed image that is as close as possible to the original input image.

The training process for a Convolutional Autoencoder is similar to that of a traditional autoencoder. The network is trained to minimize the difference between the original input image and the reconstructed output image using a loss function such as mean squared error (MSE) or binary cross-entropy (BCE). Once trained, the encoder portion of the network can be used for feature extraction, and the decoder portion of the network can be used for image generation or reconstruction.

Convolutional Autoencoders have shown impressive results in a variety of computer vision tasks, including image compression, denoising, and feature extraction. They have also been used in various applications such as image retrieval, object recognition, and anomaly detection.

### **Implementation in Pytorch:**

**Algorithm**

01.  Load the dataset using PyTorch’s ImageFolder class and define a dataloader.
02.  Define the Convolutional Autoencoder architecture by creating an Autoencoder class that contains an encoder and decoder, each with convolutional and pooling layers.
03. Initialize the autoencoder model and move it to the GPU if available using the to() method.
04. Define the loss function and optimizer to use during training. Typically, mean squared error (MSE) loss is used, and the Adam optimizer is a popular choice for deep learning tasks.
05. Set the number of epochs to train for and begin the training loop.
06. In each epoch, iterate through the batches of the dataloader, move the data to the GPU, and perform forward propagation to obtain the autoencoder’s output.
07. Calculate the loss between the output and the input using the loss function.
08. Perform backward propagation to calculate the gradients of the model parameters with respect to the loss.\
09.  Use the optimizer to update the model parameters based on the calculated gradients.
10. Print the loss after each epoch to monitor the training progress.
11. Save the trained model to a file using the state\_dict() method.

**Code:**

- Python

## Python

|     |
| --- |
| `import` `torch`<br>`import` `torch.nn as nn`<br>`import` `torch.optim as optim`<br>`import` `torchvision.datasets as datasets`<br>`import` `torchvision.transforms as transforms`<br>`# Define the autoencoder architecture`<br>`class` `Autoencoder(nn.Module):`<br>`    ``def` `__init__(` `self` `):`<br>`        ``super` `(Autoencoder, ` `self` `).__init__()`<br>`        ``self` `.encoder ` `=` `nn.Sequential(`<br>`            ``nn.Conv2d(` `3` `, ` `16` `, kernel_size` `=` `3` `, stride` `=` `1` `, padding` `=` `1` `),`<br>`            ``nn.ReLU(),`<br>`            ``nn.MaxPool2d(kernel_size` `=` `2` `, stride` `=` `2` `),`<br>`            ``nn.Conv2d(` `16` `, ` `8` `, kernel_size` `=` `3` `, stride` `=` `1` `, padding` `=` `1` `),`<br>`            ``nn.ReLU(),`<br>`            ``nn.MaxPool2d(kernel_size` `=` `2` `, stride` `=` `2` `)`<br>`        ``)`<br>`        ``self` `.decoder ` `=` `nn.Sequential(`<br>`            ``nn.ConvTranspose2d(` `8` `, ` `16` `, `<br>`                               ``kernel_size` `=` `3` `, `<br>`                               ``stride` `=` `2` `, `<br>`                               ``padding` `=` `1` `, `<br>`                               ``output_padding` `=` `1` `),`<br>`            ``nn.ReLU(),`<br>`            ``nn.ConvTranspose2d(` `16` `, ` `3` `, `<br>`                               ``kernel_size` `=` `3` `, `<br>`                               ``stride` `=` `2` `, `<br>`                               ``padding` `=` `1` `, `<br>`                               ``output_padding` `=` `1` `),`<br>`            ``nn.Sigmoid()`<br>`        ``)`<br>`        `<br>`    ``def` `forward(` `self` `, x):`<br>`        ``x ` `=` `self` `.encoder(x)`<br>`        ``x ` `=` `self` `.decoder(x)`<br>`        ``return` `x`<br>`# Initialize the autoencoder`<br>`model ` `=` `Autoencoder()`<br>`# Define transform`<br>`transform ` `=` `transforms.Compose([`<br>`    ``transforms.Resize((` `64` `, ` `64` `)),`<br>`    ``transforms.ToTensor(),`<br>`])`<br>`# Load dataset`<br>`train_dataset ` `=` `datasets.Flowers102(root` `=` `'flowers'` `, `<br>`                                    ``split` `=` `'train'` `, `<br>`                                    ``transform` `=` `transform, `<br>`                                    ``download` `=` `True` `)`<br>`test_dataset ` `=` `datasets.Flowers102(root` `=` `'flowers'` `, `<br>`                                   ``split` `=` `'test'` `, `<br>`                                   ``transform` `=` `transform)`<br>`# Define the dataloader`<br>`train_loader ` `=` `torch.utils.data.DataLoader(dataset` `=` `train_dataset, `<br>`                                           ``batch_size` `=` `128` `, `<br>`                                           ``shuffle` `=` `True` `)`<br>`test_loader ` `=` `torch.utils.data.DataLoader(dataset` `=` `test_dataset, `<br>`                                          ``batch_size` `=` `128` `)`<br>`# Move the model to GPU`<br>`device ` `=` `torch.device(` `'cuda'` `if` `torch.cuda.is_available() ` `else` `'cpu'` `)`<br>`print` `(device)`<br>`model.to(device)`<br>`# Define the loss function and optimizer`<br>`criterion ` `=` `nn.MSELoss()`<br>`optimizer ` `=` `optim.Adam(model.parameters(), lr` `=` `0.001` `)`<br>`# Train the autoencoder`<br>`num_epochs ` `=` `50`<br>`for` `epoch ` `in` `range` `(num_epochs):`<br>`    ``for` `data ` `in` `train_loader:`<br>`        ``img, _ ` `=` `data`<br>`        ``img ` `=` `img.to(device)`<br>`        ``optimizer.zero_grad()`<br>`        ``output ` `=` `model(img)`<br>`        ``loss ` `=` `criterion(output, img)`<br>`        ``loss.backward()`<br>`        ``optimizer.step()`<br>`    ``if` `epoch ` `%` `5` `=` `=` `0` `:`<br>`        ``print` `(` `'Epoch [{}/{}], Loss: {:.4f}'` `.` `format` `(epoch` `+` `1` `, num_epochs, loss.item()))`<br>`# Save the model`<br>`torch.save(model.state_dict(), ` `'conv_autoencoder.pth'` `)` |

```

```

```

```

**Output**:

```
cuda
Epoch [1/50], Loss: 0.0919
Epoch [6/50], Loss: 0.0746
Epoch [11/50], Loss: 0.0362
Epoch [16/50], Loss: 0.0239
Epoch [21/50], Loss: 0.0178
Epoch [26/50], Loss: 0.0154
Epoch [31/50], Loss: 0.0144
Epoch [36/50], Loss: 0.0124
Epoch [41/50], Loss: 0.0127
Epoch [46/50], Loss: 0.0101
```

#### Plot the original image with decoded image

- Python3

## Python3

|     |
| --- |
| `with torch.no_grad():`<br>`    ``for` `data, _ ` `in` `test_loader:`<br>`        ``data ` `=` `data.to(device)`<br>`        ``recon ` `=` `model(data)`<br>`        ``break`<br>`        `<br>`import` `matplotlib.pyplot as plt`<br>`plt.figure(dpi` `=` `250` `)`<br>`fig, ax ` `=` `plt.subplots(` `2` `, ` `7` `, figsize` `=` `(` `15` `, ` `4` `))`<br>`for` `i ` `in` `range` `(` `7` `):`<br>`    ``ax[` `0` `, i].imshow(data[i].cpu().numpy().transpose((` `1` `, ` `2` `, ` `0` `)))`<br>`    ``ax[` `1` `, i].imshow(recon[i].cpu().numpy().transpose((` `1` `, ` `2` `, ` `0` `)))`<br>`    ``ax[` `0` `, i].axis(` `'OFF'` `)`<br>`    ``ax[` `1` `, i].axis(` `'OFF'` `)`<br>`plt.show()` |

```

```

```

```

**Output**:

![Convolutional Autoencoder decoded image with original image -Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20230420135733/__results___2_1.png)

Convolutional Autoencoder decoded image with original image

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/named-entity-recognition-in-nlp/)

[Named Entity Recognition in NLP](https://www.geeksforgeeks.org/named-entity-recognition-in-nlp/)

[S](https://www.geeksforgeeks.org/user/sagarseth06/)

[sagarseth06](https://www.geeksforgeeks.org/user/sagarseth06/)

Follow

2

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [Machine Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/machine-learning/)
- [Deep-Learning](https://www.geeksforgeeks.org/tag/deep-learning/)
- [python](https://www.geeksforgeeks.org/tag/python/)
- [Python-PyTorch](https://www.geeksforgeeks.org/tag/python-pytorch/)

+1 More

Practice Tags :

- [Machine Learning](https://www.geeksforgeeks.org/explore?category=Machine%20Learning)
- [python](https://www.geeksforgeeks.org/explore?category=python)

### Similar Reads

[Overcomplete Autoencoders with PyTorch\\
\\
\\
Neural networks are used in autoencoders to encode and decode data. They are utilized in many different applications, including data compression, natural language processing, and picture and audio recognition. Autoencoders work by learning a compressed representation of the input data that may be us\\
\\
7 min read](https://www.geeksforgeeks.org/overcomplete-autoencoders-with-pytorch/?ref=ml_lbp)
[Implementing an Autoencoder in PyTorch\\
\\
\\
Autoencoders are neural networks that learn to compress and reconstruct data. In this guide weâ€™ll walk you through building a simple autoencoder in PyTorch using the MNIST dataset. This approach is useful for image compression, denoising and feature extraction. Implementation of Autoencoder in PyTor\\
\\
4 min read](https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/?ref=ml_lbp)
[Convolutional Variational Autoencoder in Tensorflow\\
\\
\\
In the age of Generative AI, the creation of generative models is very crucial for learning and synthesizing complex data distributions within the dataset. By incorporating convolutional layers with Variational Autoencoders, we can create a such kind of generative model. In this article, we will dis\\
\\
10 min read](https://www.geeksforgeeks.org/convolutional-variational-autoencoder-in-tensorflow/?ref=ml_lbp)
[Implement Deep Autoencoder in PyTorch for Image Reconstruction\\
\\
\\
Since the availability of staggering amounts of data on the internet, researchers and scientists from industry and academia keep trying to develop more efficient and reliable data transfer modes than the current state-of-the-art methods. Autoencoders are one of the key elements found in recent times\\
\\
8 min read](https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/?ref=ml_lbp)
[Deep Learning with PyTorch \| An Introduction\\
\\
\\
PyTorch in a lot of ways behaves like the arrays we love from Numpy. These Numpy arrays, after all, are just tensors. PyTorch takes these tensors and makes it simple to move them to GPUs for the faster processing needed when training neural networks. It also provides a module that automatically calc\\
\\
7 min read](https://www.geeksforgeeks.org/deep-learning-with-pytorch-an-introduction/?ref=ml_lbp)
[Building a Convolutional Neural Network using PyTorch\\
\\
\\
Convolutional Neural Networks (CNNs) are deep learning models used for image processing tasks. They automatically learn spatial hierarchies of features from images through convolutional, pooling and fully connected layers. In this article we'll learn how to build a CNN model using PyTorch. This incl\\
\\
6 min read](https://www.geeksforgeeks.org/building-a-convolutional-neural-network-using-pytorch/?ref=ml_lbp)
[How to Define a Simple Convolutional Neural Network in PyTorch?\\
\\
\\
In this article, we are going to see how to Define a Simple Convolutional Neural Network in PyTorch using Python. Convolutional Neural Networks(CNN) is a type of Deep Learning algorithm which is highly instrumental in learning patterns and features in images. CNN has a unique trait which is its abil\\
\\
5 min read](https://www.geeksforgeeks.org/how-to-define-a-simple-convolutional-neural-network-in-pytorch/?ref=ml_lbp)
[Apply a 2D Convolution Operation in PyTorch\\
\\
\\
A 2D Convolution operation is a widely used operation in computer vision and deep learning. It is a mathematical operation that applies a filter to an image, producing a filtered output (also called a feature map). In this article, we will look at how to apply a 2D Convolution operation in PyTorch.\\
\\
8 min read](https://www.geeksforgeeks.org/apply-a-2d-convolution-operation-in-pytorch/?ref=ml_lbp)
[Extending PyTorch with Custom Activation Functions\\
\\
\\
In the context of deep learning and neural networks, activation functions are mathematical functions that are applied to the output of a neuron or a set of neurons. The output of the activation function is then passed on as input to the next layer of neurons. The purpose of an activation function is\\
\\
7 min read](https://www.geeksforgeeks.org/extending-pytorch-with-custom-activation-functions/?ref=ml_lbp)
[Python PyTorch â€“ torch.linalg.cond() Function\\
\\
\\
In this article, we are going to discuss how to compute the condition number of a matrix in PyTorch. we can get the condition number of a matrix by using torch.linalg.cond() method. torch.linalg.cond() method This method is used to compute the condition number of a matrix with respect to a matrix no\\
\\
3 min read](https://www.geeksforgeeks.org/python-pytorch-torch-linalg-cond-function/?ref=ml_lbp)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/implement-convolutional-autoencoder-in-pytorch-with-cuda/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=233693109.1745057322&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130498~103130500&z=471035673)