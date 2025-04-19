- [Python Tutorial](https://www.geeksforgeeks.org/python-programming-language-tutorial/)
- [Interview Questions](https://www.geeksforgeeks.org/python-interview-questions/)
- [Python Quiz](https://www.geeksforgeeks.org/python-quizzes/)
- [Python Glossary](https://www.geeksforgeeks.org/python-glossary/)
- [Python Projects](https://www.geeksforgeeks.org/python-projects-beginner-to-advanced/)
- [Practice Python](https://www.geeksforgeeks.org/python-exercises-practice-questions-and-solutions/)
- [Data Science With Python](https://www.geeksforgeeks.org/data-science-with-python-tutorial/)
- [Python Web Dev](https://www.geeksforgeeks.org/python-web-development-django/)
- [DSA with Python](https://www.geeksforgeeks.org/python-data-structures-and-algorithms/)
- [Python OOPs](https://www.geeksforgeeks.org/python-oops-concepts/)

Sign In

▲

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/generative-adversarial-network-gan/?type%3Darticle%26id%3D265975&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Use Cases of Generative Adversarial Networks\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/use-cases-of-generative-adversarial-networks/)

# Generative Adversarial Network (GAN)

Last Updated : 10 Mar, 2025

Comments

Improve

Suggest changes

Like Article

Like

Report

**Generative Adversarial Networks (GANs)** were introduced by Ian Goodfellow and his colleagues in 2014. GANs are a class of neural networks that autonomously learn patterns in the input data to generate new examples resembling the original dataset.

GAN’s architecture consists of two neural networks:

1. **Generator**: creates synthetic data from random noise to produce data so realistic that the discriminator cannot distinguish it from real data.
2. **Discriminator**: acts as a critic, evaluating whether the data it receives is real or fake.

They use adversarial training to produce artificial data that is identical to actual data.

![Generative-Adverarial-network-Gans](https://media.geeksforgeeks.org/wp-content/uploads/20250130181755083356/Generative-Adverarial-network-Gans.webp)

The two networks engage in a continuous game of cat and mouse: the Generator improves its ability to create realistic data, while the Discriminator becomes better at detecting fakes. Over time, this adversarial process leads to the generation of highly realistic and high-quality data.

## Detailed Architecture of GANs

Let’s explore the generator and discriminator model of GANs in detail:

### 1\. Generator Model

The **generator** is a deep neural network that takes random noise as input to generate realistic data samples (e.g., images or text). It learns the underlying data distribution by adjusting its parameters through [**backpropagation**](https://www.geeksforgeeks.org/backpropagation-in-neural-network/).

The generator’s objective is to produce samples that the discriminator classifies as real. The loss function is:

JG=−1mΣi=1mlogD(G(zi))J\_{G} = -\\frac{1}{m} \\Sigma^m \_{i=1} log D(G(z\_{i}))JG​=−m1​Σi=1m​logD(G(zi​))

Where,

- JGJ\_GJG​measure how well the generator is fooling the discriminator.
- log D(G(zi))D(G(z\_i) )D(G(zi​))represents log probability of the discriminator being correct for generated samples.
- The generator aims to minimize this loss, encouraging the production of samples that the discriminator classifies as real (logD(G(zi))(log D(G(z\_i))(logD(G(zi​)), close to 1.

### 2\. Discriminator Model

The **discriminator** acts as a **binary classifier**, distinguishing between real and generated data. It learns to improve its classification ability through training, refining its parameters to **detect fake samples more accurately**.

When dealing with image data, the discriminator often employs [**convolutional layers**](https://www.geeksforgeeks.org/what-are-convolution-layers/) or other relevant architectures suited to the data type. These layers help extract features and enhance the model’s ability to differentiate between real and generated samples.

The discriminator reduces the negative log likelihood of correctly classifying both produced and real samples. This loss incentivizes the discriminator to accurately categorize generated samples as fake and real samples with the following equation:

JD=−1mΣi=1mlogD(xi)–1mΣi=1mlog(1–D(G(zi))J\_{D} = -\\frac{1}{m} \\Sigma\_{i=1}^m log\\; D(x\_{i}) – \\frac{1}{m}\\Sigma\_{i=1}^m log(1 – D(G(z\_{i}))JD​=−m1​Σi=1m​logD(xi​)–m1​Σi=1m​log(1–D(G(zi​))

- JDJ\_DJD​ assesses the discriminator’s ability to discern between produced and actual samples.
- The log likelihood that the discriminator will accurately categorize real data is represented by logD(xi)logD(x\_i)logD(xi​).
- The log chance that the discriminator would correctly categorize generated samples as fake is represented by log⁡(1−D(G(zi)))log⁡(1-D(G(z\_i)))log⁡(1−D(G(zi​))).

By **minimizing this loss**, the discriminator becomes more effective at distinguishing between real and generated samples.

### MinMax Loss

GANs follow a minimax optimization where the generator and discriminator are adversaries:

minGmaxD(G,D)=\[Ex∼pdata\[logD(x)\]+Ez∼pz(z)\[log(1–D(g(z)))\]min\_{G}\\;max\_{D}(G,D) = \[\\mathbb{E}\_{x∼p\_{data}}\[log\\;D(x)\] + \\mathbb{E}\_{z∼p\_{z}(z)}\[log(1 – D(g(z)))\]minG​maxD​(G,D)=\[Ex∼pdata​​\[logD(x)\]+Ez∼pz​(z)​\[log(1–D(g(z)))\]\
\
Where,\
\
- G is generator network and is D is the discriminator network\
- Actual data samples obtained from the true data distribution pdata(x)p\_{data}(x)\
\
\
\
pdata​(x) are represented by x.\
- Random noise sampled from a previous distribution pz(z)p\_z(z) pz​(z)(usually a normal or uniform distribution) is represented by z.\
- D(x) represents the discriminator’s likelihood of correctly identifying actual data as real.\
- D(G(z)) is the likelihood that the discriminator will identify generated data coming from the generator as authentic.\
\
The generator aims to **minimize** the loss, while the discriminator tries to **maximize** its classification accuracy.\
\
![gans_gfg-(1)](https://media.geeksforgeeks.org/wp-content/uploads/20231122180335/gans_gfg-(1).jpg)\
\
## **How does a GAN work?**\
\
Let’s understand how the generator (G) and discriminator (D) complete to improve each other over time:\
\
### **1\. Generator’s First Move**\
\
G takes a random noise vector as input. This noise vector contains random values and acts as the starting point for G’s creation process. Using its internal layers and learned patterns, G transforms the noise vector into a new data sample, like a generated image.\
\
### **2\. Discriminator’s Turn**\
\
D receives two kinds of inputs:\
\
- Real data samples from the training dataset.\
- The data samples generated by G in the previous step.\
\
D’s job is to analyze each input and determine whether it’s real data or something G cooked up. It outputs a probability score between 0 and 1. A score of 1 indicates the data is likely real, and 0 suggests it’s fake.\
\
### **3\. Adversarial Learning**\
\
- If the discriminator correctly classifies real data as real and fake data as fake, it strengthens its ability slightly.\
- If the generator successfully fools the discriminator, it receives a positive update, while the discriminator is penalized.\
\
### **4\. Generator’s Improvement**\
\
Every time the discriminator misclassifies fake data as real, the generator learns and improves. Over multiple iterations, the generator produces more convincing synthetic samples.\
\
### **5\. Discriminator’s Adaptation**\
\
The discriminator continuously refines its ability to distinguish real from fake data. This ongoing duel between the generator and discriminator enhances the overall model’s learning process.\
\
### **6\. Training Progression**\
\
- As training continues, the generator becomes highly proficient at producing realistic data.\
- Eventually, the discriminator struggles to distinguish real from fake, indicating that the GAN has reached a well-trained state.\
- At this point, the generator can be used to generate high-quality synthetic data for various applications.\
\
## Types of GANs\
\
### **1\. Vanilla GAN**\
\
Vanilla GAN is the simplest type of GAN. It consists of:\
\
- A generator and a discriminator, both are built using multi-layer perceptrons (MLPs).\
- The model optimizes its mathematical formulation using stochastic gradient descent (SGD).\
\
While Vanilla GANs serve as the foundation for more advanced GAN models, they often struggle with issues like mode collapse and unstable training.\
\
### **2\. Conditional GAN (CGAN)**\
\
[Conditional GANs (CGANs)](https://www.geeksforgeeks.org/conditional-generative-adversarial-network/) introduce an additional conditional parameter to guide the generation process. Instead of generating data randomly, CGANs allow the model to produce specific types of outputs.\
\
Working of CGANs:\
\
- A conditional variable (y) is fed into both the generator and the discriminator.\
- This ensures that the generator creates data corresponding to the given condition (e.g., generating images of specific objects).\
- The discriminator also receives the labels to help distinguish between real and fake data.\
\
### **3\. Deep Convolutional GAN (DCGAN)**\
\
[Deep Convolutional GANs (DCGANs)](https://www.geeksforgeeks.org/deep-convolutional-gan-with-keras/) are among the most popular and widely used types of GANs, particularly for image generation.\
\
What Makes DCGAN Special?\
\
- Uses Convolutional Neural Networks (CNNs) instead of simple multi-layer perceptrons (MLPs).\
- Max pooling layers are replaced with convolutional stride, making the model more efficient.\
- Fully connected layers are removed, allowing for better spatial understanding of images.\
\
DCGANs have been highly successful in generating high-quality images, making them a go-to choice for deep learning researchers.\
\
### **4\. Laplacian Pyramid GAN (LAPGAN)**\
\
Laplacian Pyramid GAN (LAPGAN) is designed to generate ultra-high-quality images by leveraging a multi-resolution approach.\
\
Working of LAPGAN:\
\
- Uses multiple generator-discriminator pairs at different levels of the Laplacian pyramid.\
- Images are first downsampled at each layer of the pyramid and upscaled again using Conditional GANs (CGANs).\
- This process allows the image to gradually refine details, reducing noise and improving clarity.\
\
Due to its ability to generate highly detailed images, LAPGAN is considered a superior approach for photorealistic image generation.\
\
### **5\. Super Resolution GAN (SRGAN)**\
\
[Super-Resolution GAN (SRGAN)](https://www.geeksforgeeks.org/super-resolution-gan-srgan/) is specifically designed to increase the resolution of low-quality images while preserving details.\
\
Working of SRGAN:\
\
- Uses a deep neural network combined with an adversarial loss function.\
- Enhances low-resolution images by adding finer details, making them appear sharper and more realistic.\
- Helps reduce common image upscaling errors, such as blurriness and pixelation.\
\
## Implementation of Generative Adversarial Network (GAN) using PyTorch\
\
Let’s explore the implementation of a Generative Adversarial Network (GAN). Our GAN will be trained on the CIFAR-10 dataset to generate realistic images.\
\
### **Step 1: Importing Required Libraries**\
\
First, we import the necessary libraries for building and training our GAN.\
\
Python`\
import torch\
import torch.nn as nn\
import torch.optim as optim\
import torchvision\
from torchvision import datasets, transforms\
import matplotlib.pyplot as plt\
import numpy as np\
# Set device\
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
`\
\
The model will utilize a **GPU** if available; otherwise, it will default to **CPU**.\
\
### Step 2: Defining Image Transformations\
\
We use **PyTorch’s transforms** to normalize and convert images into tensors before feeding them into the model.\
\
Python`\
# Define a basic transform\
transform = transforms.Compose([\
    transforms.ToTensor(),\
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\
])\
`\
\
### **Step 3: Loading the CIFAR-10 Dataset**\
\
The CIFAR-10 dataset is loaded with predefined transformations. A DataLoader is created to process the dataset in mini-batches of 32 images, shuffled for randomness.\
\
Python`\
train_dataset = datasets.CIFAR10(root='./data',\\
              train=True, download=True, transform=transform)\
dataloader = torch.utils.data.DataLoader(train_dataset, \\
                                batch_size=32, shuffle=True)\
`\
\
### **Step 4: Defining GAN Hyperparameters**\
\
Key hyperparameters are defined:\
\
- latent\_dim – Dimensionality of the noise vector.\
- lr – Learning rate of the optimizer.\
- beta1, beta2 – Adam optimizer coefficients.\
- num\_epochs – Total training iterations.\
\
Python`\
# Hyperparameters\
latent_dim = 100\
lr = 0.0002\
beta1 = 0.5\
beta2 = 0.999\
num_epochs = 10\
`\
\
### **Step 5: Building the Generator**\
\
The generator takes a random latent vector (z) as input and transforms it into an image through convolutional, batch normalization, and upsampling layers. The final output uses Tanh activation to ensure pixel values are within the expected range.\
\
Python`\
# Define the generator\
class Generator(nn.Module):\
    def __init__(self, latent_dim):\
        super(Generator, self).__init__()\
        self.model = nn.Sequential(\
            nn.Linear(latent_dim, 128 * 8 * 8),\
            nn.ReLU(),\
            nn.Unflatten(1, (128, 8, 8)),\
            nn.Upsample(scale_factor=2),\
            nn.Conv2d(128, 128, kernel_size=3, padding=1),\
            nn.BatchNorm2d(128, momentum=0.78),\
            nn.ReLU(),\
            nn.Upsample(scale_factor=2),\
            nn.Conv2d(128, 64, kernel_size=3, padding=1),\
            nn.BatchNorm2d(64, momentum=0.78),\
            nn.ReLU(),\
            nn.Conv2d(64, 3, kernel_size=3, padding=1),\
            nn.Tanh()\
        )\
    def forward(self, z):\
        img = self.model(z)\
        return img\
`\
\
### **Step 6: Building the Discriminator**\
\
The discriminator is a binary classifier that distinguishes between real and generated images. It consists of convolutional layers, batch normalization, dropout, and LeakyReLU activation to improve learning stability.\
\
Python`\
# Define the discriminator\
class Discriminator(nn.Module):\
    def __init__(self):\
        super(Discriminator, self).__init__()\
        self.model = nn.Sequential(\
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),\
        nn.LeakyReLU(0.2),\
        nn.Dropout(0.25),\
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),\
        nn.ZeroPad2d((0, 1, 0, 1)),\
        nn.BatchNorm2d(64, momentum=0.82),\
        nn.LeakyReLU(0.25),\
        nn.Dropout(0.25),\
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),\
        nn.BatchNorm2d(128, momentum=0.82),\
        nn.LeakyReLU(0.2),\
        nn.Dropout(0.25),\
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),\
        nn.BatchNorm2d(256, momentum=0.8),\
        nn.LeakyReLU(0.25),\
        nn.Dropout(0.25),\
        nn.Flatten(),\
        nn.Linear(256 * 5 * 5, 1),\
        nn.Sigmoid()\
    )\
    def forward(self, img):\
        validity = self.model(img)\
        return validity\
`\
\
### **Step 7: Initializing GAN Components**\
\
- **Generator and Discriminator** are initialized on the available device (GPU or CPU).\
- **Binary Cross-Entropy (BCE) Loss** is chosen as the loss function.\
- **Adam optimizers** are defined separately for the generator and discriminator with specified learning rates and betas.\
\
Python`\
# Define the generator and discriminator\
# Initialize generator and discriminator\
generator = Generator(latent_dim).to(device)\
discriminator = Discriminator().to(device)\
# Loss function\
adversarial_loss = nn.BCELoss()\
# Optimizers\
optimizer_G = optim.Adam(generator.parameters()\\
                         , lr=lr, betas=(beta1, beta2))\
optimizer_D = optim.Adam(discriminator.parameters()\\
                         , lr=lr, betas=(beta1, beta2))\
`\
\
### **Step 8: Training the GAN**\
\
- The discriminator is trained to differentiate between real and fake images.\
- The generator is trained to produce realistic images that fool the discriminator.\
- The loss is backpropagated using Adam optimizers, and the model updates its parameters.\
- Progress tracking: Loss values for both networks are printed, and generated images are displayed every 10 epochs for visual inspection.\
\
Python`\
# Training loop\
for epoch in range(num_epochs):\
    for i, batch in enumerate(dataloader):\
       # Convert list to tensor\
        real_images = batch[0].to(device)\
        # Adversarial ground truths\
        valid = torch.ones(real_images.size(0), 1, device=device)\
        fake = torch.zeros(real_images.size(0), 1, device=device)\
        # Configure input\
        real_images = real_images.to(device)\
        # ---------------------\
        #  Train Discriminator\
        # ---------------------\
        optimizer_D.zero_grad()\
        # Sample noise as generator input\
        z = torch.randn(real_images.size(0), latent_dim, device=device)\
        # Generate a batch of images\
        fake_images = generator(z)\
        # Measure discriminator's ability\
        # to classify real and fake images\
        real_loss = adversarial_loss(discriminator\\
                                     (real_images), valid)\
        fake_loss = adversarial_loss(discriminator\\
                                     (fake_images.detach()), fake)\
        d_loss = (real_loss + fake_loss) / 2\
        # Backward pass and optimize\
        d_loss.backward()\
        optimizer_D.step()\
        # -----------------\
        #  Train Generator\
        # -----------------\
        optimizer_G.zero_grad()\
        # Generate a batch of images\
        gen_images = generator(z)\
        # Adversarial loss\
        g_loss = adversarial_loss(discriminator(gen_images), valid)\
        # Backward pass and optimize\
        g_loss.backward()\
        optimizer_G.step()\
        # ---------------------\
        #  Progress Monitoring\
        # ---------------------\
        if (i + 1) % 100 == 0:\
            print(\
                f"Epoch [{epoch+1}/{num_epochs}]\\
                        Batch {i+1}/{len(dataloader)} "\
                f"Discriminator Loss: {d_loss.item():.4f} "\
                f"Generator Loss: {g_loss.item():.4f}"\
            )\
    # Save generated images for every epoch\
    if (epoch + 1) % 10 == 0:\
        with torch.no_grad():\
            z = torch.randn(16, latent_dim, device=device)\
            generated = generator(z).detach().cpu()\
            grid = torchvision.utils.make_grid(generated,\\
                                        nrow=4, normalize=True)\
            plt.imshow(np.transpose(grid, (1, 2, 0)))\
            plt.axis("off")\
            plt.show()\
`\
\
**Output:**\
\
```\
Epoch [10/10]                        Batch 1300/1563 Discriminator Loss: 0.4473 Generator Loss: 0.9555\
Epoch [10/10]                        Batch 1400/1563 Discriminator Loss: 0.6643 Generator Loss: 1.0215\
Epoch [10/10]                        Batch 1500/1563 Discriminator Loss: 0.4720 Generator Loss: 2.5027\
\
```\
\
![gan-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20231123110137/gan.jpg)\
\
GAN Output\
\
We have successfully implemented a GAN in PyTorch and trained it on CIFAR-10. The generator learns to create realistic images, while the discriminator helps in refining them through adversarial training.\
\
## Application Of Generative Adversarial Networks (GANs)\
\
1. **Image Synthesis & Generation:** GANs generate realistic images, avatars, and high-resolution visuals by learning patterns from training data. They are widely used in art, gaming, and AI-driven design.\
2. **Image-to-Image Translation:** GANs can transform images between domains while preserving key features. Examples include converting day images to night, sketches to realistic images, or changing artistic styles.\
3. **Text-to-Image Synthesis:** GANs create visuals from textual descriptions, enabling applications in AI-generated art, automated design, and content creation.\
4. **Data Augmentation:** GANs generate synthetic data to improve machine learning models, making them more robust and generalizable, especially in fields with limited labeled data.\
5. **High-Resolution Image Enhancement:** GANs upscale low-resolution images, improving clarity for applications like medical imaging, satellite imagery, and video enhancement.\
\
## Advantages of GAN\
\
The advantages of the GANs are as follows:\
\
1. **Synthetic data generation**: GANs can generate new, synthetic data that resembles some known data distribution, which can be useful for data augmentation, anomaly detection, or creative applications.\
2. **High-quality results**: GANs can produce high-quality, photorealistic results in image synthesis, video synthesis, music synthesis, and other tasks.\
3. **Unsupervised learning**: GANs can be trained without labeled data, making them suitable for unsupervised learning tasks, where labeled data is scarce or difficult to obtain.\
4. **Versatility**: GANs can be applied to a wide range of tasks, including image synthesis, text-to-image synthesis, image-to-image translation, anomaly detection, data augmentation, and others.\
\
Comment\
\
\
More info\
\
[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)\
\
[Next Article](https://www.geeksforgeeks.org/use-cases-of-generative-adversarial-networks/)\
\
[Use Cases of Generative Adversarial Networks](https://www.geeksforgeeks.org/use-cases-of-generative-adversarial-networks/)\
\
[R](https://www.geeksforgeeks.org/user/Rahul_Roy/)\
\
[Rahul\_Roy](https://www.geeksforgeeks.org/user/Rahul_Roy/)\
\
Follow\
\
Improve\
\
Article Tags :\
\
- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)\
- [Deep Learning](https://www.geeksforgeeks.org/category/ai-ml-ds/deep-learning/)\
- [Python](https://www.geeksforgeeks.org/category/programming-language/python/)\
- [Python-Quizzes](https://www.geeksforgeeks.org/category/programming-language/python/python-quizzes-gq/)\
- [Technical Scripter](https://www.geeksforgeeks.org/category/technical-scripter/)\
- [python](https://www.geeksforgeeks.org/tag/python/)\
\
+2 More\
\
Practice Tags :\
\
- [python](https://www.geeksforgeeks.org/explore?category=python)\
- [python](https://www.geeksforgeeks.org/explore?category=python)\
\
### Similar Reads\
\
[Deep Learning Tutorial\\
\\
\\
Deep Learning tutorial covers the basics and more advanced topics, making it perfect for beginners and those with experience. Whether you're just starting or looking to expand your knowledge, this guide makes it easy to learn about the different technologies of Deep Learning. Deep Learning is a bran\\
\\
5 min read](https://www.geeksforgeeks.org/deep-learning-tutorial/)\
\
## Introduction to Deep Learning\
\
- [Introduction to Deep Learning\\
\\
\\
Deep Learning is transforming the way machines understand, learn, and interact with complex data. Deep learning mimics neural networks of the human brain, it enables computers to autonomously uncover patterns and make informed decisions from vast amounts of unstructured data. Deep Learning leverages\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-deep-learning/)\
\
* * *\
\
- [Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning\\
\\
\\
Artificial Intelligence is basically the mechanism to incorporate human intelligence into machines through a set of rules(algorithm). AI is a combination of two words: "Artificial" meaning something made by humans or non-natural things and "Intelligence" meaning the ability to understand or think ac\\
\\
14 min read](https://www.geeksforgeeks.org/difference-between-artificial-intelligence-vs-machine-learning-vs-deep-learning/)\
\
* * *\
\
\
## Basic Neural Network\
\
- [Difference between ANN and BNN\\
\\
\\
Do you ever think of what it's like to build anything like a brain, how these things work, or what they do? Let us look at how nodes communicate with neurons and what are some differences between artificial and biological neural networks. 1. Artificial Neural Network: Artificial Neural Network (ANN)\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-ann-and-bnn/)\
\
* * *\
\
- [Single Layer Perceptron in TensorFlow\\
\\
\\
Single Layer Perceptron is inspired by biological neurons and their ability to process information. To understand the SLP we first need to break down the workings of a single artificial neuron which is the fundamental building block of neural networks. An artificial neuron is a simplified computatio\\
\\
4 min read](https://www.geeksforgeeks.org/single-layer-perceptron-in-tensorflow/)\
\
* * *\
\
- [Multi-Layer Perceptron Learning in Tensorflow\\
\\
\\
Multi-Layer Perceptron (MLP) is an artificial neural network widely used for solving classification and regression tasks. MLP consists of fully connected dense layers that transform input data from one dimension to another. It is called "multi-layer" because it contains an input layer, one or more h\\
\\
9 min read](https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)\
\
* * *\
\
- [Deep Neural net with forward and back propagation from scratch - Python\\
\\
\\
This article aims to implement a deep neural network from scratch. We will implement a deep neural network containing two input layers, a hidden layer with four units and one output layer. The implementation will go from scratch and the following steps will be implemented. Algorithm:1. Loading and v\\
\\
6 min read](https://www.geeksforgeeks.org/deep-neural-net-with-forward-and-back-propagation-from-scratch-python/)\
\
* * *\
\
- [Understanding Multi-Layer Feed Forward Networks\\
\\
\\
Let's understand how errors are calculated and weights are updated in backpropagation networks(BPNs). Consider the following network in the below figure. The network in the above figure is a simple multi-layer feed-forward network or backpropagation network. It contains three layers, the input layer\\
\\
7 min read](https://www.geeksforgeeks.org/understanding-multi-layer-feed-forward-networks/)\
\
* * *\
\
- [List of Deep Learning Layers\\
\\
\\
Deep learning (DL) is characterized by the use of neural networks with multiple layers to model and solve complex problems. Each layer in the neural network plays a unique role in the process of converting input data into meaningful and insightful outputs. The article explores the layers that are us\\
\\
7 min read](https://www.geeksforgeeks.org/ml-list-of-deep-learning-layers/)\
\
* * *\
\
\
## Activation Functions\
\
- [Activation Functions\\
\\
\\
To put it in simple terms, an artificial neuron calculates the 'weighted sum' of its inputs and adds a bias, as shown in the figure below by the net input. Mathematically, Net Input=∑(Weight×Input)+Bias\\text{Net Input} =\\sum \\text{(Weight} \\times \\text{Input)+Bias}Net Input=∑(Weight×Input)+Bias Now the value of net input can be any anything from -\\
\\
3 min read](https://www.geeksforgeeks.org/activation-functions/)\
\
* * *\
\
- [Types Of Activation Function in ANN\\
\\
\\
The biological neural network has been modeled in the form of Artificial Neural Networks with artificial neurons simulating the function of a biological neuron. The artificial neuron is depicted in the below picture: Each neuron consists of three major components:Â  A set of 'i' synapses having weigh\\
\\
4 min read](https://www.geeksforgeeks.org/types-of-activation-function-in-ann/)\
\
* * *\
\
- [Activation Functions in Pytorch\\
\\
\\
In this article, we will Understand PyTorch Activation Functions. What is an activation function and why to use them?Activation functions are the building blocks of Pytorch. Before coming to types of activation function, let us first understand the working of neurons in the human brain. In the Artif\\
\\
5 min read](https://www.geeksforgeeks.org/activation-functions-in-pytorch/)\
\
* * *\
\
- [Understanding Activation Functions in Depth\\
\\
\\
In artificial neural networks, the activation function of a neuron determines its output for a given input. This output serves as the input for subsequent neurons in the network, continuing the process until the network solves the original problem. Consider a binary classification problem, where the\\
\\
6 min read](https://www.geeksforgeeks.org/understanding-activation-functions-in-depth/)\
\
* * *\
\
\
## Artificial Neural Network\
\
- [Artificial Neural Networks and its Applications\\
\\
\\
As you read this article, which organ in your body is thinking about it? It's the brain, of course! But do you know how the brain works? Well, it has neurons or nerve cells that are the primary units of both the brain and the nervous system. These neurons receive sensory input from the outside world\\
\\
9 min read](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)\
\
* * *\
\
- [Gradient Descent Optimization in Tensorflow\\
\\
\\
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function. In other words, gradient descent is an iterative algorithm that helps to find the optimal solution to a given problem. In this blog, we will discuss gr\\
\\
15+ min read](https://www.geeksforgeeks.org/gradient-descent-optimization-in-tensorflow/)\
\
* * *\
\
- [Choose Optimal Number of Epochs to Train a Neural Network in Keras\\
\\
\\
One of the critical issues while training a neural network on the sample data is Overfitting. When the number of epochs used to train a neural network model is more than necessary, the training model learns patterns that are specific to sample data to a great extent. This makes the model incapable t\\
\\
6 min read](https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/)\
\
* * *\
\
\
## Classification\
\
- [Python \| Classify Handwritten Digits with Tensorflow\\
\\
\\
Classifying handwritten digits is the basic problem of the machine learning and can be solved in many ways here we will implement them by using TensorFlowUsing a Linear Classifier Algorithm with tf.contrib.learn linear classifier achieves the classification of handwritten digits by making a choice b\\
\\
4 min read](https://www.geeksforgeeks.org/python-classifying-handwritten-digits-with-tensorflow/)\
\
* * *\
\
- [Train a Deep Learning Model With Pytorch\\
\\
\\
Neural Network is a type of machine learning model inspired by the structure and function of human brain. It consists of layers of interconnected nodes called neurons which process and transmit information. Neural networks are particularly well-suited for tasks such as image and speech recognition,\\
\\
6 min read](https://www.geeksforgeeks.org/train-a-deep-learning-model-with-pytorch/)\
\
* * *\
\
\
## Regression\
\
- [Linear Regression using PyTorch\\
\\
\\
Linear Regression is a very commonly used statistical method that allows us to determine and study the relationship between two continuous variables. The various properties of linear regression and its Python implementation have been covered in this article previously. Now, we shall find out how to\\
\\
4 min read](https://www.geeksforgeeks.org/linear-regression-using-pytorch/)\
\
* * *\
\
- [Linear Regression Using Tensorflow\\
\\
\\
We will briefly summarize Linear Regression before implementing it using TensorFlow. Since we will not get into the details of either Linear Regression or Tensorflow, please read the following articles for more details: Linear Regression (Python Implementation)Introduction to TensorFlowIntroduction\\
\\
6 min read](https://www.geeksforgeeks.org/linear-regression-using-tensorflow/)\
\
* * *\
\
\
## Hyperparameter tuning\
\
- [Hyperparameter tuning\\
\\
\\
Machine Learning model is defined as a mathematical model with several parameters that need to be learned from the data. By training a model with existing data we can fit the model parameters. However there is another kind of parameter known as hyperparameters which cannot be directly learned from t\\
\\
8 min read](https://www.geeksforgeeks.org/hyperparameter-tuning/)\
\
* * *\
\
\
## Introduction to Convolution Neural Network\
\
- [Introduction to Convolution Neural Network\\
\\
\\
Convolutional Neural Network (CNN) is an advanced version of artificial neural networks (ANNs), primarily designed to extract features from grid-like matrix datasets. This is particularly useful for visual datasets such as images or videos, where data patterns play a crucial role. CNNs are widely us\\
\\
8 min read](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)\
\
* * *\
\
- [Digital Image Processing Basics\\
\\
\\
Digital Image Processing means processing digital image by means of a digital computer. We can also say that it is a use of computer algorithms, in order to get enhanced image either to extract some useful information. Digital image processing is the use of algorithms and mathematical models to proc\\
\\
7 min read](https://www.geeksforgeeks.org/digital-image-processing-basics/)\
\
* * *\
\
- [Difference between Image Processing and Computer Vision\\
\\
\\
Image processing and Computer Vision both are very exciting field of Computer Science. Computer Vision: In Computer Vision, computers or machines are made to gain high-level understanding from the input digital images or videos with the purpose of automating tasks that the human visual system can do\\
\\
2 min read](https://www.geeksforgeeks.org/difference-between-image-processing-and-computer-vision/)\
\
* * *\
\
- [CNN \| Introduction to Pooling Layer\\
\\
\\
Pooling layer is used in CNNs to reduce the spatial dimensions (width and height) of the input feature maps while retaining the most important information. It involves sliding a two-dimensional filter over each channel of a feature map and summarizing the features within the region covered by the fi\\
\\
5 min read](https://www.geeksforgeeks.org/cnn-introduction-to-pooling-layer/)\
\
* * *\
\
- [CIFAR-10 Image Classification in TensorFlow\\
\\
\\
Prerequisites:Image ClassificationConvolution Neural Networks including basic pooling, convolution layers with normalization in neural networks, and dropout.Data Augmentation.Neural Networks.Numpy arrays.In this article, we are going to discuss how to classify images using TensorFlow. Image Classifi\\
\\
8 min read](https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/)\
\
* * *\
\
- [Implementation of a CNN based Image Classifier using PyTorch\\
\\
\\
Introduction: Introduced in the 1980s by Yann LeCun, Convolution Neural Networks(also called CNNs or ConvNets) have come a long way. From being employed for simple digit classification tasks, CNN-based architectures are being used very profoundly over much Deep Learning and Computer Vision-related t\\
\\
9 min read](https://www.geeksforgeeks.org/implementation-of-a-cnn-based-image-classifier-using-pytorch/)\
\
* * *\
\
- [Convolutional Neural Network (CNN) Architectures\\
\\
\\
Convolutional Neural Network(CNN) is a neural network architecture in Deep Learning, used to recognize the pattern from structured arrays. However, over many years, CNN architectures have evolved. Many variants of the fundamental CNN Architecture This been developed, leading to amazing advances in t\\
\\
11 min read](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-architectures/)\
\
* * *\
\
- [Object Detection vs Object Recognition vs Image Segmentation\\
\\
\\
Object Recognition: Object recognition is the technique of identifying the object present in images and videos. It is one of the most important applications of machine learning and deep learning. The goal of this field is to teach machines to understand (recognize) the content of an image just like\\
\\
5 min read](https://www.geeksforgeeks.org/object-detection-vs-object-recognition-vs-image-segmentation/)\
\
* * *\
\
- [YOLO v2 - Object Detection\\
\\
\\
In terms of speed, YOLO is one of the best models in object recognition, able to recognize objects and process frames at the rate up to 150 FPS for small networks. However, In terms of accuracy mAP, YOLO was not the state of the art model but has fairly good Mean average Precision (mAP) of 63% when\\
\\
6 min read](https://www.geeksforgeeks.org/yolo-v2-object-detection/)\
\
* * *\
\
\
## Recurrent Neural Network\
\
- [Natural Language Processing (NLP) Tutorial\\
\\
\\
Natural Language Processing (NLP) is the branch of Artificial Intelligence (AI) that gives the ability to machine understand and process human languages. Human languages can be in the form of text or audio format. Applications of NLPThe applications of Natural Language Processing are as follows: Voi\\
\\
5 min read](https://www.geeksforgeeks.org/natural-language-processing-nlp-tutorial/)\
\
* * *\
\
- [Introduction to NLTK: Tokenization, Stemming, Lemmatization, POS Tagging\\
\\
\\
Natural Language Toolkit (NLTK) is one of the largest Python libraries for performing various Natural Language Processing tasks. From rudimentary tasks such as text pre-processing to tasks like vectorized representation of text - NLTK's API has covered everything. In this article, we will accustom o\\
\\
5 min read](https://www.geeksforgeeks.org/introduction-to-nltk-tokenization-stemming-lemmatization-pos-tagging/)\
\
* * *\
\
- [Word Embeddings in NLP\\
\\
\\
Word Embeddings are numeric representations of words in a lower-dimensional space, capturing semantic and syntactic information. They play a vital role in Natural Language Processing (NLP) tasks. This article explores traditional and neural approaches, such as TF-IDF, Word2Vec, and GloVe, offering i\\
\\
15+ min read](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)\
\
* * *\
\
- [Introduction to Recurrent Neural Networks\\
\\
\\
Recurrent Neural Networks (RNNs) work a bit different from regular neural networks. In neural network the information flows in one direction from input to output. However in RNN information is fed back into the system after each step. Think of it like reading a sentence, when you're trying to predic\\
\\
12 min read](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)\
\
* * *\
\
- [Recurrent Neural Networks Explanation\\
\\
\\
Today, different Machine Learning techniques are used to handle different types of data. One of the most difficult types of data to handle and the forecast is sequential data. Sequential data is different from other types of data in the sense that while all the features of a typical dataset can be a\\
\\
8 min read](https://www.geeksforgeeks.org/recurrent-neural-networks-explanation/)\
\
* * *\
\
- [Sentiment Analysis with an Recurrent Neural Networks (RNN)\\
\\
\\
Recurrent Neural Networks (RNNs) excel in sequence tasks such as sentiment analysis due to their ability to capture context from sequential data. In this article we will be apply RNNs to analyze the sentiment of customer reviews from Swiggy food delivery platform. The goal is to classify reviews as\\
\\
3 min read](https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/)\
\
* * *\
\
- [Short term Memory\\
\\
\\
In the wider community of neurologists and those who are researching the brain, It is agreed that two temporarily distinct processes contribute to the acquisition and expression of brain functions. These variations can result in long-lasting alterations in neuron operations, for instance through act\\
\\
5 min read](https://www.geeksforgeeks.org/short-term-memory/)\
\
* * *\
\
- [What is LSTM - Long Short Term Memory?\\
\\
\\
Long Short-Term Memory (LSTM) is an enhanced version of the Recurrent Neural Network (RNN) designed by Hochreiter & Schmidhuber. LSTMs can capture long-term dependencies in sequential data making them ideal for tasks like language translation, speech recognition and time series forecasting. Unli\\
\\
7 min read](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)\
\
* * *\
\
- [Long Short Term Memory Networks Explanation\\
\\
\\
Prerequisites: Recurrent Neural Networks To solve the problem of Vanishing and Exploding Gradients in a Deep Recurrent Neural Network, many variations were developed. One of the most famous of them is the Long Short Term Memory Network(LSTM). In concept, an LSTM recurrent unit tries to "remember" al\\
\\
7 min read](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/)\
\
* * *\
\
- [LSTM - Derivation of Back propagation through time\\
\\
\\
Long Short-Term Memory (LSTM) are a type of neural network designed to handle long-term dependencies by handling the vanishing gradient problem. One of the fundamental techniques used to train LSTMs is Backpropagation Through Time (BPTT) where we have sequential data. In this article we summarize ho\\
\\
4 min read](https://www.geeksforgeeks.org/lstm-derivation-of-back-propagation-through-time/)\
\
* * *\
\
- [Text Generation using Recurrent Long Short Term Memory Network\\
\\
\\
LSTMs are a type of neural network that are well-suited for tasks involving sequential data such as text generation. They are particularly useful because they can remember long-term dependencies in the data which is crucial when dealing with text that often has context that spans over multiple words\\
\\
6 min read](https://www.geeksforgeeks.org/text-generation-using-recurrent-long-short-term-memory-network/)\
\
* * *\
\
\
Like\
\
We use cookies to ensure you have the best browsing experience on our website. By using our site, you\
acknowledge that you have read and understood our\
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &\
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)\
Got It !\
\
\
![Lightbox](https://www.geeksforgeeks.org/generative-adversarial-network-gan/)\
\
Improvement\
\
Suggest changes\
\
Suggest Changes\
\
Help us improve. Share your suggestions to enhance the article. Contribute your expertise and make a difference in the GeeksforGeeks portal.\
\
![geeksforgeeks-suggest-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/suggestChangeIcon.png)\
\
Create Improvement\
\
Enhance the article with your expertise. Contribute to the GeeksforGeeks community and help create better learning resources for all.\
\
![geeksforgeeks-improvement-icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/createImprovementIcon.png)\
\
Suggest Changes\
\
min 4 words, max Words Limit:1000\
\
## Thank You!\
\
Your suggestions are valuable to us.\
\
## What kind of Experience do you want to share?\
\
[Interview Experiences](https://write.geeksforgeeks.org/posts-new?cid=e8fc46fe-75e7-4a4b-be3c-0c862d655ed0) [Admission Experiences](https://write.geeksforgeeks.org/posts-new?cid=82536bdb-84e6-4661-87c3-e77c3ac04ede) [Career Journeys](https://write.geeksforgeeks.org/posts-new?cid=5219b0b2-7671-40a0-9bda-503e28a61c31) [Work Experiences](https://write.geeksforgeeks.org/posts-new?cid=22ae3354-15b6-4dd4-a5b4-5c7a105b8a8f) [Campus Experiences](https://write.geeksforgeeks.org/posts-new?cid=c5e1ac90-9490-440a-a5fa-6180c87ab8ae) [Competitive Exam Experiences](https://write.geeksforgeeks.org/posts-new?cid=5ebb8fe9-b980-4891-af07-f2d62a9735f2)\
\
[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=938650290.1745057248&gtm=45je54g3v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103130495~103130497&z=340394473)