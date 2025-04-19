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

[Open In App](https://geeksforgeeksapp.page.link/?link=https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/?type%3Darticle%26id%3D1022491&apn=free.programming.programming&isi=1641848816&ibi=org.geeksforgeeks.GeeksforGeeksDev&efr=1)

[Next Article:\\
\\
Gradient Descent With RMSProp from Scratch\\
\\
![Next article icon](https://media.geeksforgeeks.org/auth-dashboard-uploads/ep_right.svg)](https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/)

# Bidirectional LSTM in NLP

Last Updated : 26 Feb, 2025

Comments

Improve

Suggest changes

2 Likes

Like

Report

[Long Short-Term Memory (LSTM)](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/) are a type of Neural Network designed to handle long-term dependencies and overcome the vanishing gradient problem of RNN. It uses a memory cell along with input, forget and output gates to selectively retain or discard information.

**Bidirectional Long Short-Term Memory (BiLSTM)** is an extension of the traditional LSTM (Long Short-Term Memory) network. Unlike conventional LSTMs that process sequences in only one direction, BiLSTMs allow information to flow from both forward and backward enabling them to capture richer contextual information. This makes BiLSTMs particularly effective for tasks where understanding both past and future context is crucial. In this article we will learn more about them and implement a sentiment analysis model using BiLSTM in TensorFlow.

## **Understanding Bidirectional LSTM (BiLSTM)**

A **Bidirectional LSTM (BiLSTM)** consists of two **separate** LSTM layers:

- **Forward LSTM**: Processes the sequence from start to end
- **Backward LSTM**: Processes the sequence from end to start

The outputs of both LSTMs are then combined to form the final output. Mathematically, the final output at time **t** is computed as:

pt=ptf+ptbp\_t = p\_{t\_f} + p\_{t\_b}pt​=ptf​​+ptb​​

where,

- ptp\_t


pt​: Final probability vector of the network.
- ptfp\_{tf}


ptf​: Probability vector from the forward LSTM network.
- ptbp\_{tb}


ptb​: Probability vector from the backward LSTM network.

The following diagram represents the **BiLSTM layer**:

![Bidirectional LSTM layer Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20230529103946/Bidirectional-LSTM-(1).jpg)

Bidirectional LSTM layer Architecture

Here

- XiX\_i Xi​is the input token
- YiY\_i Yi​is the output token
- A and A’ A’ A’ are Forward and backward LSTM units
- The final output of YiY\_i Yi​is the combination of AA Aand A’ A’ A’ LSTM nodes.

## **Implementation: Sentiment Analysis Using BiLSTM**

Now let us look into an implementation of a review system using BiLSTM layers in Python using Tensorflow. We would be performing **sentiment analysis on the IMDB movie review dataset.** We would implement the network from scratch and train it to identify if the review is positive or negative.

### **Step 1: Importing Libraries**

To implement sentiment analysis using bidirectional LSTM, we will be using python libraries like numpy, pandas and tenserflow. Tenserflow is the key library for building our model.

Python`
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
`

### **Step 2: Loading and Preparing the IMDB Dataset**

- We loaded IMDB dataset from tenserflow which contains **25,000** labeled movie reviews for training and testing.
- **Shuffling** ensures that the model does not learn patterns based on the order of reviews.

Python`
dataset = tfds.load('imdb_reviews', as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
batch_size = 32
train_dataset = train_dataset.shuffle(10000).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
`

Printing a sample review and its label from the training set.

Python`
example, label = next(iter(train_dataset))
print('Text:\n', example.numpy()[0])
print('\nLabel: ', label.numpy()[0])
`

**Output:**

> Text: b “Having seen men Behind the Sun … 1 as a treatment of the subject).”
>
> Label: 0

### **Step 3: Text Vectorization**

We will first perform [text vectorization](https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/) and let the encoder map all the words in the training dataset to a token. We can also see in the example below how we can encode and decode the sample review into a vector of integers.

- **vectorize\_layer :** tokenizes and normalizes the text.
- It converts words into numeric values for the neural network to process easily.

Python`
vectorize_layer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=100)
vectorize_layer.adapt(train_dataset.map(lambda x, y: x))
`

### **Step 4: Model Architecture (BiLSTM Layers)**

In this section we define the model for sentiment analysis. The first layer **Text Vectorization** encodes input text into token indices. These tokens are passed through an **embedding layer** which maps words to trainable vectors. Over training these vectors adjust such that words with similar meanings have similar representations. The **Bidirectional LSTM** layers then process these sequences to generate meaningful representations which are converted for classification.

- **Embedding Layer**: Converts words into trainable word vectors.
- **Bidirectional LSTM (64 units)**: First layer processes in both directions.
- **Bidirectional LSTM (32 units)**: Second layer refines the learned patterns.
- **Dense Layers**: Fully connected layers for classification.
- **Final Output**: A single neuron output to classify sentiment.

Python`
model = tf.keras.Sequential([\
    vectorize_layer,\
    tf.keras.layers.Embedding(len(vectorize_layer.get_vocabulary()), 64, mask_zero=True),\
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),\
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),\
    tf.keras.layers.Dense(64, activation='relu'),\
    tf.keras.layers.Dense(1)\
])
model.summary()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
`

**Output:**

![Screenshot-2025-02-26-111049](https://media.geeksforgeeks.org/wp-content/uploads/20250226111101887421/Screenshot-2025-02-26-111049.png)

Model Architecture

### Step 5: Model Training

Now we will train the model we defined in the previous step for five epochs.

Python`
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=test_dataset,
)
`

**Output:**

![Screenshot-2025-02-26-111302](https://media.geeksforgeeks.org/wp-content/uploads/20250226111407535360/Screenshot-2025-02-26-111302.webp)

Model Training

Plotting training vs validation accuracy and training vs validation loss.

Python`
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])
plt.show()
`

**Output:**

![download](https://media.geeksforgeeks.org/wp-content/uploads/20250226111540296186/download.png)

The plot of training and validation accuracy and loss

The model learns well on training data reaching 98% accuracy but struggles with validation data staying around 86%. The increasing validation loss shows overfitting meaning the model remembers training data but doesn’t generalize well. To fix this we can use Dropout, L2 regularization, early stopping or simplify the model to improve real-world performance.

Comment


More info

[Advertise with us](https://www.geeksforgeeks.org/about/contact-us/?listicles)

[Next Article](https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/)

[Gradient Descent With RMSProp from Scratch](https://www.geeksforgeeks.org/gradient-descent-with-rmsprop-from-scratch/)

![https://media.geeksforgeeks.org/auth/avatar.png](https://media.geeksforgeeks.org/wp-content/uploads/20200717172614/authPreLogo.png)

GeeksforGeeks

2

Improve

Article Tags :

- [AI-ML-DS](https://www.geeksforgeeks.org/category/ai-ml-ds/)
- [NLP](https://www.geeksforgeeks.org/category/ai-ml-ds/nlp/)
- [AI-ML-DS With Python](https://www.geeksforgeeks.org/tag/ai-ml-ds-python/)

### Similar Reads

[Bidirectional RNNs in NLP\\
\\
\\
The state of a recurrent network at a given time unit only knows about the inputs that have passed before it up to that point in the sentence; it is unaware of the states that will come after that. With knowledge of both past and future situations, the outcomes are significantly enhanced in some app\\
\\
10 min read](https://www.geeksforgeeks.org/bidirectional-rnns-in-nlp/)
[Emotion Detection using Bidirectional LSTM\\
\\
\\
Emotion Detection is one of the hottest topics in research nowadays. Emotion-sensing technology can facilitate communication between machines and humans. It will also help to improve the decision-making process. Many Machine Learning Models have been proposed to recognize emotions from the text. But\\
\\
10 min read](https://www.geeksforgeeks.org/emotion-detection-using-bidirectional-lstm/)
[Difference Between a Bidirectional LSTM and an LSTM\\
\\
\\
Long Short-Term Memory (LSTM) networks are capable of learning long-term dependencies. They were introduced to address the vanishing and exploding gradient problems that standard RNNs face. While LSTMs are powerful in many sequence-based tasks, a variation called Bidirectional LSTM (BiLSTM) enhances\\
\\
3 min read](https://www.geeksforgeeks.org/difference-between-a-bidirectional-lstm-and-an-lstm/)
[Bidirectional Recurrent Neural Network\\
\\
\\
Recurrent Neural Networks (RNNs) are type of neural networks designed to process sequential data such as speech, text and time series. Unlike feedforward neural networks that process input as fixed-length vectors RNNs can handle sequence data by maintaining a hidden state which stores information fr\\
\\
6 min read](https://www.geeksforgeeks.org/bidirectional-recurrent-neural-network/)
[Building Language Models in NLP\\
\\
\\
Building language models is a fundamental task in natural language processing (NLP) that involves creating computational models capable of predicting the next word in a sequence of words. These models are essential for various NLP applications, such as machine translation, speech recognition, and te\\
\\
4 min read](https://www.geeksforgeeks.org/building-language-models-in-nlp/)
[History and Evolution of NLP\\
\\
\\
As we know Natural language processing (NLP) is an exciting area that has grown at some stage in time, influencing the junction of linguistics, synthetic intelligence (AI), and computer technology knowledge. This article takes you on an in-depth journey through the history of NLP, diving into its co\\
\\
13 min read](https://www.geeksforgeeks.org/history-and-evolution-of-nlp/)
[LSTM Based Poetry Generation Using NLP in Python\\
\\
\\
One of the major tasks that one aims to accomplish in Conversational AI is Natural Language Generation (NLG) which refers to employing models for the generation of natural language. In this article, we will get our hands on NLG by building an LSTM-based poetry generator. Note: The readers of this ar\\
\\
7 min read](https://www.geeksforgeeks.org/lstm-based-poetry-generation-using-nlp-in-python/)
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
[NLP \| Extracting Named Entities\\
\\
\\
Recognizing named entity is a specific kind of chunk extraction that uses entity tags along with chunk tags. Common entity tags include PERSON, LOCATION and ORGANIZATION. POS tagged sentences are parsed into chunk trees with normal chunking but the trees labels can be entity tags in place of chunk p\\
\\
2 min read](https://www.geeksforgeeks.org/nlp-extracting-named-entities/)

Like2

We use cookies to ensure you have the best browsing experience on our website. By using our site, you
acknowledge that you have read and understood our
[Cookie Policy](https://www.geeksforgeeks.org/cookie-policy/) &
[Privacy Policy](https://www.geeksforgeeks.org/privacy-policy/)
Got It !


![Lightbox](https://www.geeksforgeeks.org/bidirectional-lstm-in-nlp/)

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

[iframe](https://td.doubleclick.net/td/ga/rul?tid=G-DWCCJLKX3X&gacid=633399669.1745057230&gtm=45je54g3h1v884918195za200&dma=0&gcd=13l3l3l3l1l1&npa=0&pscdl=noapi&aip=1&fledge=1&frm=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316~103116026&z=779569280)[iframe](https://td.doubleclick.net/td/rul/796001856?random=1745057230001&cv=11&fst=1745057230001&fmt=3&bg=ffffff&guid=ON&async=1&gtm=45be54h0h2v877914216za200zb858768136&gcd=13l3l3R3l5l1&dma=0&tag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&ptag_exp=102803279~102813109~102887800~102926062~103027016~103051953~103055465~103077950~103106314~103106316&u_w=1280&u_h=1024&url=https%3A%2F%2Fwww.geeksforgeeks.org%2Fbidirectional-lstm-in-nlp%2F&hn=www.googleadservices.com&frm=0&tiba=Bidirectional%20LSTM%20in%20NLP%20%7C%20GeeksforGeeks&npa=0&pscdl=noapi&auid=118090224.1745057230&uaa=x86&uab=64&uafvl=Google%2520Chrome%3B135.0.7049.95%7CNot-A.Brand%3B8.0.0.0%7CChromium%3B135.0.7049.95&uamb=0&uam=&uap=Linux%20x86_64&uapv=6.6.72&uaw=0&fledge=1&data=event%3Dgtag.config)

[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](about:blank)[iframe](https://securepubads.g.doubleclick.net/static/topics/topics_frame.html)