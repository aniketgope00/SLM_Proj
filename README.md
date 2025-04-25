## Contents:
1. [RAG Chat App](#Chat-with-Multiple-PDFs)
2. [TinyGPT](#Train-Your-Own-TinyGPT-Model)


## Chat with Multiple PDFs
<a name="Chat-with-Multiple-PDFs"></a>

This project is a Streamlit-based application that empowers users to upload multiple PDF documents and engage with their content through an intuitive conversational interface. By harnessing the power of Natural Language Processing (NLP) and Machine Learning, the application intelligently extracts text, images, tables, and equations from your PDFs, allowing you to ask questions and instantly retrieve pertinent information.

## Key Features

* **PDF Upload and Processing:**
    * Effortlessly upload multiple PDF files via the convenient sidebar.
    * Intelligently extracts text content from uploaded PDFs using robust libraries like PyPDF2 and PyMuPDF.
    * Goes beyond basic text extraction by also capturing images and tables, leveraging the capabilities of PyMuPDF and unstructured.
    * Detects and extracts mathematical equations embedded within the PDF documents.

* **Conversational Interface:**
    * Engage in natural language conversations about the content of your uploaded PDFs.
    * Powered by a sophisticated Conversational Retrieval Chain, built with LangChain and Google Generative AI, ensuring accurate and contextually relevant responses.

* **Media Extraction:**
    * Automatically extracts all images and tables present in your PDF documents.
    * Provides crucial context for each extracted media element, including surrounding text snippets and the corresponding page number.

* **Equation Detection:**
    * Smartly identifies mathematical equations within the PDF content.
    * Renders detected equations beautifully using LaTeX formatting for clear presentation.

* **Vector Search:**
    * Creates an efficient vector store using cutting-edge Hugging Face Embeddings.
    * Utilizes FAISS (Facebook AI Similarity Search) for lightning-fast retrieval of relevant text chunks based on your queries.

* **Logging and Debugging:**
    * Maintains a detailed log of application activity in `app_current.log` for effective debugging and monitoring.
    * Provides a real-time view of the application logs directly in the sidebar with auto-refresh functionality, keeping you informed.

* **Customizable Context Filtering:**
    * Offers the flexibility to filter extracted media (images and tables) based on user-defined keywords or contextual terms, allowing you to focus on specific information.

* **Device Compatibility:**
    * Intelligently detects and seamlessly utilizes available GPU (CUDA) resources for accelerated processing, ensuring a smoother and faster user experience.

## Technologies Used

* **Streamlit:** The framework of choice for building the interactive and user-friendly web application.
* **LangChain:** The powerful library enabling conversational AI and sophisticated retrieval-based question answering workflows.
* **Hugging Face Transformers:** Providing state-of-the-art embeddings and a wide range of NLP capabilities.
* **FAISS (Facebook AI Similarity Search):** Ensuring efficient and rapid vector-based similarity search for relevant document snippets.
* **PyPDF2:** A fundamental library for extracting text content from PDF files.
* **PyMuPDF:** A versatile library offering advanced PDF parsing and comprehensive media extraction capabilities.
* **Unstructured:** Employed for advanced PDF parsing, handling complex layouts, and improving the formatting of extracted content.
* **Google Generative AI:** Powering the conversational language model for intelligent and context-aware responses.
* **Logging:** The standard Python library used for meticulous monitoring and debugging of application behavior.

## How It Works

1.  **Upload PDFs:** Begin by uploading one or more PDF files using the intuitive file uploader located in the sidebar.
2.  **Process Documents:** The application automatically processes the uploaded PDFs, extracting text, images, tables, and identifying equations. The text is then split into manageable chunks.
3.  **Create Vector Store:** Embeddings are generated for each text chunk using Hugging Face Transformers, and these embeddings are stored in a FAISS index for efficient similarity searching.
4.  **Ask Questions:** Pose your questions related to the content of the PDFs in the chat interface. The application leverages the Conversational Retrieval Chain to fetch relevant text chunks from the vector store.
5.  **Generate Answers:** The retrieved text chunks, along with the conversation history, are fed to the Google Generative AI model to generate accurate and contextually appropriate answers. The source document(s) for the answer are also provided.
6.  **View Media:** Navigate to the media viewing section to see all the extracted images and tables. Utilize the context filtering feature to narrow down the displayed media based on specific keywords or context.

## Example Use Cases

* **Research:** Effortlessly extract and query critical information from a multitude of academic papers, streamlining your research process.
* **Business:** Efficiently analyze and retrieve key data points from business reports, contracts, and other crucial documents.
* **Education:** Enhance your learning experience by interactively engaging with textbooks, lecture notes, and study materials.


## Train Your Own TinyGPT Model
<a name="Train-Your-Own-TinyGPt-Model"></a>

This project is a Streamlit-based application that empowers users to train their very own lightweight GPT-like language model using custom text data. The application provides an intuitive and interactive interface for every step of the process, from uploading your training data and configuring model parameters to generating text with your newly trained model.

## Key Features

* **Custom Model Training:**
    * Effortlessly upload your training data in either markdown or plain text file formats.
    * Trains a simplified GPT-like language model leveraging a transformer-based architecture.
    * Offers a range of configurable hyperparameters to tailor the model to your specific needs.

* **Interactive Hyperparameter Tuning:**
    * Fine-tune crucial hyperparameters through an easy-to-use graphical interface.
    * Adjust settings such as embedding dimensions, the number of attention heads, the number of layers, the learning rate, and the dropout rate to optimize model performance.

* **Text Generation:**
    * Unleash the power of your trained model by providing custom prompts and generating novel text.
    * Control the characteristics of the generated text using adjustable settings for maximum tokens and temperature (influencing randomness).

* **Model Persistence:**
    * Save your trained models, along with their associated vocabulary and hyperparameter configurations, for convenient future use.
    * Easily select and load previously trained models directly from the application sidebar.

* **Batch Data Processing:**
    * Efficiently handles large text datasets by intelligently splitting them into distinct training and validation sets.
    * Supports batching during the training process with configurable block sizes to optimize resource utilization.

* **Transformer-Based Architecture:**
    * Implements a streamlined GPT-like architecture incorporating essential components such as multi-head attention mechanisms, feedforward layers, and positional embeddings.
    * Includes a sophisticated token generation mechanism enabling autoregressive text generation capabilities.

* **Real-Time Training Feedback:**
    * Stay informed about the training progress with dynamic, real-time updates on both training and validation losses.
    * Visualize the training trajectory with a live loss chart and monitor progress with an intuitive progress bar.

* **Error Handling and Debugging:**
    * Features robust error handling mechanisms to gracefully manage potential issues during data preparation, model training, and text generation phases.
    * Provides detailed debug logs during text generation to aid in troubleshooting and understanding model behavior.

## Technologies Used

* **Streamlit:** The foundation for building the interactive and user-friendly web application interface.
* **PyTorch:** The powerful deep learning framework used to implement the transformer-based GPT model and the entire training pipeline.
* **Torch.nn:** Provides the essential building blocks for defining the model's architecture, including attention mechanisms and feedforward layers.
* **Torch.optim:** Offers various optimization algorithms to efficiently train the model.
* **JSON and Pickle:** Utilized for reliably saving and loading model states, hyperparameter configurations, and vocabulary.

## How It Works

1.  **Upload Training Data:** Begin by uploading one or more markdown or plain text files through the convenient sidebar file uploader. The application will automatically combine the content of these files to create your training dataset.
2.  **Configure Model Parameters:** Navigate to the configuration section to adjust the model's hyperparameters. You can customize settings such as the embedding size, the number of layers in the transformer, and the learning rate. Sensible default parameters are provided for a quick start.
3.  **Train the Model:** Initiate the training process. The application will split your uploaded data into training and validation sets. Real-time feedback on the training progress, including loss metrics and a progress bar, will be displayed.
4.  **Save and Load Models:** Once training is complete, you can save your trained model along with a timestamp for easy identification and future use. Previously saved models can be effortlessly loaded from the sidebar.
5.  **Generate Text:** Unleash the creative potential of your trained model by providing a starting prompt in the text generation interface. Experiment with the generation settings, such as temperature and the maximum number of tokens, to fine-tune the output.

## Example Use Cases

* **Custom Chatbots:** Train a lightweight chatbot tailored to specific domains or datasets.
* **Text Completion:** Develop models capable of generating creative writing prompts, completing sentences, or assisting with content creation.
* **Educational Tools:** Build small-scale language models for educational purposes, allowing for experimentation and a deeper understanding of language modeling concepts.



## Use these Apps:

1. Activate venv as: ```slm_env/Scripts/Activate.ps1``` (for Powershell)
2. Run app as ```streamlit run <app_name>.py ```
