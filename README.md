# Chat with Multiple PDFs

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

This project showcases the transformative potential of integrating NLP, machine learning, and interactive web applications to deliver a seamless and powerful document analysis experience.
