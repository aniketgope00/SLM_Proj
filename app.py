import os
import asyncio
import torch
import streamlit as st
import firecrawl
from PyPDF2 import PdfReader
import logging
from datetime import datetime
import atexit
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.text import partition_text
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.embeddings import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from streamlit_navigation_bar import st_navbar
from html_templates import css, bot_template, user_template

from pdf2image import convert_from_bytes
import fitz  # PyMuPDF
import base64
import tempfile
import re
import io
from PIL import Image

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
LOG_FILE = "app_current.log"

def setup_logging():
    # Clear existing log file if it exists
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE)
        ]
    )
    return logging.getLogger(__name__)

def cleanup_logs():
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception as e:
        print(f"Error cleaning up log file: {e}")

# Register cleanup function
atexit.register(cleanup_logs)

# Initialize logger
logger = setup_logging()
logger.info(f"Using device: {device}")

# Initialize session state for logs
if 'log_buffer' not in st.session_state:
    st.session_state.log_buffer = []

def read_logs():
    """Read logs from file and update session state buffer"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = f.read()
                return logs
        return "No logs available yet."
    except Exception as e:
        return f"Error reading logs: {str(e)}"

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

def get_pdf_text(docs):
    logger.info("Starting PDF text extraction")
    text = ""
    all_equations = []
    equation_pattern = r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))'
    
    for document in docs:
        logger.info(f"Processing document: {document.name}")
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(document.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Use unstructured to extract text and maintain formatting
            elements = partition_pdf(tmp_path, include_metadata=True)
            
            # Process each element
            for element in elements:
                element_text = str(element)
                
                # Extract equations
                equations = re.finditer(equation_pattern, element_text, re.DOTALL)
                for eq in equations:
                    eq_text = eq.group(0)
                    # Clean up the equation
                    eq_text = eq_text.strip('$').strip()
                    if eq_text not in all_equations:
                        all_equations.append(eq_text)
                
                text += element_text + "\n"
                
            logger.info(f"Successfully processed document using unstructured")
        except Exception as e:
            logger.error(f"Error processing document with unstructured: {str(e)}")
            # Fallback to PyPDF2
            reader = PdfReader(document)
            for page_num, page in enumerate(reader.pages):
                text += page.extract_text() + "\n"
            logger.info("Used PyPDF2 as fallback")
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    logger.info(f"Found {len(all_equations)} unique equations")
    st.session_state.detected_equations = all_equations
    return text

def extract_images_from_pdf(pdf_bytes, document_name):
    """Direct image extraction function using both PyMuPDF and pdf2image"""
    logger.info(f"Extracting images from {document_name}")
    images = []
    
    try:
        # Method 1: Use PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"PDF has {len(pdf_document)} pages")
        
        for page_num, page in enumerate(pdf_document):
            logger.info(f"Processing page {page_num+1}")
            
            # Get images from this page
            image_list = page.get_images(full=True)
            logger.info(f"Found {len(image_list)} images on page {page_num+1}")
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        try:
                            # Create a PIL Image from bytes for verification
                            pil_img = Image.open(io.BytesIO(image_bytes))
                            width, height = pil_img.size
                            
                            # Skip tiny images that might be icons or artifacts
                            if width < 50 or height < 50:
                                logger.info(f"Skipping small image ({width}x{height}) on page {page_num+1}")
                                continue
                                
                            logger.info(f"Successfully extracted image {img_index} from page {page_num+1}, size: {width}x{height}, format: {image_ext}")
                            
                            # Encode as base64
                            img_b64 = base64.b64encode(image_bytes).decode('utf-8')
                            img_html = f'<img src="data:image/{image_ext};base64,{img_b64}" style="max-width: 100%; margin: 10px 0;">'
                            
                            # Get text near the image
                            rect = page.get_image_bbox(xref)
                            margin = 100  # Larger margin to capture more context
                            expanded_rect = fitz.Rect(
                                rect.x0 - margin,
                                rect.y0 - margin,
                                rect.x1 + margin,
                                rect.y1 + margin
                            )
                            # Keep within page bounds
                            expanded_rect.intersect(page.rect)
                            nearby_text = page.get_text("text", clip=expanded_rect)
                            
                            images.append({
                                'type': 'image',
                                'content': img_html,
                                'context': nearby_text[:200] + '...' if len(nearby_text) > 200 else nearby_text,
                                'page': page_num + 1,
                                'width': width,
                                'height': height,
                                'format': image_ext
                            })
                            
                        except Exception as img_err:
                            logger.error(f"Error processing image: {str(img_err)}")
                            continue
                except Exception as e:
                    logger.error(f"Error extracting image {img_index} on page {page_num+1}: {str(e)}")
                    continue
        
        pdf_document.close()
        
        # If no valid images were found using PyMuPDF, try pdf2image as fallback
        if not images:
            logger.info("No images found with PyMuPDF, trying pdf2image")
            try:
                # Convert PDF pages to PIL images
                pil_images = convert_from_bytes(pdf_bytes)
                
                for i, img in enumerate(pil_images):
                    # Save image to bytes buffer
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # Encode as base64
                    img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    img_html = f'<img src="data:image/png;base64,{img_b64}" style="max-width: 100%; margin: 10px 0;">'
                    
                    images.append({
                        'type': 'image',
                        'content': img_html,
                        'context': f"Full page {i+1}",
                        'page': i + 1,
                        'width': img.width,
                        'height': img.height,
                        'format': 'png'
                    })
                    logger.info(f"Extracted full page as image: page {i+1}")
                    
            except Exception as pdf2img_err:
                logger.error(f"Error using pdf2image: {str(pdf2img_err)}")
    
    except Exception as e:
        logger.error(f"Error in extract_images_from_pdf: {str(e)}")
    
    logger.info(f"Total images extracted: {len(images)}")
    return images

def extract_tables_from_pdf(pdf_bytes, document_name):
    """Direct table extraction function using PyMuPDF"""
    logger.info(f"Extracting tables from {document_name}")
    tables = []
    
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for page_num, page in enumerate(pdf_document):
            try:
                # Try to find tables using PyMuPDF's table finder
                tab = page.find_tables()
                if tab and hasattr(tab, 'tables') and len(tab.tables) > 0:
                    logger.info(f"Found {len(tab.tables)} tables on page {page_num+1}")
                    
                    for table_idx, table in enumerate(tab.tables):
                        try:
                            # Convert to HTML
                            rows = []
                            for row_idx in range(table.rows):
                                row_cells = []
                                for col_idx in range(table.cols):
                                    cell = table.cells[row_idx][col_idx]
                                    if cell:
                                        text = page.get_text("text", clip=cell.rect)
                                        row_cells.append(text)
                                    else:
                                        row_cells.append("")
                                rows.append(row_cells)
                            
                            # Generate HTML table
                            html = '<table border="1" style="border-collapse: collapse; width: 100%;">'
                            for row in rows:
                                html += '<tr>'
                                for cell in row:
                                    html += f'<td style="padding: 5px;">{cell}</td>'
                                html += '</tr>'
                            html += '</table>'
                            
                            # Get surrounding text for context
                            rect = table.rect
                            margin = 50
                            expanded_rect = fitz.Rect(
                                rect.x0 - margin,
                                rect.y0 - margin,
                                rect.x1 + margin,
                                rect.y1 + margin
                            )
                            # Keep within page bounds
                            expanded_rect.intersect(page.rect)
                            surrounding_text = page.get_text("text", clip=expanded_rect)
                            
                            tables.append({
                                'type': 'table',
                                'content': html,
                                'context': surrounding_text[:200] + '...' if len(surrounding_text) > 200 else surrounding_text,
                                'page': page_num + 1
                            })
                            logger.info(f"Successfully extracted table {table_idx} from page {page_num+1}")
                        except Exception as table_err:
                            logger.error(f"Error processing table {table_idx} on page {page_num+1}: {str(table_err)}")
                            continue
            except Exception as e:
                logger.error(f"Error finding tables on page {page_num+1}: {str(e)}")
                continue
                
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error in extract_tables_from_pdf: {str(e)}")
    
    logger.info(f"Total tables extracted: {len(tables)}")
    return tables

def extract_images_and_tables(docs, context=None):
    """Main extraction function that processes documents and applies context filtering"""
    logger.info("Starting image and table extraction")
    all_media = []
    
    if not docs:
        logger.warning("No documents provided for extraction")
        return all_media
    
    for doc in docs:
        try:
            # Reset file position to beginning
            doc.seek(0)
            doc_bytes = doc.getvalue()
            
            # Extract images
            images = extract_images_from_pdf(doc_bytes, doc.name)
            
            # Extract tables
            tables = extract_tables_from_pdf(doc_bytes, doc.name)
            
            # Combine media
            all_media.extend(images + tables)
            
        except Exception as e:
            logger.error(f"Error processing document {doc.name}: {str(e)}")
    
    # Filter by context if provided
    if context and all_media:
        logger.info(f"Filtering media by context: {context}")
        context_terms = context.lower().split()
        
        filtered_media = []
        for media in all_media:
            # Check if any context term appears in the media context
            media_context = media['context'].lower()
            if any(term in media_context for term in context_terms):
                filtered_media.append(media)
        
        logger.info(f"Filtered from {len(all_media)} to {len(filtered_media)} media items based on context")
        return filtered_media
    
    return all_media

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_vectorstore(text_chunks):
    try:
        logger.info("Initializing embeddings model")
        model_kwargs = {'device': device}
        encode_kwargs = {'device': device, 'batch_size': 32}
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Creating vector store")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        logger.error(f"Error in get_vectorstore: {str(e)}")
        # Fallback to CPU if there are device issues
        logger.info("Falling back to CPU embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 16}
        )
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vectorstore

def get_conversation_chain(vectorstore):
    logger.info("Creating conversation chain")
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            convert_system_message_to_human=True
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3, "fetch_k": 5}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )
        logger.info("Conversation chain created successfully")
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        raise

def handle_userinput(user_question):
    logger.info(f"Processing user question: {user_question}")
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history with sources
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            
            # Display sources and similarity scores
            if 'source_documents' in response:
                st.markdown("""
                <style>
                .source-container {
                    background-color: #2e3440;
                    color: #d8dee9;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                    border: 1px solid #434c5e;
                }
                .source-header {
                    color: #88c0d0;
                    margin-bottom: 5px;
                    font-weight: bold;
                }
                .source-text {
                    color: #e5e9f0;
                    font-size: 0.95em;
                }
                .similarity-score {
                    color: #a3be8c;
                    font-size: 0.9em;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown("### Source Documents", unsafe_allow_html=True)
                for idx, doc in enumerate(response['source_documents']):
                    if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                        similarity = f"{doc.metadata['score']:.2%}"
                    else:
                        similarity = "N/A"
                    
                    source_text = doc.page_content[:200] + "..."  # Show first 200 chars
                    st.markdown(f"""
                    <div class="source-container">
                        <div class="source-header">Source {idx + 1}</div>
                        <div class="similarity-score">Similarity: {similarity}</div>
                        <div class="source-text">{source_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Display relevant media content
    if st.session_state.pdf_docs:
        with st.spinner("Extracting images and tables..."):
            # Extract and cache media if not already done
            if 'all_media' not in st.session_state:
                logger.info("Extracting all media from documents")
                st.session_state.all_media = extract_images_and_tables(st.session_state.pdf_docs)
            
            # Extract keywords from question for context filtering
            keywords = [word.lower() for word in user_question.split() if len(word) > 3]
            context_string = ' '.join(keywords) if keywords else None
            
            if context_string:
                # Filter media by context
                relevant_media = []
                for media in st.session_state.all_media:
                    media_context = media['context'].lower()
                    if any(keyword in media_context for keyword in keywords):
                        relevant_media.append(media)
                logger.info(f"Found {len(relevant_media)} media items relevant to the query")
            else:
                # If no context, show all media
                relevant_media = st.session_state.all_media
                logger.info(f"Displaying all {len(relevant_media)} media items")
        
        # Display media content
        if relevant_media:
            # Separate images and tables
            images = [m for m in relevant_media if m['type'] == 'image']
            tables = [m for m in relevant_media if m['type'] == 'table']
            
            # Display images
            if images:
                st.markdown("### Images")
                # Display images in a grid (2 per row)
                cols_per_row = 2
                for i in range(0, len(images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(images):
                            with cols[j]:
                                st.markdown(f"**Image from page {images[idx]['page']}**")
                                # Display the image using HTML
                                st.markdown(images[idx]['content'], unsafe_allow_html=True)
                                # Show context in an expander
                                with st.expander("Context", expanded=False):
                                    st.markdown(images[idx]['context'])
            
            # Display tables
            if tables:
                st.markdown("### Tables")
                for table in tables:
                    st.markdown(f"**Table from page {table['page']}**")
                    st.markdown(table['content'], unsafe_allow_html=True)
                    with st.expander("Context", expanded=False):
                        st.markdown(table['context'])
        else:
            st.info("No relevant images or tables found for this query.")

    # Display equations if relevant
    if "detected_equations" in st.session_state and st.session_state.detected_equations and any(eq_term in user_question.lower() for eq_term in ['equation', 'formula', 'math']):
        st.markdown("### Related Equations")
        for eq in st.session_state.detected_equations:
            st.latex(eq)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "media_content" not in st.session_state:
        st.session_state.media_content = []
    if "detected_equations" not in st.session_state:
        st.session_state.detected_equations = []
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "all_media" not in st.session_state:
        st.session_state.all_media = []

    st.header("Chat with multiple PDFs :books:")
    
    # Add logging display in sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process"):
                with st.spinner("Processing"):
                    logger.info("Starting document processing")
                    # Store PDF docs in session state for later use
                    st.session_state.pdf_docs = pdf_docs
                    
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    logger.info(f"Extracted {len(raw_text)} characters of text")

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    logger.info(f"Created {len(text_chunks)} text chunks")

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    logger.info("Created vector store")

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    # Pre-extract all media
                    logger.info("Pre-extracting all media")
                    st.session_state.all_media = extract_images_and_tables(pdf_docs)
                    logger.info(f"Extracted {len(st.session_state.all_media)} media items in total")
                    
                    logger.info("Setup complete")
                    st.success("Documents processed successfully!")
        
        with col2:
            if st.session_state.pdf_docs and st.button("View All Images"):
                st.session_state.show_all_images = True
        
        # Display logs with auto-refresh
        st.subheader("Application Logs")
        logs_placeholder = st.empty()
        
        # Update logs every 1 second
        def update_logs():
            current_logs = read_logs()
            logs_placeholder.code(current_logs)
        
        # Schedule log updates
        if st.session_state.get('refresh_logs', True):
            update_logs()
    
    # Display all images if button was clicked
    if st.session_state.get('show_all_images', False) and 'all_media' in st.session_state:
        st.subheader("All Extracted Images")
        images = [m for m in st.session_state.all_media if m['type'] == 'image']
        
        if images:
            cols_per_row = 2
            for i in range(0, len(images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(images):
                        with cols[j]:
                            st.markdown(f"**Image from page {images[idx]['page']}**")
                            st.markdown(images[idx]['content'], unsafe_allow_html=True)
        else:
            st.info("No images found in the documents.")
        
        # Add button to return to chat
        if st.button("Return to Chat"):
            st.session_state.show_all_images = False
            st.experimental_rerun()

    # Only show chat interface if not showing all images
    if not st.session_state.get('show_all_images', False):
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_logs()