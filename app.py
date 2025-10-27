from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
import pdfplumber
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
import asyncio
import re
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Set up logging for debugging
logging.basicConfig(filename='app.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # Chat model, e.g., gpt-4o-mini
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")  # Embedding model

# Validate credentials
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_EMBEDDING_DEPLOYMENT]):
    st.error("Missing Azure OpenAI credentials or embedding deployment. Please set environment variables.")
    logger.error("Missing Azure OpenAI credentials or embedding deployment.")
    st.stop()

# Log credentials for debugging
logger.info(f"Endpoint: {AZURE_OPENAI_ENDPOINT}, API Key: {AZURE_OPENAI_API_KEY[:4]}..., Chat Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}, Embedding Deployment: {AZURE_EMBEDDING_DEPLOYMENT}")

# Initialize Semantic Kernel
kernel = sk.Kernel()

# Add Azure OpenAI chat service
try:
    chat_service = AzureChatCompletion(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        service_id="chat_service"
    )
    kernel.add_service(chat_service)
    logger.info("Chat service added successfully.")
except Exception as e:
    st.error(f"Failed to add chat service: {e}")
    logger.error(f"Chat service error: {e}")
    st.stop()

# Add embedding service
try:
    embedding_gen = AzureTextEmbedding(
        deployment_name=AZURE_EMBEDDING_DEPLOYMENT,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        service_id="embedding_service"
    )
    kernel.add_service(embedding_gen)
    logger.info("Embedding service added successfully.")
except Exception as e:
    st.error(f"Failed to add embedding service: {e}")
    logger.error(f"Embedding service error: {e}")
    st.stop()

# Extract FAQs from PDF
def extract_faqs_from_pdf(pdf_path):
    faqs = []
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning("Empty text extracted from a PDF page.")

        if not text.strip():
            raise ValueError("No text extracted from PDF. Is it image-based?")

        # Log extracted text for debugging
        logger.info(f"Extracted PDF text (first 500 chars): {text[:500]}...")

        # Parse FAQs with numbered format (e.g., "1. Q: Question" or "Q: Question")
        lines = text.split('\n')
        current_q = None
        current_a = []
        for line in lines:
            line = line.strip()
            # Match lines starting with optional number (e.g., "1. Q:", "2. Q:")
            question_match = re.match(r'^(?:\d+\.\s*)?Q:\s*(.+)', line)
            if question_match:
                if current_q:
                    faqs.append({'question': current_q, 'answer': ' '.join(current_a)})
                current_q = question_match.group(1).strip()
                current_a = []
            elif line.startswith("A:"):
                current_a.append(line[2:].strip())
            elif current_a:
                current_a.append(line)

        if current_q:
            faqs.append({'question': current_q, 'answer': ' '.join(current_a)})

        if not faqs:
            raise ValueError("No FAQs extracted from PDF. Check PDF format.")

        # Log extracted FAQs for debugging
        for i, faq in enumerate(faqs):
            logger.info(f"FAQ {i+1}: Q: {faq['question'][:50]}... A: {faq['answer'][:50]}...")

        logger.info(f"Extracted {len(faqs)} FAQs from PDF.")
        return faqs[:20]  # Limit to 20 FAQs
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        logger.error(f"PDF extraction error: {e}")
        return []

# Load FAQs and generate embeddings
if 'faqs' not in st.session_state:
    st.session_state.faqs = extract_faqs_from_pdf('faqs.pdf')
    st.session_state.faq_embeddings = []

    if not st.session_state.faqs:
        st.error("No FAQs extracted from PDF. Please check the file and its format.")
        logger.error("No FAQs extracted from PDF.")
    else:
        # Generate embeddings for FAQs
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2023-05-15"
            )
            for faq in st.session_state.faqs:
                response = client.embeddings.create(
                    input=faq['question'],
                    model=AZURE_EMBEDDING_DEPLOYMENT
                )
                embedding = response.data[0].embedding
                st.session_state.faq_embeddings.append(embedding)
                logger.info(f"Generated embedding for FAQ: {faq['question'][:50]}...")
            logger.info(f"Generated {len(st.session_state.faq_embeddings)} FAQ embeddings successfully.")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            logger.error(f"Embedding generation error: {e}")

# Find most similar FAQ
async def get_most_similar_faq(query):
    try:
        if not st.session_state.faq_embeddings:
            logger.error("No FAQ embeddings available.")
            return None

        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2023-05-15"
        )
        response = client.embeddings.create(
            input=query,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

        embeddings = np.array(st.session_state.faq_embeddings)
        similarities = cosine_similarity(query_embedding, embeddings)
        best_index = np.argmax(similarities)

        if similarities[0][best_index] > 0.7:
            logger.info(f"Matched FAQ with similarity {similarities[0][best_index]}")
            return st.session_state.faqs[best_index]['answer']
        return None
    except Exception as e:
        st.error(f"Error in similarity search: {e}")
        logger.error(f"Similarity search error: {e}")
        return None

# Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    similar_answer = asyncio.run(get_most_similar_faq(prompt))

    if similar_answer:
        response = similar_answer
    else:
        # Fallback to Semantic Kernel
        try:
            chat_function = kernel.create_function_from_prompt(
                prompt=f"Answer based on clinic FAQs if possible, else say 'Not found in clinic FAQs': {prompt}"
            )
            result = asyncio.run(kernel.invoke(chat_function))
            response = str(result)
            logger.info("Fallback to Semantic Kernel chat.")
        except Exception as e:
            response = f"Error processing request: {e}"
            logger.error(f"Chat fallback error: {e}")

    return response

st.set_page_config(page_title="Clinic FAQ ChatBot - An LLM-powered Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('🏥💬 Clinic FAQ ChatBot')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot for clinic FAQs built using:
    - [Streamlit](https://streamlit.io/)
    - [Azure OpenAI](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)
    
    💡 Note: Requires Azure OpenAI credentials!
    ''')
    add_vertical_space(5)
    st.write('Made  by Team 4 (Capella Capstone Project 2025)')

# Generate empty lists for generated and past.
# generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! I'm your Clinic FAQ Assistant. How may I help you today?"]
# past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
# Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input", placeholder="Ask about clinic FAQs...")
    return input_text

# Applying the user input box
with input_container:
    user_input = get_text()

# Response output
# Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)


    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))