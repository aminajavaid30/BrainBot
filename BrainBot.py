import streamlit as st
import requests
import tempfile
import validators
import os
from utils import setup_logging, log_error

# Custom CSS
with open('styles.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Setup Logging
setup_logging()

## FUNCTIONS
## -------------------------------------------------------------------------------------------
# Function to save the uploaded file as a temporary file and return its path.
def save_uploaded_file(uploaded_file):
    file_content = uploaded_file.read()  # Load the document
    
    # Create a directory if it doesn't exist
    data_dir = "/data"
    # os.makedirs(data_dir, exist_ok=True)
    
    # Create a temporary file in the data directory
    with tempfile.NamedTemporaryFile(delete=False, dir=data_dir) as temp_file:
        temp_file.write(file_content)  # Write the uploaded file content to the temporary file
        temp_file_path = temp_file.name  # Get the path of the temporary file
        return temp_file_path

# Function to save the uploaded image as a temporary file  and return its path.
def save_uploaded_image(uploaded_image):
    # Create a directory named "images" if it doesn't exist
    images_dir = "/images"
    # os.makedirs(images_dir, exist_ok=True)
    
    # Create a temporary file path within the "images" directory with .png extension
    temp_file_path = os.path.join(images_dir, tempfile.NamedTemporaryFile(suffix=".png").name)
    
    # Write the uploaded image content to the temporary file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_image.read())
    return temp_file_path

## LOGO and TITLE
## -------------------------------------------------------------------------------------------
# Show the logo and title side by side
col1, col2 = st.columns([1, 4])
with col1:
    st.image("brainbot.png", use_column_width=True,)
with col2:
    st.title("Hi, I am BrainBot - Your AI Learning Assistant!")

# Main content
st.header("Upload any 📄 file, 🖼️ image, or 🔗 webpage link and ask me anything from it!")
st.subheader("Supported file formats: PDF, DOCX, TXT, PPTX, HTML")
st.subheader("Supported image formats: PNG, JPG, JPEG")

col3, col4 = st.columns([2, 3])
with col3:
    ## LLM OPTIONS
    # Select the LLM to use (either GPT-4 or GROQ)
    llm = st.radio(
        "Choose the LLM", ["GPT-4", "GROQ"],
        index=1
    )

    st.session_state["llm"] = llm

    ## CHAT OPTIONS - FILE, IMAGE, WEBSITE
    ## -------------------------------------------------------------------------------------------
    # User Inputs
    uploaded_file = None
    uploaded_image = None
    website_link = None
    question = None

    if llm == "GPT-4" and "api_key_flag" not in st.session_state:
        st.warning("Please enter your OpenAI API key.")
        # Get OpenAI API Key from user
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        # Send POST request to a FastAPI endpoint to set the OpenAI API key as an environment 
        # variable
        with st.spinner("Activating OpenAI API..."):
            try:
                FASTAPI_URL = "http://localhost:8000/set_api_key"
                data = {"api_key": openai_api_key}
                if openai_api_key:
                    response = requests.post(FASTAPI_URL, json=data)
                    st.sidebar.success(response.text)
                    st.session_state['api_key_flag'] = True
                    st.experimental_rerun()
            except Exception as e:
                log_error(str(e))
                st.switch_page("error.py")
with col4:
    if llm == "GROQ" or "api_key_flag" in st.session_state:
        # Select to upload file, image, or link to chat with them
        upload_option = st.radio(
            "Select an option", ["📄 Upload File", "🖼️ Upload Image", "🔗 Upload Link"]
        )
        # Select an option to show the appropriate file_uploader
        if upload_option == "📄 Upload File":
            uploaded_file = st.file_uploader("Choose a file", 
                                                    type=["txt", "pdf", "docx", "pptx", "html"])
        elif upload_option == "🖼️ Upload Image":
            uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        elif upload_option == "🔗 Upload Link":
            website_link = st.text_input("Enter a website URL")

## CHAT HISTORY
## -------------------------------------------------------------------------------------------
# Initialize an empty list to store chat messages with files
if 'file_chat_history' not in st.session_state:
    st.session_state['file_chat_history'] = []
# Initialize an empty list to store image interpretations
if 'image_chat_history' not in st.session_state:
    st.session_state['image_chat_history'] = [] 
# Initialize an empty list to store chat messages with websites
if 'web_chat_history' not in st.session_state:
    st.session_state['web_chat_history'] = []

## FILE
## -------------------------------------------------------------------------------------------
# Load the uploaded file, then save it into a vector store, and enable the input field to ask 
# a question
st.session_state['uploaded_file'] = False
if uploaded_file is not None: 
    with st.spinner("Loading file..."):
        # Save the uploaded file to a temporary path
        temp_file_path = save_uploaded_file(uploaded_file)
            
        try:    
            # Send POST request to a FastAPI endpoint to load the file into a vectorstore
            data = {"file_path": temp_file_path, "file_type": uploaded_file.type}
            FASTAPI_URL = f"http://localhost:8000/load_file/{llm}"
            response = requests.post(FASTAPI_URL, json=data)
            st.success(response.text)
            st.session_state['current_file'] = uploaded_file.name
            st.session_state['uploaded_file'] = True
            st.switch_page("pages/File-chat.py")
        except Exception as e:
            log_error(str(e))
            st.switch_page("error.py")

## IMAGE
## -------------------------------------------------------------------------------------------
# Load the uploaded image if user uploads an image, then interpret the image
st.session_state['uploaded_image'] = False
if uploaded_image is not None:
    try:
        # Save uploaded image to a temporary file
        temp_img_path = save_uploaded_image(uploaded_image)
    except Exception as e:
                log_error(str(e))
                st.switch_page("error.py")

    st.session_state['temp_img_path'] = temp_img_path
    st.session_state['current_image'] = uploaded_image.name
    st.session_state['uploaded_image'] = True
    st.switch_page("pages/Image-scan.py")

## WEBSITE LINK
## -------------------------------------------------------------------------------------------
# Load the website content, then save it into a vector store, and enable the input field to 
# ask a question
st.session_state['uploaded_link'] = False
if website_link is not None:
    if website_link:
        # Ensure that the user has entered a correct URL
        if validators.url(website_link):
            try:
                # Send POST request to a FastAPI endpoint to scrape the webpage and load its text 
                # into a vector store
                FASTAPI_URL = f"http://localhost:8000/load_link/{llm}"
                data = {"website_link": website_link}
                with st.spinner("Loading website..."):
                    response = requests.post(FASTAPI_URL, json=data)
                    st.success(response.text)
                    st.session_state['current_website'] = website_link
                    st.session_state['uploaded_link'] = True
                    st.switch_page("pages/Web-chat.py")
            except Exception as e:
                log_error(str(e))
                st.switch_page("error.py")
        else:
            st.error("Invalid URL. Please enter a valid URL.")