import streamlit as st
import requests
import utils
from utils import setup_logging, log_error

# Custom CSS
with open('styles.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Setup Logging
setup_logging()

## LOGO and TITLE
## -------------------------------------------------------------------------------------------
# Show the logo and title side by side
col1, col2 = st.columns([1, 4])
with col1:
    st.image("brainbot.png", width=100)
with col2:
    st.title("Image-Scan")

llm = st.session_state["llm"]

if "current_image" in st.session_state:
    current_image = st.session_state['current_image']
    if st.sidebar.button("Upload New Image"):
        st.switch_page("BrainBot.py")
    st.subheader("Your image has been uploaded successfully.")
    st.success(current_image)
else:
    st.warning("Upload an image to interpret it.")
    if st.button("Upload Image"):
        st.switch_page("BrainBot.py")

## CHAT
# Clear the image chat history if user has uploaded a new image
if st.session_state['uploaded_image'] == True:
    st.session_state['image_chat_history'] = []

# Display the image chat history
for image in st.session_state['image_chat_history']:
    with st.chat_message("user"):
        st.image(image["path"], caption=current_image)
    with st.chat_message("ai"):
        st.markdown(utils.format_response(image["Description"]))        

## IMAGE
# Display the image uploaded by the user
if "temp_img_path" in st.session_state and st.session_state['uploaded_image'] == True:
    temp_img_path = st.session_state['temp_img_path']
    with st.chat_message("human"):
        st.image(temp_img_path, width=300, caption=current_image)

    try:
        # Send POST request to a FastAPI endpoint with temporary image path
        FASTAPI_URL = f"http://localhost:8000/image/{llm}"
        with st.spinner("Interpreting image..."):
            response = requests.post(FASTAPI_URL, json={"image_path": temp_img_path})
            # Append the image and response to the chat history
            st.session_state['image_chat_history'].append({"path": temp_img_path, "Description": response.text})
            st.session_state['uploaded_image'] = False

            # Display the AI's interpretation of the image in chat
            with st.chat_message("assistant"):
                # Format the response
                formatted_response = utils.format_response(response.text)
                st.markdown(formatted_response)
    except Exception as e:
        log_error(str(e))
        st.switch_page("pages/error.py")
