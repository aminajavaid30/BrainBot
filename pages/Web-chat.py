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
    st.title("Web-Chat")

question = None
llm = st.session_state["llm"]

if "current_website" in st.session_state:
    current_website = st.session_state['current_website']
    if st.sidebar.button("Upload New Webpage Link"):
        st.switch_page("BrainBot.py")
    st.subheader("Your webpage content has been uploaded successfully. You can now chat with it.")
    st.success(current_website)
    question = st.chat_input("Type your question here...")
else:
    st.warning("Upload a webpage link to begin chat with it.")
    if st.button("Upload Webpage Link"):
        st.switch_page("BrainBot.py")

## CHAT
# Clear the web chat history if user has uploaded a new webpage link
if st.session_state['uploaded_link'] == True:
    st.session_state['web_chat_history'] = []

# Display the web chat history
for message in st.session_state['web_chat_history']:
    with st.chat_message("user"):
        st.write(message["Human"])
    with st.chat_message("ai"):
        st.markdown(utils.format_response(message["AI"]))

## QUESTION - WITH CHAT HISTORY
## -------------------------------------------------------------------------------------------
# Retrieve the answer to the question asked by the user 
if question is not None:
    # Display the question entered by the user in chat
    with st.chat_message("user"):
        st.write(question)

    resource = "web"

    try:
        # Send POST request to a FastAPI endpoint to retrieve an answer for the question
        data = {"question": question, "resource": resource}
        FASTAPI_URL = f"http://localhost:8000/answer_with_chat_history/{llm}"
        with st.spinner("Generating response..."):
            response = requests.post(FASTAPI_URL, json=data)
            # Append the response to the chat history
            st.session_state['web_chat_history'].append({"Human": question, "AI": response.text})
            st.session_state['uploaded_link'] = False
            # Display the AI's response to the question in chat
            with st.chat_message("ai"):
                # Format the response
                formatted_response = utils.format_response(response.text)
                st.markdown(formatted_response)
    except Exception as e:
        log_error(e)
        st.switch_page("error.py")