# BrainBot - AI Learning Assistant
BrainBot is an AI learning assistant utilizing the capabilities of Natural Language Processing (NLP) and Retreival Augmented Generation (RAG). It allows students to interact and chat with different files, webpages, and images to help them understand their content. It has been built using state-of-the-art technologies and practices including LangChain, LLMs (GPT-4 and GROQ), FastAPI, Docker, and LangSmith.

**LangChain, FastAPI, Docker, LangSmith and LLMs (GPT-4 and GROQ)**

This application allows users to chat with their documents, images or webpages by asking questions about them. The documents could be pdfs, word docs, text files, html files or power point presentations. Users can upload any image to interpret it or a webpage link to chat with it.
- Supported file formates are PDF, DOCS, TXT, PPTX HTML
- Supported image formats are PNG, JPG, JPEG
- The application's frontend has been built using Streamlit.
- The application's backend has been built using FastAPI and LangChain.
- LLM options are available through OpenAPI API (GPT-4) and through GROQ API (Llama3-8b-8192).
- It uses the following pre-trained model from Hugging Face for image interpretation:
  - Salesforce/blip-image-captioning-large 
- LangSmith functionality has been added for monitoring the application.
- The API uses Retrieval Augmented Generation (RAG) to implement context specific chat with user documents.
- It uses the FAISS vectorstore for implementing semantic search capability.

## Table of Contents
- [BrainBot - AI Learning Assistant](#brainbot)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Building and Running the Docker Container](#building-and-running-the-docker-container)
  - [Testing the API](#testing-the-api)
  - [Interacting with the API](#interacting-with-the-api)
  - [Acknowledgments](#acknowledgments)
  - [References](#references)
  - [License](#license)

## Introduction
This application is built using LangChain, FastAPI, and Streamlit and leverages the power of LLMs (GPT-4, GROQ) and a pre-trained image-to-text model from the Hugging Face model hub. It allows the user to upload any file, image, or webpage link and chat with it by asking questions using retrieval augmented generation (RAG). 

## Installation
To install and run the API locally, follow these steps: 

1. Clone this repository to your local machine.
2. Ensure you have Docker installed.
3. Build the Docker container using the provided Dockerfile.
4. Run the Docker container.

## Usage
The application has a Streamlit frontend through which user can choose the LLM, and upload any content to chat with it. 

To use the backend API, send HTTP requests to the appropriate endpoints. The API provides the following endpoints:

- `GET /`: Welcome endpoint, returns a greeting message.
- `POST /set_api_key`: Sets the API Key entered by the userfor using OpenAI API.
- `POST /load_file/{llm}`: Loads the file uploaded by the user, splits the text into document chunks, and stores it into a vectorstore.
- `POST /image/{llm}`: Loads the image uploaded by the user.
- `POST /load_link/{llm}`: Loads the webpage link uploaded by the user, splits the text into document chunks, and stores it into a vectorstore.
- `POST /answer_with_chat_history/{llm}`: Takes the user question, extracts the relevant content from the vector store through semantic search, and sends the question along with that retrieved content to the LLM to generate a response for the user.  

## Documentation
The backend API is documented using FastAPI's automatic documentation features. You can access the API documentation using the Swagger UI or ReDoc interface. Simply navigate to the appropriate URL after starting the API server.

- **Swagger UI**  `http://localhost:8000/docs`
- **ReDoc**  `http://localhost:8000/redoc`

## Building and Running the Docker Container
To build and run the Docker container, follow these steps:
1. Navigate to the folder in which your application resides.
2. Build a Docker image using the following command
    ```
    docker build -t brainbot_image .
    ```
3. Containerize the application by creating a Docker container from the built image
    ```
    docker run -p 8000:8000 -p 8501:8501 brainbot_image
    ```
4. The application Streamlit frontend will be avaialble at `http://localhost:8501`
5. The API will be available at `http://localhost:8000`
6. The API documentaion will be avaialable at `http://localhost:8000/docs` or `http://localhost:8000/redoc`

## Testing the API
Test the API using the following command:
```
pytest
```
It will automatically run the predefined test cases.
   
## Interacting with the API
Once the application is running, you can interact with it through the Streamlit frontend. To interact with the API, yopu can send HTTP requests through Swagger UI.

## Acknowledgments
This API was built with inspiration from various open-source projects and libraries. Special thanks to the developers and contributors of FastAPI, LangChain, OpenAI API, GROQ API, Docker, and Hugging Face.

## References
[How to Build an AI Tutor that Can Adapt to Any Course and Provide Accurate Answers Using Large Language Model and Retrieval-Augmented Generation](https://arxiv.org/pdf/2311.17696)

## License
This project is licensed under the [Apache license version 2.0](LICENSE).

