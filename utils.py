import re
import os
import logging

## LOGGING CONFIGURATION
## -------------------------------------------------------------------------------------------
# Configure logging to write to a file
def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.ERROR)

def log_error(message):
    logging.error(message)

## HELPER FUNCTIONS
## ------------------------------------------------------------------------------------------
# Function to format response received from a FastAPI endpoint
def format_response(response_text):
    # Replace \n with newline character in markdown
    response_text = re.sub(r'\\n', '\n', response_text)

    # Check for bullet points and replace with markdown syntax
    response_text = re.sub(r'^\s*-\s+(.*)$', r'* \1', response_text, flags=re.MULTILINE)

    # Check for numbered lists and replace with markdown syntax
    response_text = re.sub(r'^\s*\d+\.\s+(.*)$', r'1. \1', response_text, flags=re.MULTILINE)

    # Check for headings and replace with markdown syntax
    response_text = re.sub(r'^\s*(#+)\s+(.*)$', r'\1 \2', response_text, flags=re.MULTILINE)
        
    return response_text

# Function to unlink all images when the application closes
def unlink_images(folder_path):
    # List all files in the folder
    image_files = os.listdir(folder_path)
    
    # Iterate over image files and unlink them
    for image_file in image_files:
        try:
            os.unlink(os.path.join(folder_path, image_file))
            print(f"Deleted: {image_file}")
        except Exception as e:
            print(f"Error deleting {image_file}: {e}")