import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

BASE_URL = "http://3.85.208.131"  # The API base URL

def vectorize_text(text, company_id):
    """
    Sends the extracted text (saved as a .txt file) and company ID to the vectorize endpoint for vectorization.
    """
    with open("temp.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    files = {'file': open('temp.txt', 'rb')}
    
    # Use the "bge-m3" model for vectorization, which is confirmed from Swagger
    response = requests.post(
        f"{BASE_URL}/vectorize/{company_id}?model_name=bge-m3",
        files=files
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"


def chat_with_model(company_id, model, chat_input):
    """
    Sends a chat request to the external API with the user's input and cleans up the response.
    """
    try:
        response = requests.post(
            f"{BASE_URL}/chat/{model}/{company_id}",
            params={"chat_input": chat_input, "vectorizer_model": "bge-m3"},  # Passing the correct vectorizer_model
            timeout=10
        )
        
        if response.status_code == 200:
            json_response = response.json()
            
            # Check if the response is a list and handle it correctly
            if isinstance(json_response, list):
                json_response = json_response[0]
                
            chatbot_response = json_response.get("response")
            if chatbot_response:
                return chatbot_response.strip()
            else:
                return "No response from the model. Please try again."
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.Timeout:
        return "Request timed out. Please try again later."
    except requests.RequestException as e:
        return f"An error occurred: {e}"


def main():
    st.header('Chat with Text File')
    st.sidebar.title('LLM Chat APP')

    # Add a text input for the company ID
    company_id = st.text_input("Enter your Company ID", value="", max_chars=50)

    # File uploader to select a .txt file
    txt_file = st.file_uploader("Upload your Text File", type='txt')

    if txt_file is not None and company_id:
        # Display the file name and selected company ID
        st.write(f"Company ID: {company_id}")
        st.write(f"File selected: {txt_file.name}")
        
        # Read the text file content
        text = txt_file.read().decode('utf-8')
        
        # Split the text content into chunks (if needed)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Vectorize the extracted text and company ID
        vector_response = vectorize_text(text, company_id)
        
        if vector_response:
            st.write('Vectorization successful!')
            #st.json(vector_response)  # Display the vectorization response in JSON format
        
        query = st.text_input("Ask a Question about your Text File")
        
        if query:
            # Assuming you have a function `chat_with_model` for handling chat
            model = "llama3.1"  # Use a model that supports chat
            response = chat_with_model(company_id, model, query)
            st.write(response)  # Display the chat response

if __name__ == '__main__':
    main()
