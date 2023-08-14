import pinecone
from sentence_transformers import SentenceTransformer
import subprocess
import streamlit as st
import torch
import vertexai
from vertexai.language_models import TextGenerationModel
import yaml

# Import config
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup Pinecone
api_key = config["pinecone"]["api_key"]
environment = config["pinecone"]["environment"]
search_index_name = config["pinecone"]["search_index_name"]
pinecone.init(api_key = api_key, environment = environment)
search_index = pinecone.Index(search_index_name)

# Setup Vertex AI
model_name = config["vertex_ai"]["model_name"]
project_id = config["vertex_ai"]["project_id"]
location = config["vertex_ai"]["location"]

# Setup Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device = device)

def single_prediction(
    input_text: str,
    model_name: str
) -> str:

    vertexai.init(project=project_id, location=location)
    parameters = {
        "temperature": 0,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 512,  # Token limit determines the maximum amount of text output.
        "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 1,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained(model_name)
    response = model.predict(
        input_text,
        **parameters,
    )
    return response.text

def rtr_qa(query):
    embedded_query = model.encode(query).tolist()
    query_results = search_index.query(embedded_query, top_k=1, include_metadata=True)
    instruction = query_results["matches"][0]["metadata"]["instruction"]
    context = query_results["matches"][0]["metadata"]["context"]
    input_text = "question: " + instruction + " context: " + context
    response = single_prediction(input_text, model_name)
    return response, context

# --- BEGIN UI ---

st.title("Request Then Read Question Answering")

st.markdown("""Welcome to the **Request Then Read Question Answering Demo**!

Do you want to avoid browsing pages to find the necessary answers? Say goodbye
to the traditional way of searching, and experience the future of intelligent
question-answering:

- **Semantic Search**: Leveraging Pinecone's vector database can precisely match
  your query to previously answered questions.
- **Fine-tuned Responses**: The fine-tuned PaLM 2 model endpoint refines the
  context retrieved from the search to formulate accurate and detailed answers.
- **User-Friendly Interface**: Ask your question, and let the system do the hard
  work.

Start now by typing your question, and experience the magic!""")

query = st.text_input(
    label="What would you like to know?",
    placeholder="What would you like to know?",
    label_visibility="hidden")

if st.button("Ask"):
    response, context = rtr_qa(query)
    st.markdown(f"""{response}
    
Interested in understanding how we arrived at this answer?
    
**View the Context:**

{context}
  
The context provides the detailed information used to answer your question. By
understanding the underlying context, you can gain deeper insights and verify
the accuracy of the response.
  
If you have any further questions or need clarification, please don't hesitate
to ask!""")
