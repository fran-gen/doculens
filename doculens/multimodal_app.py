import streamlit as st
import os
import base64
from dotenv import load_dotenv
from byaldi import RAGMultiModalModel
from claudette import Chat, models
from gpt4o_model import GPT4OInferenceModel

# Load environment variables from .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# Initialize RAG Model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
RAG.index(
    input_path="../docs/attention.pdf",
    index_name="attention",
    store_collection_with_index=True,
    overwrite=True,
)

# Initialize Models
claude_chat = Chat(models[1])
gpt4o_model = GPT4OInferenceModel()

# Streamlit App
st.title("Multimodal Document Query App")

# Query Section
query = st.text_input("Enter your query", "Tell me what the BLEU score for the transformer base model is.")
model_choice = st.radio("Choose the LLM to use:", ("Claude", "GPT-4O"))

if st.button("Query the Indexed PDF") and query:
    st.write("Searching the document...")
    results = RAG.search(query, k=1)

    # Verify results
    st.write("### Search Results")
    if not results:
        st.warning("No results found.")
    else:
        st.write("Raw Results Output:")
        st.write(results)

        # Decode Base64 image from results
        image_bytes = base64.b64decode(results[0].base64)

        # Query the selected LLM
        st.write(f"### Querying {model_choice}...")
        if model_choice == "Claude":
            try:
                claude_response = claude_chat([image_bytes, query])
                st.write("#### Claude's Response:")
                st.success(claude_response)
            except Exception as e:
                st.error(f"Error querying Claude: {e}")
        elif model_choice == "GPT-4O":
            try:
                gpt4o_response = gpt4o_model.chat(
                    image_bytes=image_bytes,
                    query=query,
                    max_tokens=4096,
                    temperature=0.0,
                )
                st.write("#### GPT-4O's Response:")
                st.success(gpt4o_response)
            except Exception as e:
                st.error(f"Error querying GPT-4O: {e}")
