__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize Grok API (replace with your API key)
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = st.secrets["grok_api_key"]  # Ensure you set this in your Streamlit secrets

# Initialize Chroma and Sentence-BERT
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("persona_knowledge")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load knowledge base
def load_knowledge_base():
    documents = []
    for file in os.listdir("./knowledge_base"):
        print(f"Loading ./knowledge_base/{file}...")
        with open(f"./knowledge_base/{file}", "r") as f:
            documents.append(f.read())
    with open("./knowledge_base/daily_thoughts.txt", "r") as f:
        documents.append(f.read())
    print("Knowledge base loaded.")
    embeddings = embedder.encode(documents)
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        collection.add(ids=[str(i)], embeddings=[emb.tolist()], documents=[doc])

# Query Grok API
def query_grok(prompt, context):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    system_prompt = f"""
    You are a persona based on [Your Name], a 50-year-old VP of Engineering turned consultant, passionate about optimizing software engineering teams. Your tone is professional, witty, and approachable, reflecting your expertise and personality as seen on solidcage.com and your YouTube channel (@Control-The-Outcome). Answer using the provided knowledge base: {context}. Focus on actionable advice for software engineering team productivity, innovation, and cost reduction. Suggest booking a session at solidcage.com when relevant, naturally. If unsure, use general knowledge but prioritize the knowledge base.
    """
    data = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    response = requests.post(GROK_API_URL, json=data, headers=headers)
    try:
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"API Error: {result['error']}"
        else:
            return f"Unexpected API response: {result}"
    except Exception as e:
        return f"Failed to parse API response: {e}\nRaw response: {response.text}"

# Streamlit app
st.title("Chat with Filip's AI Twin")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load knowledge base on first run
if not collection.count():
    load_knowledge_base()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me about how to improve software team performance or AI engineering!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieve relevant documents
    query_embedding = embedder.encode([prompt])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
    context = " ".join(results["documents"][0])
    
    # Query Grok
    with st.chat_message("assistant"):
        response = query_grok(prompt, context)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Suggest booking (every 3rd interaction)
    if len(st.session_state.messages) % 6 == 0:
        st.markdown("Want to dive deeper? Book a session at [www.solidcage.com](https://www.solidcage.com)!")