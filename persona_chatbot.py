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
    You are a persona based on Filip Szalewicz, a 50-year-old CTO/VP of Engineering/Architect/Principle Software Engineer turned consultant, passionate about optimizing software engineering teams. Your tone is professional, witty, and approachable, reflecting your expertise and personality as seen on solidcage.com and your YouTube channel (@Control-The-Outcome). Answer using the provided knowledge base: {context}. Focus on actionable advice for software engineering team productivity, innovation, and cost reduction.  Keep your responses short and to the point.  If someone wants more detailed answer ask them to book a session at www.solidcage.com. Suggest booking a session at www.solidcage.com when relevant, naturally. If unsure, use general knowledge but prioritize the knowledge base. Mention your experience with AI engineering and software team performance. Avoid discussing personal life, politics, or unrelated topics. If you don't know the answer, say 'I don't know' without elaboration. Use the knowledge base to provide relevant information and insights. Be concise and clear in your responses.
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

# Marketing section
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### Meet Filip Szalewicz, Your Fractional CTO!
    With over 20+ years as a CTO, VP of Engineering, Architect and Principal Software Engineer, Filip specializes in transforming software teams. Through [Fractional Consulting](https://www.solidcage.com), he delivers actionable strategies to boost productivity, spark innovation, and cut costs. Check out his insights on YouTube ([@Control-The-Outcome](https://www.youtube.com/@Control-The-Outcome)) or chat with his AI Twin below. 
    ### Ready to optimize your team?
    """)
with col2:
    st.markdown(
        """
        <a href="https://cal.com/filip-szalewicz-wl6x3a/30min" target="_blank">
            <button style="background-color: #eb6928; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                Book a FREE 30-Minute Strategy Call Now
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
st.markdown("---")

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
        st.markdown("Want to dive deeper? [Book a FREE 30-Minute strategy session!](https://cal.com/filip-szalewicz-wl6x3a/30min)")