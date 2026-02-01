try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
st.set_page_config(
    page_title="Filip's Digital Twin", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# Helper to get secrets with fallback
def get_secret(key, default=None):
    # Try Environment Variables first (preferred for Cloud Run)
    val = os.getenv(key.upper()) or os.getenv(key.lower())
    if val:
        return val
        
    # Fallback to st.secrets only if available (prevents StreamlitSecretNotFoundError)
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
        
    return default

# Initialize Grok API
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = get_secret("grok_api_key")

# Initialize Instantly.ai API credentials
INSTANTLY_API_KEY = get_secret("instantly_api_key")
INSTANTLY_CAMPAIGN_ID = get_secret("instantly_campaign_id")
INSTANTLY_API_URL = "https://api.instantly.ai/api/v2/leads"

# Custom CSS for Premium Look
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Outfit:wght@500;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .stApp {
        background-color: #0e1117;
    }

    .chat-bubble {
        padding: 1.2rem;
        border-radius: 1.5rem;
        margin-bottom: 1rem;
        line-height: 1.5;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .user-bubble {
        background: linear-gradient(135deg, #eb6928 0%, #d4561d 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.3rem;
    }

    .assistant-bubble {
        background-color: #1e293b;
        color: #f1f5f9;
        margin-right: auto;
        border-bottom-left-radius: 0.3rem;
        border: 1px solid #334155;
    }

    .stButton>button {
        background: linear-gradient(135deg, #eb6928 0%, #d4561d 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(235, 105, 40, 0.3);
    }

    /* Hide streamlit main menu and footer, but keep sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Branding */
    .brand-text {
        color: #eb6928;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Chroma and Sentence-BERT with caching
@st.cache_resource
def get_vector_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("persona_knowledge")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return collection, embedder

collection, embedder = get_vector_db()

# GitHub Repo Details for Knowledge Base
GITHUB_KB_URL = "https://api.github.com/repos/fszale/solidcage_ai_twin/contents/knowledge_base"

# Load knowledge base with caching and dynamic GitHub fetching
@st.cache_resource
def load_knowledge_base(force_refresh=False):
    if collection.count() > 0 and not force_refresh:
        print("Knowledge base already loaded in persistent storage.")
        return
        
    if force_refresh:
        print("Forcing refresh of knowledge base...")
        # Note: In a real app we might want to clear the collection first
        # collection.delete(where={}) 
        
    documents = []
    
    # Try fetching from GitHub API
    try:
        print(f"Fetching knowledge base index from GitHub: {GITHUB_KB_URL}")
        response = requests.get(GITHUB_KB_URL)
        if response.status_code == 200:
            files = response.json()
            for file_info in files:
                if file_info["type"] == "file" and file_info["name"].endswith(('.md', '.txt')):
                    print(f"Downloading {file_info['name']}...")
                    file_content_res = requests.get(file_info["download_url"])
                    if file_content_res.status_code == 200:
                        documents.append(file_content_res.text)
        else:
            print(f"Failed to fetch from GitHub API ({response.status_code}). Falling back to local files.")
    except Exception as e:
        print(f"GitHub fetch error: {e}. Falling back to local files.")

    # Fallback to local files if GitHub failed or returned nothing
    if not documents:
        kb_path = "./knowledge_base"
        if os.path.exists(kb_path):
            for file in os.listdir(kb_path):
                file_path = os.path.join(kb_path, file)
                if os.path.isfile(file_path) and file.endswith(('.md', '.txt')):
                    print(f"Loading local {file_path}...")
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents.append(f.read())
    
    if documents:
        print(f"Embedding {len(documents)} documents...")
        embeddings = embedder.encode(documents)
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            collection.add(ids=[f"doc_{i}"], embeddings=[emb.tolist()], documents=[doc])
        print("Knowledge base loaded into ChromaDB.")
    else:
        print("No knowledge base files found anywhere.")

# Query Grok API
def query_grok(prompt, context):
    if not GROK_API_KEY:
        return "Grok API key is missing. Please set it in secrets or environment variables."
        
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    system_prompt = f"""
    You are a digital twin of Filip Szalewicz, a 50-year-old CTO/VP of Engineering consultant. 
    Your tone: Professional, witty, approachable, and actionable.
    Context: {context}
    Mission: Help tech leaders optimize software teams (productivity, innovation, cost).
    Style: Keep responses concise. Prioritize the knowledge base provided. 
    Call to Action: Suggest booking a session at www.solidcage.com if more depth is needed. 
    Avoid: Politics, personal life, or unrelated topics. If unknown, say "I don't know."
    """
    data = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    try:
        response = requests.post(GROK_API_URL, json=data, headers=headers)
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return f"Error from Grok: {result.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Request failed: {str(e)}"

# Function to sync lead data with Instantly.ai
def sync_lead_to_instantly(name, email, team_size, cycle_time, ai_usage, lead_score):
    if not INSTANTLY_API_KEY:
        print("Instantly API key missing, skipping sync.")
        return False
        
    headers = {
        "Authorization": f"Bearer {INSTANTLY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "campaign_id": INSTANTLY_CAMPAIGN_ID,
        "email": email,
        "first_name": name,
        "custom_variables": {
            "team_size": team_size,
            "cycle_time": cycle_time,
            "ai_usage": ai_usage,
            "lead_score": lead_score
        }
    }
    try:
        response = requests.post(INSTANTLY_API_URL, json=payload, headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Instantly sync error: {e}")
        return False

# Agent Logic
def lead_qualification_agent(prompt, step):
    state = st.session_state.agent_state
    
    if step == 1:
        try:
            team_size = int(re.search(r'\d+', prompt).group())
            state["team_size"] = team_size
            state["lead_score"] += 2 if team_size > 10 else 0
            state["step"] = 2
            return "Got it. Typically, what's your team's cycle time for delivering a feature (in days)?"
        except:
            return "I need a number to help you properly. How many engineers are in your team?"

    if step == 2:
        try:
            cycle_time = float(re.search(r'\d+\.?\d*', prompt).group())
            state["cycle_time"] = cycle_time
            state["lead_score"] += 3 if cycle_time > 7 else 0
            state["step"] = 3
            return "Are you currently leveraging AI coding assistants or automated testing? (Yes/No)"
        except:
            return "Just a number please. What's the average cycle time in days?"

    if step == 3:
        ai_usage = prompt.lower()
        state["ai_usage"] = ai_usage
        state["lead_score"] += 2 if "no" in ai_usage else 0
        state["step"] = 4
        return "Perfect. To send you a personalized mini-report, what's your name and best email? (e.g. Filip, filip@solidcage.com)"

    if step == 4:
        match = re.search(r'([\w\s]+),\s*([\w\.-]+@[\w\.-]+\.\w+)', prompt)
        if match:
            name, email = match.groups()
            name = name.strip()
            sync_lead_to_instantly(name, email, state.get("team_size"), state.get("cycle_time"), state.get("ai_usage"), state.get("lead_score"))
            
            cta = "[Book a strategy session](https://crm.solidcage.com/widget/bookings/filip-szalewicz-fractional-cto-calendar-vfs0lblxh)" if state["lead_score"] >= 5 else "Check out my [resources](https://www.solidcage.com)"
            response = f"Thanks {name}! Based on your team size and cycle time, there's significant room for optimization. {cta} to see how we can hit those goals."
            
            # Reset
            st.session_state.agent_state = {"lead_qualifying": False, "diagnostic_active": False, "lead_score": 0, "team_metrics": {}, "step": 0}
            return response
        return "Please provide your name and email separated by a comma. (e.g. John, john@example.com)"

def matches_trigger(prompt, triggers):
    prompt = prompt.lower()
    return any(trigger in prompt for trigger in triggers)

# --- APP LAYOUT ---
inject_custom_css()

# Health Check
if st.query_params.get("path") == "health":
    st.write("OK")
    st.stop()

# Header
st.markdown(f'# <span class="brand-text">Filip Szalewicz</span> AI Twin', unsafe_allow_html=True)
st.markdown("### Operator / Fractional CTO")

# Sidebar / Info
with st.sidebar:
    st.image("https://www.solidcage.com/images/LogoforSolidCage-MinimalistFractionalCTOBusiness.png", width=150)
    st.markdown("---")
    st.markdown("### About Filip")
    st.write("20+ years of experience building and scaling software teams. I help you transform engineering from a cost center to a growth engine.")
    st.markdown("[www.solidcage.com](https://www.solidcage.com)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/fszalewicz)")
    st.markdown("[GitHub](https://github.com/fszale)")
    st.markdown("[YouTube](https://www.youtube.com/@Control-The-Outcome)")

# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"lead_qualifying": False, "diagnostic_active": False, "lead_score": 0, "team_metrics": {}, "step": 0}

# Load KB
load_knowledge_base()

# Display chat history
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask about team performance or 'help my team'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        LEAD_TRIGGERS = ["help my team", "assess my team", "qualify", "diagnose", "improve performance"]
        
        response = ""
        if st.session_state.agent_state["lead_qualifying"]:
            response = lead_qualification_agent(prompt, st.session_state.agent_state["step"])
        elif matches_trigger(prompt, LEAD_TRIGGERS):
            st.session_state.agent_state["lead_qualifying"] = True
            st.session_state.agent_state["step"] = 1
            response = "I'd love to help! To provide the best advice, how many engineers are on your team?"
        else:
            # RAG Query
            query_embedding = embedder.encode([prompt])[0]
            results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
            context = " ".join(results["documents"][0]) if results["documents"] else "No specific context found."
            response = query_grok(prompt, context)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f'<div class="chat-bubble assistant-bubble">{response}</div>', unsafe_allow_html=True)
        
        # Periodic CTA
        if len(st.session_state.messages) > 1 and len(st.session_state.messages) % 5 == 0:
            st.info("💡 Want a direct strategy session? [Book a 30-min call here](https://crm.solidcage.com/widget/bookings/filip-szalewicz-fractional-cto-calendar-vfs0lblxh)")
