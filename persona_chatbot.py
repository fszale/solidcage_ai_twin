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

# Initialize session state for lead data and agent state
if "lead_data" not in st.session_state:
    st.session_state.lead_data = []  # Store lead information
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"lead_qualifying": False, "diagnostic_active": False, "lead_score": 0, "team_metrics": {}}

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
    You are a persona based on Filip Szalewicz, a 50-year-old CTO/VP of Engineering/Architect/Principle Software Engineer turned consultant, passionate about optimizing software engineering teams. Your tone is professional, witty, and approachable, reflecting your expertise and personality as seen on solidcage.com and your YouTube channel (@Control-The-Outcome). Answer using the provided knowledge base: {context}. Focus on actionable advice for software engineering team productivity, innovation, and cost reduction. Keep your responses short and to the point. If someone wants more detailed answer ask them to book a session at www.solidcage.com. Suggest booking a session at www.solidcage.com when relevant, naturally. If unsure, use general knowledge but prioritize the knowledge base. Mention your experience with AI engineering and software team performance. Avoid discussing personal life, politics, or unrelated topics. If you don't know the answer, say 'I don't know' without elaboration. Use the knowledge base to provide relevant information and insights. Be concise and clear in your responses. Naturally ask if the users want to have their teams qualified or diagnosed to trigger the agents or if they are using AI tools. If they are not using AI tools, ask them to book a free session at www.solidcage.com.
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

# Lead Qualification Agent
def lead_qualification_agent(prompt):
    state = st.session_state.agent_state
    lead_data = st.session_state.lead_data

    # Step 1: Check if we're in lead qualification mode
    if not state["lead_qualifying"]:
        state["lead_qualifying"] = True
        state["lead_score"] = 0
        return "Let’s see if I can help your team. How many engineers are on your team?"

    # Step 2: Collect team size
    if "team_size" not in state:
        try:
            team_size = int(prompt)
            state["team_size"] = team_size
            if team_size > 10:
                state["lead_score"] += 2  # High potential for larger teams
            return "Got it. What’s your current cycle time for delivering features (in days)?"
        except ValueError:
            return "Please enter a number for your team size. How many engineers are on your team?"

    # Step 3: Collect cycle time
    if "cycle_time" not in state:
        try:
            cycle_time = float(prompt)
            state["cycle_time"] = cycle_time
            if cycle_time > 7:
                state["lead_score"] += 3  # High potential if cycle time is slow
            return "Are you currently using AI tools to improve performance? (Yes/No)"
        except ValueError:
            return "Please enter a number for your cycle time in days. What’s your current cycle time?"

    # Step 4: Collect AI usage
    if "ai_usage" not in state:
        ai_usage = prompt.lower()
        state["ai_usage"] = ai_usage
        if "no" in ai_usage:
            state["lead_score"] += 2  # High potential if not using AI
        return "Thanks! Lastly, may I have your name and email to follow up with personalized insights?"

    # Step 5: Collect lead information
    if "lead_info" not in state:
        lead_info = prompt.split(",")
        if len(lead_info) >= 2:
            name, email = lead_info[0].strip(), lead_info[1].strip()
            lead_data.append({"name": name, "email": email, "team_size": state["team_size"], "cycle_time": state["cycle_time"], "ai_usage": state["ai_usage"], "lead_score": state["lead_score"]})
            state["lead_info"] = True
            # Determine response based on lead score
            if state["lead_score"] >= 5:
                response = f"Hi {name}, based on your input, I can significantly help your team! With {state['team_size']} engineers and a {state['cycle_time']}-day cycle time, I can cut that by 50% using AI-driven practices. Let’s discuss how—book a free 30-minute strategy session: [https://cal.com/filip-szalewicz-wl6x3a/30min](https://cal.com/filip-szalewicz-wl6x3a/30min)."
            else:
                response = f"Thanks for sharing, {name}! I’ve got some ideas to improve your team’s performance. Check out my guide on AI-driven engineering at www.solidcage.com, and feel free to reach out if you’d like to explore further."
            # Reset state for next user
            state["lead_qualifying"] = False
            state.pop("team_size", None)
            state.pop("cycle_time", None)
            state.pop("ai_usage", None)
            state.pop("lead_info", None)
            state["lead_score"] = 0
            return response
        return "Please provide your name and email, separated by a comma (e.g., John, john@email.com)."

# Engineering Performance Diagnostic Agent
def engineering_diagnostic_agent(prompt):
    state = st.session_state.agent_state

    # Step 1: Check if we're in diagnostic mode
    if not state["diagnostic_active"]:
        state["diagnostic_active"] = True
        state["team_metrics"] = {}
        return "Let’s diagnose your team’s performance. What’s your current cycle time for delivering features (in days)?"

    # Step 2: Collect cycle time
    if "cycle_time" not in state["team_metrics"]:
        try:
            cycle_time = float(prompt)
            state["team_metrics"]["cycle_time"] = cycle_time
            return "How many features does your team deliver per quarter?"
        except ValueError:
            return "Please enter a number for your cycle time in days. What’s your current cycle time?"

    # Step 3: Collect features per quarter
    if "features_per_quarter" not in state["team_metrics"]:
        try:
            features = int(prompt)
            state["team_metrics"]["features_per_quarter"] = features
            return "What’s your team’s turnover rate (as a percentage)?"
        except ValueError:
            return "Please enter a number for features per quarter. How many features do you deliver?"

    # Step 4: Collect turnover rate and provide diagnosis
    if "turnover_rate" not in state["team_metrics"]:
        try:
            turnover_rate = float(prompt)
            state["team_metrics"]["turnover_rate"] = turnover_rate

            # Diagnostic logic
            cycle_time = state["team_metrics"]["cycle_time"]
            features = state["team_metrics"]["features_per_quarter"]
            diagnosis = []
            if cycle_time > 7:
                diagnosis.append(f"Your cycle time of {cycle_time} days is {round((cycle_time - 7) / 7 * 100)}% slower than my benchmark of 5-7 days. I can help reduce it by 50% with AI-driven practices.")
            else:
                diagnosis.append(f"Your cycle time of {cycle_time} days is within my benchmark of 5-7 days—great work! I can help sustain and improve it further.")
            if features < 4:
                diagnosis.append(f"Your team delivers {features} features per quarter, below my benchmark of 4-6. I can help increase this to 4-6 with continuous delivery practices.")
            else:
                diagnosis.append(f"Your {features} features per quarter align with my benchmark of 4-6—nice! Let’s aim for even more innovation.")
            if turnover_rate > 10:
                diagnosis.append(f"Your turnover rate of {turnover_rate}% is above my benchmark of 10%. I can help reduce it below 10% with team alignment strategies like EOS.")
            else:
                diagnosis.append(f"Your turnover rate of {turnover_rate}% is at or below my benchmark of 10%—excellent! I can help maintain this with ongoing cultural strategies.")

            response = "\n".join(diagnosis) + "\nLet’s discuss how I can help your team improve further. Book a free 30-minute strategy session: [https://cal.com/filip-szalewicz-wl6x3a/30min](https://cal.com/filip-szalewicz-wl6x3a/30min)."

            # Reset state for next user
            state["diagnostic_active"] = False
            state["team_metrics"] = {}
            return response
        except ValueError:
            return "Please enter a percentage for your turnover rate (e.g., 15). What’s your turnover rate?"

# Define trigger phrases for agents
LEAD_QUALIFICATION_TRIGGERS = [
    "qualify my team", "tell me about my team", "assess my team readiness",
    "evaluate my team", "check if my team fits", "can you help my team",
    "is my team a good fit", "support my team"
]

DIAGNOSTIC_TRIGGERS = [
    "diagnose my team", "assess my team", "evaluate my team performance",
    "check my team performance", "analyze my team", "team performance diagnosis",
    "how is my team doing", "improve my team performance"
]

# Streamlit app
st.title("Chat with Filip's AI Twin")

# Marketing section
st.markdown("---")
st.markdown("""
#### Meet Filip Szalewicz, Your Fractional CTO!
With over 20+ years as a CTO, VP of Engineering, Architect and Principal Software Engineer, Filip specializes in transforming software teams. Through [Fractional Consulting](https://www.solidcage.com), he delivers actionable strategies to boost productivity, spark innovation, and cut costs. Check out his insights on YouTube ([@Control-The-Outcome](https://www.youtube.com/@Control-The-Outcome)) or chat with his AI Twin below. 
#### Ready to optimize your team?
""")
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

# Handle user input with updated placeholder
if prompt := st.chat_input("Ask about software team performance, AI engineering, or say 'qualify my team' or 'diagnose my team' to trigger agents!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if we should trigger an agent
    prompt_lower = prompt.lower()
    response = None
    if any(phrase in prompt_lower for phrase in LEAD_QUALIFICATION_TRIGGERS):
        response = lead_qualification_agent(prompt)
    elif any(phrase in prompt_lower for phrase in DIAGNOSTIC_TRIGGERS):
        response = engineering_diagnostic_agent(prompt)
    else:
        # Retrieve relevant documents for general queries
        query_embedding = embedder.encode([prompt])[0]
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
        context = " ".join(results["documents"][0])
        response = query_grok(prompt, context)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Suggest booking (every 3rd interaction)
    if len(st.session_state.messages) % 6 == 0:
        st.markdown("Want to dive deeper? [Book a FREE 30-Minute strategy session!](https://cal.com/filip-szalewicz-wl6x3a/30min)")