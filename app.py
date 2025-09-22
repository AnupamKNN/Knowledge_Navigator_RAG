__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re
import time
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, SerpAPIWrapper
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import streamlit as st
import torch

# Load environment variables
load_dotenv()

# Initialize embedding model (force CPU to avoid meta tensor errors if needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="resources/db/chromadb_data")
collection = client.get_collection(name="infofusion_chunks")

# Initialize Wikipedia and SerpAPI agents
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_agent = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)
serp_api_key = os.environ.get("SERP_API_KEY")
serp_api_wrapper = SerpAPIWrapper(serpapi_api_key=serp_api_key)

internet_search_agents = {
    "Wikipedia": wikipedia_agent,
    "SerpAPI": serp_api_wrapper
}

def semantic_retrieval(query, top_k=1):
    query_emb = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    retrieved_chunks = results['documents'][0]
    return retrieved_chunks

def internet_search(query):
    results = {}
    for agent_name, agent_object in internet_search_agents.items():
        try:
            response = agent_object.run(query)
            results[agent_name] = response
        except Exception as e:
            print(f"Agent {agent_name} failed with error: {e}")
    return results

class VectorDBTool(BaseTool):
    name = "VectorDB"
    description = "Searches for answers in the embedded document database."
    def _run(self, query: str) -> str:
        chunks = semantic_retrieval(query, top_k=3)
        return "\n---\n".join(chunks)
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

class WikipediaTool(BaseTool):
    name = "Wikipedia"
    description = "Performs Wikipedia search to answer questions."
    def _run(self, query: str) -> str:
        try:
            return wikipedia_agent.run(query)
        except Exception as e:
            return f"Wikipedia Agent error: {e}"
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

class SerpApiTool(BaseTool):
    name = "SerpAPI"
    description = "Uses SerpAPI for live web search."
    def _run(self, query: str) -> str:
        try:
            return serp_api_wrapper.run(query)
        except Exception as e:
            return f"SerpAPI Agent error: {e}"
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

tools = [VectorDBTool(), WikipediaTool(), SerpApiTool()]


# ========== PROMPTS (UNCHANGED) ==========
direct_custom_prompt = """
You are an expert AI assistant with access to three powerful tools:
- **VectorDB:** An internal vector database containing high-quality, trusted documents and knowledge, preferred for all answers when possible.
- **Wikipedia:** A tool to search Wikipedia for up-to-date factual content when the internal database does not fully answer the question.
- **SerpAPI:** A live internet search tool for real-time and broad web knowledge, used only if neither VectorDB nor Wikipedia provide sufficient information.
**For every user request:**
1. Carefully consider the user's question, reduce it to its core intent if needed.
2. Query VectorDB for the most semantically relevant content chunks.
      - If the retrieved content directly and confidently answers the user's question, use ONLY this content. Compose a precise, well-written answer using and summarizing these chunks. Cite/refer to the found content in your synthesis.
3. If VectorDB does not contain enough or sufficiently relevant information, use the Wikipedia tool.
      - Integrate the best Wikipedia result(s) clearly and accurately with any helpful context you have from earlier steps.
4. If both VectorDB and Wikipedia fail to contain the required answer, use SerpAPI as a last resort.
      - When using SerpAPI, prioritize concise, factual, and up-to-date responses.
5. In all cases, create a direct, complete answer that directly addresses the user's actual question.
      - Explain reasoning step by step if helpful for clarity, but avoid any filler, speculation, or unsupported content.
      - Ground the answer in the retrieved sources and cite their origin.
      - Do NOT hallucinate or invent information beyond what is retrieved.
6. If no sources provide a sufficient answer, honestly state that the information could not be found.
**Remember:**   
- Always try VectorDB first and prefer it for trusted, detailed content.
- Only use external search (Wikipedia, SerpAPI) if necessary, and integrate results transparently and carefully.
- Answer in a natural manner, and do not mention sources explicitly.
**User Question:**   
{input}
{agent_scratchpad}
"""

qa_prompt_template = PromptTemplate.from_template(
    "You are a helpful AI assistant.\n\n"
    "Here is the raw answer:\n\n{query}\n\n"
    "Please rewrite it in a **clear, structured markdown format** with headings (###), "
    "bullet points or numbered steps where helpful, and keep it professional yet engaging."
)

mcq_prompt_template = PromptTemplate.from_template(
    "Using the following textbook content, generate exactly ONE multiple-choice question "
    "with exactly 4 answer options labeled A-D, and only one correct answer.\n\n"
    "The format must be exactly as follows and nothing else:\n\n"
    "Q1: [question text]\n"
    "A1: [correct answer text]\n"
    "A. <option 1>\n"
    "B. <option 2>\n"
    "C. <option 3>\n"
    "D. <option 4>\n"
    "CorrectOption: <A/B/C/D>\n\n"
    "Make sure 'A1' matches exactly one of the four options.\n"
    "Do not generate more than one question.\n"
    "Generate a question that has not been generated before for this topic.\n\n"
    "Content:\n{content}"
)


# ========= Q&A + MCQ EXECUTION =========
def run_direct_qa(question: str, agent_executer, qa_chain):
    result = agent_executer.invoke({"input": question, "agent_scratchpad": ""})
    tool_name = None
    if "intermediate_steps" in result and result["intermediate_steps"]:
        action, _ = result["intermediate_steps"][-1]
        tool_name = action.tool
    raw_answer = result["output"]
    response = qa_chain.invoke({"query": raw_answer})
    return response.get("text", raw_answer), tool_name


def generate_mcq(topic: str, mcq_chain, max_attempts: int = 3):
    for attempt in range(1, max_attempts + 1):
        chunks = semantic_retrieval(topic, top_k=3)
        content_str = "\n".join(chunks)
        resp = mcq_chain.invoke({"content": content_str})
        mcq_text = resp.get("text", "").strip()
        parsed = parse_mcq(mcq_text)
        if parsed["question"] and parsed["options"] and parsed["correct_label"]:
            return parsed
        time.sleep(0.3)
    return parsed


def parse_mcq(mcq_text: str):
    lines = [ln.strip() for ln in mcq_text.splitlines() if ln.strip()]
    question = ""
    correct_text = None
    options = []
    correct_label = None

    for ln in lines:
        if re.match(r'^Q\d*\s*:\s*', ln, flags=re.I):
            question = re.sub(r'^Q\d*\s*:\s*', '', ln, flags=re.I).strip()
        elif re.match(r'^A1\s*:\s*', ln, flags=re.I):
            correct_text = re.sub(r'^A1\s*:\s*', '', ln, flags=re.I).strip()
        elif re.match(r'^[A-D][\.]\s+', ln):
            m = re.match(r'^([A-D])[\.]\s*(.+)$', ln)
            if m:
                label = m.group(1).upper()
                text = m.group(2).strip()
                options.append((label, text))
        elif re.match(r'^CorrectOption\s*:\s*', ln, flags=re.I):
            cl = re.sub(r'^CorrectOption\s*:\s*', '', ln, flags=re.I).strip().upper()
            if cl in ('A', 'B', 'C', 'D'):
                correct_label = cl

    if correct_label is None and correct_text and options:
        norm_correct = re.sub(r'\W+', ' ', correct_text).strip().lower()
        for label, text in options:
            norm_opt = re.sub(r'\W+', ' ', text).strip().lower()
            if norm_correct in norm_opt or norm_opt in norm_correct or norm_correct == norm_opt:
                correct_label = label
                break

    return {
        "question": question,
        "correct_text": correct_text,
        "options": options,
        "correct_label": correct_label,
        "raw": mcq_text
    }


# ========= STREAMLIT APP =========
def main():
    st.title("Knowledge Navigator for InfoFusion Technologies Pvt. Ltd.")

    # --- Sidebar: Provider + Model + Key persistence ---
    st.sidebar.header("⚙️ Model Configuration")
    if "provider" not in st.session_state:
        st.session_state.provider = "OpenAI"

    provider = st.sidebar.selectbox(
        "Choose LLM Provider",
        ["OpenAI", "ChatGroq"],
        index=["OpenAI", "ChatGroq"].index(st.session_state.provider),
        key="provider"
    )

    llm = None
    if provider == "OpenAI":
        if "openai_model" not in st.session_state:
            st.session_state.openai_model = "gpt-4o"
        if "openai_api_key" not in st.session_state:
            st.session_state.openai_api_key = ""

        model_choice = st.sidebar.selectbox(
            "Choose OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-5-nano"],
            index=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-5-nano"].index(st.session_state.openai_model),
            key="openai_model"
        )
        openai_api_key = st.sidebar.text_input(
            "Enter OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            key="openai_api_key"
        )

        if openai_api_key:
            llm = ChatOpenAI(openai_api_key=openai_api_key, model=model_choice, temperature=1)

    elif provider == "ChatGroq":
        if "groq_model" not in st.session_state:
            st.session_state.groq_model = "llama-3.3-70b-versatile"
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = ""

        model_choice = st.sidebar.selectbox(
            "Choose ChatGroq Model",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile"
            ],
            index=[
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile"
            ].index(st.session_state.groq_model),
            key="groq_model"
        )
        groq_api_key = st.sidebar.text_input(
            "Enter ChatGroq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            key="groq_api_key"
        )

        if groq_api_key:
            llm = ChatGroq(groq_api_key=groq_api_key, model=model_choice)

    if not llm:
        st.sidebar.warning("⚠️ Please enter a valid API key to use the assistant.")
        return

    # Build agent + chains dynamically
    prompt = PromptTemplate.from_template(direct_custom_prompt)
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt_template)
    mcq_chain = LLMChain(llm=llm, prompt=mcq_prompt_template)

    tabs = st.tabs(["Q&A", "MCQ Quiz"])

    # --- Direct Q&A Tab ---
    with tabs[0]:
        st.header("Question & Answer")
        user_question = st.text_area("Enter your question (e.x. What is machine learning?):")

        if st.button("Get Answer", key="direct_qa"):
            if user_question.strip():
                with st.spinner("Processing your question..."):
                    answer, tool_used = run_direct_qa(user_question, agent_executer, qa_chain)

                if tool_used:
                    st.info(f"🔧 Tool used: **{tool_used}**")
                else:
                    st.info("ℹ️ No external tool was required (answered directly).")

                st.markdown(answer)
            else:
                st.warning("Please enter a valid question.")

    # ========== MCQ Tab ==========
    with tabs[1]:
        st.header("MCQ Quiz")

        # initialize state
        if "mcq_state" not in st.session_state:
            st.session_state.mcq_state = {
                "topic": None,
                "parsed": None,
                "qid": 0  # simple counter for unique radio keys
            }

        topic = st.text_input("Enter a topic to generate MCQs (e.x. Generative AI):", key="mcq_topic_input")

        generate_btn = st.button("Generate First Question", key="mcq_generate")

        if generate_btn:
            if not topic or not topic.strip():
                st.warning("Please enter a valid topic.")
            else:
                with st.spinner("Generating MCQ from VectorDB..."):
                    parsed = generate_mcq(topic.strip(), mcq_chain)
                st.session_state.mcq_state.update({
                    "topic": topic.strip(),
                    "parsed": parsed,
                    "qid": st.session_state.mcq_state.get("qid", 0) + 1
                })

        # Display current question if exists
        parsed = st.session_state.mcq_state.get("parsed")
        if parsed and parsed.get("question"):
            st.subheader(parsed["question"])

            # prepare radio choices as full strings like "A. Option text"
            options = parsed["options"]
            radio_choices = [f"{label}. {text}" for label, text in options]

            # use qid to ensure unique widget-key (avoids stale selection)
            radio_key = f"mcq_radio_{st.session_state.mcq_state.get('qid', 0)}"
            selected = st.radio("Choose your answer:", radio_choices, key=radio_key)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Submit Answer", key=f"mcq_submit_{st.session_state.mcq_state.get('qid', 0)}"):
                    # Extract selected label
                    sel_label = selected.split('.', 1)[0].strip().upper()
                    correct_label = parsed.get("correct_label")
                    correct_text = parsed.get("correct_text")

                    if correct_label:
                        if sel_label == correct_label:
                            st.success("✅ Correct answer!")
                        else:
                            # find correct option text
                            corr_opt_text = next((t for l, t in options if l == correct_label), correct_text)
                            st.error(f"❌ Wrong. The correct answer is: {correct_label}. {corr_opt_text}")
                    else:
                        # fallback: compare selected text to correct_text
                        sel_text = selected.split('.', 1)[1].strip().lower()
                        if correct_text and (correct_text.strip().lower() in sel_text or sel_text in correct_text.strip().lower()):
                            st.success("✅ Correct answer! (matched by text)")
                        else:
                            if correct_text:
                                st.error(f"❌ Wrong. The correct answer text is: {correct_text}")
                            else:
                                st.error("❌ Unable to determine correct answer from generator output. Try regenerating the question.")

            with col2:
                if st.button("Next Question", key=f"mcq_next_{st.session_state.mcq_state.get('qid', 0)}"):
                    # generate next one from same topic
                    topic_current = st.session_state.mcq_state.get("topic")
                    if not topic_current:
                        st.warning("No topic selected. Enter a topic first.")
                    else:
                        with st.spinner("Generating next MCQ..."):
                            parsed_next = generate_mcq(topic_current, mcq_chain)
                        st.session_state.mcq_state.update({
                            "parsed": parsed_next,
                            "qid": st.session_state.mcq_state.get("qid", 0) + 1
                        })
                        # rerun to show new question immediately
                        st.rerun()

            with col3:
                if st.button("Quit Quiz", key="mcq_quit"):
                    st.session_state.mcq_state = {"topic": None, "parsed": None, "qid": 0}
                    st.info("👋 Thank you for participating in the quiz!")

        else:
            st.info("No MCQ yet. Enter a topic and click 'Generate First Question' to begin.")

if __name__ == "__main__":
    main()
