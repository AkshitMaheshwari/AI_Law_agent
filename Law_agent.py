import streamlit as st
import os
import sys
import pysqlite3
# Patch the standard sqlite3 with pysqlite3
sys.modules["sqlite3"] = pysqlite3
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.wikipedia import WikipediaTools
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.pdf import PDFKnowledgeBase,PDFReader
from phi.embedder.google import GeminiEmbedder
from dotenv import load_dotenv
import tempfile
from phi.document.chunking.document import DocumentChunking
load_dotenv()

st.set_page_config(page_title="Legal Agent",page_icon="⚖️",layout="wide")

st.title("AI Law Firm 🎓")
st.markdown("This is a legal agent that can answer questions about law, legal documents, and provide legal advice. It analyzes your legal documents and provide you legal strategies and guidance along with risk analysis.")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = ChromaDb(
        collection = "law",path = "tmp/chromadb",persistent_client=True,embedder = GeminiEmbedder()
    )
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Document Parsing and Knowledge Base
with st.sidebar:
    
    chunk_size_in = st.sidebar.number_input("Chunk Size", min_value=1, max_value=5000, value=1000)
    overlap_in = st.sidebar.number_input("Overlap", min_value=1, max_value=1000, value=200)
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload legal document (PDF)", 
        type=["pdf"] 
        # accept_multiple_files=True
    )
    if uploaded_file:
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    st.session_state.knowledge_base = PDFKnowledgeBase(
                        path = temp_path,
                        vector_db = st.session_state.vector_db,
                        reader = PDFReader(),
                        chunking_strategy=DocumentChunking(
                            chunk_size = chunk_size_in,
                            overlap = overlap_in
                        )
                    )
                    st.session_state.knowledge_base.load(recreate=True, upsert=True)
                    st.session_state.processed_files.add(uploaded_file.name)

                    st.success("✅Document parsed and added to the knowledge base.")
                except Exception as e:
                    st.error(f"❌ Error processing document: {e}")

# agents
if st.session_state.knowledge_base:
    legal_researcher = Agent(
        name = "Legal Researcher",
        model = Gemini(id="gemini-2.5-pro"),
        knowledge = st.session_state.knowledge_base,
        search_knowledge=True,
        description = "A Legal Researcher AI agent that finds and cites relevant legal cases , regulations, and precedents using all data in the knowledge base.",
        instructions=[
            "Extract all available data from the knowledge base and search for legal cases, regulations, and citations.",
            "If needed, use DuckDuckGo or wikipedia for aditional legal information and references.",
            "Always provide source refrences in your answers."
        ],
        tools=[DuckDuckGo(), WikipediaTools()],
        show_tool_calls=True,
        markdown=True
    )

    contract_analyst = Agent(
        name = "Contract Analyst",
        model = Gemini(id="gemini-2.5-pro"),
        knowledge = st.session_state.knowledge_base,
        search_knowledge=True,
        description = "A Contract Analyst AI agent that reviews contracts and identifies key clauses, risks, and obligations using the full document data",
        instructions=[
            "Extract all available data from the knowledge base and analyze the contract to identify key clauses, risks, and obligations and potential ambiguities.",
            "Reference specific sections of the contract where possible."
        ],
        show_tool_calls=True,
        markdown=True
    )

    strategy_agent = Agent(
        name = "Legal Strategy Agent",
        model = Gemini(id = "gemini-2.5-pro"),
        knowledge = st.session_state.knowledge_base,
        search_knowledge=True,
        description = "A legal strategist AI agent that provides comprehensive risk assessment and strategic recommendations based on all the available data from the contract.",
        instructions=[
            "Using all data from the knowledge base, assess the contract for legal risks and opportunities.",
            "Provide actionable recommendations and ensure compliance with applicable laws."
        ],
        show_tool_calls=True,
        markdown=True
    )

    team_lead = Agent(
        name = "Team Lead",
        model = Gemini(id = "gemini-2.5-pro"),
        description="Team Lead AI - Integrates responses from the Legal Researcher, Contract Analyst, and Legal Strategist into a comprehensive report.",
        instructions=[
            "Combine and summarize all insights provided by the Legal Researcher, Contract Analyst, and Legal Strategist. "
            "Ensure the final report includes references to all relevant sections from the document."
        ],
        show_tool_calls=True,
        markdown=True
    )

    def get_response(query):
        research_response =  legal_researcher.run(query)
        contract_response = contract_analyst.run(query)
        strategy_response = strategy_agent.run(query)
        final_response = team_lead.run(
            f"Summarize and integrate the following insights gathered using the full contract data:\n\n"
            f"Legal Researcher Response:\n{research_response.content}\n"
            f"Contract Analyst Response:\n{contract_response.content}\n"
            f"Legal Strategy Response:\n{strategy_response.content}\n"
            f"Provide a structured legal analysis report that includes key terms, obligations, potential risks and recommendations with references to the document."
        )
        return final_response
    
# App 
if st.session_state.knowledge_base:
    st.header("🔍 Select Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Contract Review", "Legal Research", "Risk Assessment", "Compliance Check", "Custom Query"]
    )

    query = None
    if analysis_type == "Custom Query":
        query = st.text_area("Enter your custom legal question:")
    else:
        predefined_queries = {
            "Contract Review": (
                "Analyze this document, contract, or agreement using all available data from the knowledge base. "
                "Identify key terms, obligations, and risks in detail."
            ),
            "Legal Research": (
                "Using all available data from the knowledge base, find relevant legal cases and precedents related to this document, contract, or agreement. "
                "Provide detailed references and sources."
            ),
            "Risk Assessment": (
                "Extract all data from the knowledge base and identify potential legal risks in this document, contract, or agreement. "
                "Detail specific risk areas and reference sections of the text."
            ),
            "Compliance Check": (
                "Evaluate this document, contract, or agreement for compliance with legal regulations using all available data from the knowledge base. "
                "Highlight any areas of concern and suggest corrective actions."
            )
        }
        query = predefined_queries[analysis_type]

    if st.button("Analyze"):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Analyzing..."):
                response = get_response(query)

                
                tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])

                with tabs[0]:
                    st.subheader("📑 Detailed Analysis")
                    st.markdown(response.content if response.content else "No response generated.")

                with tabs[1]:
                    st.subheader("📌 Key Points Summary")
                    key_points_response = team_lead.run(
                        f"Summarize the key legal points from this analysis:\n{response.content}"
                    )
                    st.markdown(key_points_response.content if key_points_response.content else "No summary generated.")

                with tabs[2]:
                    st.subheader("📋 Recommendations")
                    recommendations_response = team_lead.run(
                        f"Provide specific legal recommendations based on this analysis:\n{response.content}"
                    )
                    st.markdown(recommendations_response.content if recommendations_response.content else "No recommendations generated.")
