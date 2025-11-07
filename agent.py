import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

# Configuration
VECTOR_STORE_PATH = "local_vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K_RETRIEVAL = 3


# Define the agent state
class AgentState(TypedDict):
    """State object that flows through the agent graph"""
    question: str  # User's question
    needs_retrieval: bool  # Whether retrieval is needed (from plan node)
    retrieved_docs: list  # Documents retrieved from vector store
    context: str  # Formatted context from retrieved docs
    answer: str  # Generated answer
    reflection_score: float  # Quality score from reflection (0-1)
    reflection_feedback: str  # Feedback from reflection

# Define Reflection Schema
class ReflectionResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0, 0.0 if no context available and 1.0 if context is perfect")
    feedback: str = Field(..., description="Feedback on the answer generated using the available context")

def initialize_components():
    """
    Initialize LLM, embeddings, and vector store
    """
    print("\nInitializing agent components...")
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3 
    )
    print(f"LLM: {GEMINI_MODEL}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"Embeddings: {EMBEDDING_MODEL}")
    
    # Load vector store
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}."
        )
    
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    print(f"Vector store loaded.\n")
    
    return llm, embeddings, vectorstore


# Initialize global components (reused across queries)
llm, embeddings, vectorstore = initialize_components()



# ==================== NODE 1: PLAN ====================
def plan_node(state: AgentState) -> AgentState:
    """
    Plan node: Decides if retrieval is needed for the question, use simple keyword matching
    """
    print(f"\n{'='*60}")
    print("NODE 1: PLAN")
    print(f"{'='*60}")
    print(f"Question: {state['question']}")
    
    # Simple planning logic
    if any(keyword in state['question'].lower() for keyword in ['solar', 'wind', 'energy', 'renewable', 'sustainable', 'sustainable development goals', 'SDG']):
        state['needs_retrieval'] = True
    
    print(f"Decision: {'Retrieval needed' if state['needs_retrieval'] else 'No retrieval needed'}\n")
    
    return state


# ==================== NODE 2: RETRIEVE ====================
def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieve node: Queries vector store for relevant documents
    """
    print(f"{'='*60}")
    print("NODE 2: RETRIEVE")
    print(f"{'='*60}")
    
    if not state["needs_retrieval"]:
        print("Skipping retrieval (not needed)\n")
        state["retrieved_docs"] = []
        state["context"] = ""
        return state
    
    # Query vector store
    print(f"Querying vector store (top {TOP_K_RETRIEVAL} chunks)...")
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": TOP_K_RETRIEVAL})
    docs = retriever.invoke(state['question'])
    
    print(f"Retrieved {len(docs)} documents\n")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    state["retrieved_docs"] = docs
    state["context"] = context
    
    print()
    return state


# ==================== NODE 3: ANSWER ====================
def answer_node(state: AgentState) -> AgentState:
    """
    Answer node: Generates answer using LLM and retrieved context
    """
    print(f"{'='*60}")
    print("NODE 3: ANSWER")
    print(f"{'='*60}")
    
    
    # prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant specializing in renewable energy and sustainable development.
Use the provided context to answer the question accurately and concisely.
If the context doesn't contain enough information or if it is empty, Just say Context not available.

Context:
{context}"""),
        ("human", "{question}")
    ])
    
    # answer generation
    print("Generating answer with Gemini...")
    messages = prompt_template.format_messages(
        context=state["context"],
        question=state["question"]
    )
    
    response = llm.invoke(messages)
    answer = response.content
    
    print("Generated answer: ", answer, "\n")
    state["answer"] = answer
    
    return state


# ==================== NODE 4: REFLECT ====================
def reflect_node(state: AgentState) -> AgentState:
    """
    Reflect node: Evaluates answer quality and provides feedback
    Returns a quality score (0.0-1.0) and feedback string
    """
    print(f"{'='*60}")
    print("NODE 4: REFLECT")
    print(f"{'='*60}")
    
    # Create reflection prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator of AI-generated answers.
Evaluate the given answer based on these criteria:
1. Relevance: Does it address the question?
2. Accuracy: Is it consistent with the context?
3. Completeness: Does it provide sufficient detail?
4. Clarity: Is it well-structured and clear?

Provide:
- A quality score from 0.0 to 1.0 (1 if the context perfectly supports the answer, 0.0 if no context is available or empty)
- Brief feedback on strengths and weaknesses"""),
        ("human", """Question: {question}

Answer to evaluate: {answer}

Context used: {context}""")
    ])
    
    llm_with_structured_output = prompt | llm.with_structured_output(ReflectionResult)
    response = llm_with_structured_output.invoke({
        'question': state['question'], 
        'answer': state['answer'], 
        'context': state['context']
    })
    score = response.score
    feedback = response.feedback
    
    print(f"Quality Score: {score:.2f}")
    print(f"Feedback: {feedback}")
    
    state["reflection_score"] = score
    state["reflection_feedback"] = feedback
    
    return state

# ==================== ROUTING LOGIC ====================
def should_retrieve(state: AgentState) -> str:
    """
    Decides whether to retrieve documents or skip to answer
    
    Returns:
        "retrieve" to retrieve, or "answer" to skip
    """
    return "retrieve" if state["needs_retrieval"] else "answer"         

# ==================== BUILD GRAPH ====================
def build_agent_graph():
    """
    builds the LangGraph workflow with 4 nodes
    """
    print("\nBuilding LangGraph agent workflow...")
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("reflect", reflect_node)
    
    # Define edges
    workflow.set_entry_point("plan")

    workflow.add_conditional_edges(
        "plan",
        should_retrieve,
        {
            "retrieve": "retrieve",
            "answer": 'answer'
        }
    )

    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "reflect")
    workflow.add_edge("reflect", END)
    
    
    # Compile
    app = workflow.compile()
    
    print("Agent graph built successfully\n")
    
    return app


# Create the agent application
agent_app = build_agent_graph()

# ==================== MAIN QUERY FUNCTION ====================
def query_agent(question: str) -> dict:
    """
    Main function to query the agent
    """
    print("\n" + "="*60)
    print("STARTING AGENT EXECUTION")
    print("="*60)
    
    # Initialize state
    initial_state = {
        "question": question,
        "needs_retrieval": False,
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "reflection_score": 0.0,
        "reflection_feedback": ""
    }
 
    final_state = agent_app.invoke(initial_state)
    

    result = {
        "question": question,
        "answer": final_state.get("answer", ""),
        "retrieved_docs": final_state.get("retrieved_docs", []),
        "reflection_score": final_state.get("reflection_score", 0.0),
        "feedback": final_state.get("reflection_feedback", "")
    }
    
    print("="*60)
    print("AGENT EXECUTION COMPLETED")
    print("="*60)
    
    return result
