import streamlit as st
from agent import query_agent
import os

st.set_page_config(
    page_title="RAG Q&A Agent",
    layout="centered",
)

def check_vector_store():
    return os.path.exists("local_vector_store")


def main():

    st.title("**RAG-based Q&A Agent**")

    if not check_vector_store():
        st.error("Vector store not found")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "metadata" in message and message["role"] == "assistant":
                metadata = message["metadata"]
                score = metadata.get("score", 0)

                st.write(f"**Quality Score:** {score:.2f}")

                if metadata.get("retrieved_docs"):
                    with st.expander("Retrieved Context and feedback"):
                        for i, doc in enumerate(metadata["retrieved_docs"], 1):
                            st.write(f"**Document {i}:**")
                            st.markdown(doc.page_content[:300] + "...")
                            st.divider()
                        st.markdown(f"**Feedback:** {metadata.get('feedback', '')}")


    # Chat input box
    if question := st.chat_input("Type your Query Here..."):
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query_agent(question)

                    st.markdown(result["answer"])
                    st.write(f"**Quality Score:** {result['reflection_score']:.2f}")

                    if result.get("retrieved_docs"):
                        with st.expander("Retrieved Context and feedback"):
                            for i, doc in enumerate(result["retrieved_docs"], 1):
                                st.write(f"**Document {i}:**")
                                st.markdown(doc.page_content[:250] + "...")
                                st.divider()
                            st.markdown(f"**Feedback:** {result['feedback']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "score": result["reflection_score"],
                            "retrieved_docs": result.get("retrieved_docs", []),
                            "feedback": result.get("feedback", "")
                        }
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please check that the agent is configured properly and vector store exists.")

    with st.sidebar:
        st.sidebar.title("Example Queries you can ask:")
        example_questions = [
            "What are the benefits of solar energy?",
            "How does wind energy generation work?",
            "What is the role of renewable energy in combating climate change?",
            "Can you explain the concept of energy storage in renewable systems?",
            "What are some recent advancements in renewable energy technologies?"
        ]
        for no, q in enumerate(example_questions, 1):
            st.sidebar.markdown(f"{no}. {q}")  


if __name__ == "__main__":
    main()
