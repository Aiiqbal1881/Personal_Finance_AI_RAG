import streamlit as st
import requests

st.set_page_config(page_title="Personal Finance Advisor", page_icon=" ")

st.title("Personal Finance Advisor (AI RAG)")
st.write("Ask finance-related questions and get reliable AI-based answers.")

# Input box
question = st.text_input(
    "Enter your finance question:",
    placeholder="e.g. How should a student save money?"
)

# Button
if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/ask",
                    json={"question": question},
                    timeout=300
                )
                data = response.json()
                st.success("Answer:")
                st.write(data["answer"])
            except Exception as e:
                st.error("Failed to connect to backend API.")
