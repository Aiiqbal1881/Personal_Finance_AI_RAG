from backend.rag_pipeline import ask_finance_question

question = "How should a student save money effectively?"
answer = ask_finance_question(question)

print("Answer:\n", answer)
