system_message = f"""
    You are a helpful assistant that answers questions based on the provided context.
    Use only the information from the context to answer the question.
    If the context doesn't contain enough information to answer the question, say so.
"""

# Create the prompt with context
user_message = (
    lambda context, prompt: f"""
    Based on the following context, please answer the question.

    Context: {context}

    Question: {prompt}

    Answer:"""
)
