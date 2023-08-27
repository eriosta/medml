import streamlit as st
import replicate
import os
import random

# App title
# st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
def llama2():

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Function for generating LLaMA2 response
    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                    "temperature":0.1, "top_p":0.9, "max_length":512, "repetition_penalty":1})
        return output

    suggested_questions = [
        "How does this chatbot work?",
        "How can I get my own API key?",
        "What's the difference between a dataset and a database?",
        "Can you explain the basics of AI and ML and DL?",
        "What are data cleaning best practices?",
        "What is supervised vs unsupervised learning?",
        "What are some challenges in implementing AI in clinics?",
        "How do I evaluate the reliability of an AI tool in medicine?",
        "What is the importance of data quality in medical AI?",
        "How can AI be integrated into clinical workflows?",
        "What are the regulatory considerations for medical AI?",
        "How do I keep up-to-date with the latest AI in healthcare research?",
        "Can you explain overfitting and underfitting?",
        "How do I ensure fairness and equity when using AI in medical decision-making?",
        "What is the impact of AI on radiology and pathology?",
        "How can AI be used in personalized medicine and genomics?"
    ]

    # Display the suggested questions and populate the chat input when clicked
    st.sidebar.subheader("Suggested Questions:")
    for question in suggested_questions:
        if st.sidebar.button(question):
            st.session_state.suggested_input = question  # Store the clicked question in the session state

    # Check if there's any clicked question to populate the chat input
    suggested_input = random.choice(suggested_questions)
    st.write(f"Suggested question: {suggested_input}")

    if prompt := st.chat_input(disabled=not replicate_api):

        st.session_state.suggested_input = ""  # Clear the stored suggested input after use
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)