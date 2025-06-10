import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

#path_file = "/content/drive/MyDrive/Celerates MSIB/ALL Dataset/Copy of API key Gemini.xlsx"
#API_KEY = pd.read_excel(path_file)["api_key"][1]

API_KEY = "AIzaSyAtdTs4aVol1viAQlPz1lthFUsCisGPQj0"

def chat(contexts, history, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You can use given data to answer question about sentiment.",
            ),
            ("human", "This is the data : {contexts}\nUse this chat history to generate relevant answer from recent conversation: {history}\nUser question : {question}"),
        ]
    )
    
    chain = prompt | llm
    completion = chain.invoke(
        {
            "contexts": contexts,
            "history": history,
            "question": question,
        }
    )

    answer = completion.content
    input_tokens = completion.usage_metadata['input_tokens']
    completion_tokens = completion.usage_metadata['output_tokens']

    result = {}
    result["answer"] = answer
    result["input_tokens"] = input_tokens
    result["completion_tokens"] = completion_tokens
    return result


# Begin streamlit with title
st.title("AI Chatbot Assistant")

# Give external data contexts
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        # Check file extension and load file accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            df = df.drop_duplicates()
            contexts = df.to_string()
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            df = df.drop_duplicates()
            contexts = df.to_string()
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            df = None
            contexts = ''
        
        # Display the file content
        if df is not None:
            st.write(f"File has shape : {df.shape}")
            st.write(f"Preview of `{uploaded_file.name}`:")
            st.dataframe(df)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("No file uploaded yet.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Get chat history if not Null
    messages_history = st.session_state.get("messages", [])
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat(contexts, history, prompt)
    answer = response["answer"]
    input_tokens = response["input_tokens"]
    completion_tokens = response["completion_tokens"]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        container = st.container(border=True)
        container.write(f"Input Tokens : {input_tokens}")
        container.write(f"Completion Tokens: {completion_tokens}")
    
    # Display history chat
    with st.expander("See Chat History"):
        st.write("**History Chat:**")
        st.code(history)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})