import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# API Key Gemini Anda
API_KEY = ""

# Fungsi chat Gemini
def chat(contexts, history, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda adalah asisten cerdas yang membantu menjelaskan dan memberikan informasi sejarah atau konteks mengenai daftar instansi pemerintahan."),
        ("human", "Berikut adalah daftar instansi yang relevan:\n{contexts}\n\n"
                  "Riwayat percakapan sebelumnya:\n{history}\n\n"
                  "Pertanyaan pengguna:\n{question}")
    ])

    chain = prompt | llm
    completion = chain.invoke({
        "contexts": contexts,
        "history": history,
        "question": question,
    })

    return {
        "answer": completion.content,
        "input_tokens": completion.usage_metadata['input_tokens'],
        "completion_tokens": completion.usage_metadata['output_tokens']
    }

# Streamlit App
st.title("Chatbot Arsip Sejarah Instansi")

uploaded_file = st.file_uploader("Unggah Dataset Arsip Instansi", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)
    df.fillna("", inplace=True)

    st.write("📄 Data Instansi (Contoh):")
    st.dataframe(df.head())

    # Inisialisasi konteks
    full_context = "\n".join(df["Data_Arpus"].astype(str).tolist())

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanyakan tentang instansi atau sejarahnya..."):
        history = "\n".join([f'{m["role"]}: {m["content"]}' for m in st.session_state.messages])
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Kirim ke Gemini
        response = chat(full_context, history, prompt)
        answer = response["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("📊 Token Info"):
                st.write(f"Input Tokens : {response['input_tokens']}")
                st.write(f"Completion Tokens : {response['completion_tokens']}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
