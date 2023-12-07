import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.OpenAIAPI.openai_api_key
st.title("教師ChatBotアプリ")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "こんにちは！私は質問に対して、解説と確認クイズを出すChatBotです。何でも質問してください！"}
    ]

@st.cache_resource(show_spinner=False)
# チャットボットとやりとりする関数
def load_data():
    with st.spinner(text="しばらくお待ちください"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="""
        {テーマ} = 「安達としまむら」と「現代哲学」 

あなたは{テーマ}の専門家です。{テーマ}の理解をテストするための4択問題を提供してください。 はじめに初級レベルの問題から始め、正しい回答が得られるたびに問題の難易度を徐々に上げてください。 

・ユーザーの要求
・あなたが出題 
・ユーザーの回答 
・あなたが正解の発表及び次の出題 
・ユーザーの回答 
・あなたが正解の発表及び次の出題 
・ユーザーの回答 
... 
以下同様に繰り返す

という流れで進みます。 「ユーザーの要求」、「ユーザーの回答」の部分はユーザーが入力する部分です。あなたはユーザーの入力を待ちます。

        """))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
             
index = load_data()


if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
