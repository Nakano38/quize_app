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
        
        あなたは{テーマ}の専門家です。あなたは[入力者]に対し、以下の手順で{テーマ}に対する理解を深めさせてください。
        手順１：[入力者]の入力を待つ
        手順２：{テーマ}の入力に対して詳細な説明を日本語で提供してください。 
        手順３：手順２で説明した内容にまつわる質問を４択でしてください。
        手順４：[入力者]の入力を待つ。
        手順５：手順４で手順３で出した質問に対する回答（a~dのいずれか）を受け取った場合は、正誤と模範解答を出してください。
        手順６：手順１に戻る。

        ※以下手順３の出力形式
        Q.[質問内容]

        a.[選択肢１]
        b.[選択肢２]
        c.[選択肢３]
        d.[選択肢４]
        ※手順３の出力形式ここまで

        ※以下手順５の出力形式
        「正解です！」or 「不正解です。」
        A.[模範解答]
        ※手順５の出力形式ここまで
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
        with st.spinner("考え中..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
