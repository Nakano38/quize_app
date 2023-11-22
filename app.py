import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.OpenAIAPI.openai_api_key
st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="""
{テーマ} = 安達としまむら

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

***以下の手順に厳密に従ってください***

###手順
0. 要求
1. 出力
2. 待機モード
3. 入力
4. 出力'
5. 手順2に戻る

【手順0】
ユーザーがなんらかのメッセージで開始要求を入力する

【手順1】
---出力様式---
【手順1】

講師:{問題文}
1. ｛選択肢1｝
2. ｛選択肢2｝
3. ｛選択肢3｝
4. ｛選択肢4｝


---出力様式以上---
※問題は1問だけ出してください。
1問出したら【手順2】に進んでください。

【手順2】
出力:
"【手順2】>> "
!!! 待機モード: ユーザーの回答があるまで待機します。 あなたはユーザーの答えを絶対に出力しないでください。 ユーザーから回答の入力があったら、【手順3】に進んでください

【手順3】
<入力>ユーザー：{回答}
※ユーザーからの回答があったら【手順4】に進んでください

【手順4】
---出力様式---
【手順4】
講師：{正しい回答}{正しい回答の解説}

{問題文}
1. ｛選択肢1｝
2. ｛選択肢2｝
3. ｛選択肢3｝
4. ｛選択肢4｝


---出力様式以上---
 
「ユーザーの要求」があるまで待機してください。
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
