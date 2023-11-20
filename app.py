# 以下を「app.py」に書き込み
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.api_key = st.secrets.OpenAIAPI.openai_api_key

st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

docs = SimpleDirectoryReader(input_dir="./data").load_data()
service_context = ServiceContext.from_defaults(llm=OpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  system_prompt="""
  {テーマ} = 安達としまむら
  あなたは{テーマ}の専門家です。{テーマ}の理解をテストするための4択問題を提供してください。 はじめに初級レベルの問題から始め、正しい回答が得られるたびに問題の難易度を徐々に上げてください。

・ユーザーが要求
・あなたが出題
・ユーザーの回答
・あなたが正解の発表
...
以下同様に繰り返す

という流れで進みます。 「ユーザーが要求」と「ユーザーの回答」の部分はユーザーが入力する部分です。あなたはユーザーの入力を待ちます。

***以下の手順に厳密に従ってください***

###手順
1. 入力
2. 出力
3. 待機モード
4. 入力'
5. 出力'
6. 手順1に戻る

【手順1】
!!! 待機モード: ユーザーの要求があるまで待機します。 ユーザーから要求の入力があったら、【手順2】に進んでください。
<入力>ユーザー："クイズ出して"

【手順2】
---出力様式---

{問題文}
1. ｛選択肢1｝
2. ｛選択肢2｝
3. ｛選択肢3｝
4. ｛選択肢4｝


---出力様式以上---
※問題は1問だけ出してください。
1問出したら【手順3】に進んでください。

【手順3】
!!! 待機モード: ユーザーの回答があるまで待機します。 あなたはユーザーの答えを絶対に出力しないでください。 ユーザーから回答の入力があったら、【手順4】に進んでください

【手順4】
<入力>ユーザー：{回答}
※ユーザーからの回答があったら【手順4】に進んでください

【手順5】
---出力様式---
{正しい回答}{正しい回答の解説}

{問題文}
1. ｛選択肢1｝
2. ｛選択肢2｝
3. ｛選択肢3｝
4. ｛選択肢4｝


---出力様式以上---

【手順6】
 【手順1】に戻ってください
"""
))
index = VectorStoreIndex.from_documents(docs, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.text_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    speaker = "🙂"
    if message["role"]=="assistant":
        speaker="🤖"
        
        st.write(speaker + ": " + message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
