
import streamlit as st
import openai
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# Streamlit Community Cloudの「Secrets」からOpenAI API keyを取得
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": """
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
        }
        ]

# チャットボットとやりとりする関数
def communicate():
    messages = st.session_state["messages"]

    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    documents = SimpleDirectoryReader("data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(user_message)

    bot_message = response["choices"][0]["message"]
    messages.append(bot_message)

    st.session_state["user_input"] = ""  # 入力欄を消去


# ユーザーインターフェイスの構築
st.title("My AI Assistant")
st.write("ChatGPT APIを使ったチャットボットです。")

user_input = st.text_input("メッセージを入力してください。", key="user_input", on_change=communicate)

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in reversed(messages[1:]):  # 直近のメッセージを上に
        speaker = "🙂"
        if message["role"]=="assistant":
            speaker="🤖"

        st.write(speaker + ": " + message["content"])
