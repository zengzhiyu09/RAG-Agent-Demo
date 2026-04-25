import time

import streamlit as st
from agent.react_agent import ReactAgent
from utils.logger_handler import logger

# 标题
st.title("您的理财管家")
st.caption("本回答由AI生成，仅供参考，请仔细甄别，谨慎投资。")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        complete_res = "\n\n".join(response_messages)
        st.session_state["message"].append({"role": "assistant", "content": complete_res})
        st.rerun()


