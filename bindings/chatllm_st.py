import streamlit as st
import sys
from chatllm import ChatLLM, LibChatLLM, ChatLLMStreamer

st.title("ChatLLM Chatbot Demo")

if 'llm' not in st.session_state:
    args = sys.argv[1:]
    args.append('--hide_banner')
    st.session_state.llm = ChatLLM(LibChatLLM(), args)
    st.session_state.llm_streamer = ChatLLMStreamer(st.session_state.llm)


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.llm_streamer.chat(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})