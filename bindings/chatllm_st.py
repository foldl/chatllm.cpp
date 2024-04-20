import streamlit as st
import sys
from chatllm import ChatLLM, LibChatLLM, ChatLLMStreamer

st.title("ChatLLM Chatbot Demo")

if 'llm' not in st.session_state:
    args = sys.argv[1:]
    st.session_state.llm = ChatLLM(LibChatLLM(), args, False)
    st.session_state.llm_streamer = ChatLLMStreamer(st.session_state.llm)
    st.session_state.banner = st.session_state.llm_streamer.flush_output()

if st.session_state.banner != '':
    with st.expander("LLM Information"):
        st.code(st.session_state.banner, language=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def stop_streaming():
    st.session_state.llm_streamer.abort()
    response = st.session_state.llm_streamer.get_acc_resp()
    st.session_state.messages.append({"role": "assistant", "content": response})

# Accept user input
if prompt := st.chat_input("What is up?"):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.button('Stop', on_click=stop_streaming)
        response = st.write_stream(st.session_state.llm_streamer.chat(prompt))
        references = st.session_state.llm_streamer.llm.references
        if len(references) > 0:
            with st.expander("References"):
                for r in references:
                    st.caption(r)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()