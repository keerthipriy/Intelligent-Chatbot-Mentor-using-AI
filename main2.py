import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    st.error("Please set GEMINI_API_KEY in your .env file")
    st.stop()

os.environ["GOOGLE_API_KEY"] = gemini_key
genai.configure(api_key=gemini_key)

st.set_page_config(page_title="AI Chatbot Mentor", page_icon="🤖", layout="centered")
st.title(" AI Chatbot Mentor ")

MODULES = ["Python", "SQL", "Machine Learning", "Deep Learning", "Generative AI"]
LEVELS = ["Beginner", "Intermediate", "Experienced"]

if "selected_module" not in st.session_state:
    st.session_state.selected_module = None
if "selected_level" not in st.session_state:
    st.session_state.selected_level = None
if "messages" not in st.session_state:
    st.session_state.messages = []

query_params = st.query_params
if "module" in query_params and st.session_state.selected_module is None:
    if query_params["module"][0] in MODULES:
        st.session_state.selected_module = query_params["module"][0]
if "level" in query_params and st.session_state.selected_level is None:
    if query_params["level"][0] in LEVELS:
        st.session_state.selected_level = query_params["level"][0]

if st.session_state.selected_module is None or st.session_state.selected_level is None:
    st.markdown("### 👋 Welcome to AI Chatbot Mentor")
    st.markdown("Please select a topic and level to begin.")

    col1, col2 = st.columns(2)
    with col1:
        module = st.selectbox(" Choose Topic", ["Select a topic"] + MODULES)
    with col2:
        level = st.selectbox(" Choose Level", ["Select a level"] + LEVELS)

    if module != "Select a topic" and level != "Select a level":
        if st.button("🚀 Start Mentoring Session", use_container_width=True):
            st.session_state.selected_module = module
            st.session_state.selected_level = level

            system_content = (
                f"You are an expert AI mentor specialized ONLY in {module} at {level} level. "
                f"Provide clear, structured educational answers with examples. "
                f"If the question is not related to {module}, respond exactly with: "
                "\"Sorry, I don’t know about this question. Please ask something related to the selected module.\""
            )

            welcome_msg = (
                f"Welcome to {module} AI Mentor ({level} Level)! \n\n"
                "What would you like to learn today?"
            )

            st.session_state.messages = [
                SystemMessage(content=system_content),
                AIMessage(content=welcome_msg),
            ]

            st.query_params["module"] = module
            st.query_params["level"] = level
            st.rerun()

else:
    module = st.session_state.selected_module
    level = st.session_state.selected_level
    st.header(f" {module} | Level: {level}")

    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input(f"Ask about {module} ({level} level)..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.7,
                        max_output_tokens=1024,
                    )

                    response = llm.invoke(st.session_state.messages)

                    ai_content = ""
                    if isinstance(response.content, str):
                        ai_content = response.content
                    elif isinstance(response.content, list):
                        ai_content = "".join(
                            part.get("text", "")
                            for part in response.content
                            if isinstance(part, dict)
                        )

                    st.markdown(ai_content)
                    st.session_state.messages.append(AIMessage(content=ai_content))

                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("📥 Download Chat History")

    def build_chat_text():
        lines = []
        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                continue
            role = "User" if isinstance(msg, HumanMessage) else "AI Mentor"
            lines.append(f"{role}: {msg.content}")
        return "\n\n".join(lines)

    if len(st.session_state.messages) > 1:
        st.download_button(
            label=" Download Conversation (.txt)",
            data=build_chat_text(),
            file_name=f"{module}_{level}_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")
    if st.button("🔄 End Session & Choose New Topic"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()
