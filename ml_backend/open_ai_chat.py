from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from threading import Thread


def get_initial_prompt():
    return "You're a cow"


def load_api_key():
    try:
        api_key = st.secrets["openai"]["api_key"]
    except Exception:
        load_dotenv()
        api_key = os.getenv("OPEN_AI_KEY")

    if not api_key:
        st.error("OpenAI API key not found. Check your .env or Streamlit secrets.")

    else:
        return api_key


def add_message(message: tuple[str, str]):  # FIXME (Problem with Threads)
    if st.session_state.messages is None:
        st.session_state.messages = [message]
    else:
        st.session_state.messages.append(message)

    if len(st.session_state.messages) > 10:
        st.session_state.messages.pop(0)
        st.session_state.messages_ai.pop(1)


def add_ai_message(message: str):
    if st.session_state.messages_ai is None:
        st.session_state.messages_ai = [
            {"role": "system", "content": get_initial_prompt()}
        ]
        st.session_state.messages_ai = [message]
    else:
        st.session_state.messages_ai.append(message)


def ask_open_ai(message):
    add_ai_message(message)

    def run_open_ai_request(messages_ai, apy_key):
        client = OpenAI(api_key=apy_key)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.7,
            messages=messages_ai,
        )

        message = completion.choices[0].message.content

        add_ai_message(message)
        add_message((message, "ai"))

        st.session_state.ai_is_thinking = False

        return "HI"

    thread = Thread(
        target=run_open_ai_request, args=(st.session_state.messages_ai, load_api_key())
    )
    print(thread.start())
