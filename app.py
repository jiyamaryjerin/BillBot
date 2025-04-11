import streamlit as st

# Set page config
st.set_page_config(page_title="Mistral Chat", layout="wide")

# Initialize session state
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fake Mistral call — replace this
def call_mistral(prompt):
    return f"Mistral's answer to: {prompt}"

# Toggle chat visibility
def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open

# Floating button (bottom right)
chat_button_html = """
<style>
    .chat-toggle-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #25D366;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 30px;
        text-align: center;
        cursor: pointer;
        z-index: 9999;
    }
</style>
<div>
    <form action="" method="post">
        <button name="chat_click" class="chat-toggle-button">💬</button>
    </form>
</div>
"""
st.markdown(chat_button_html, unsafe_allow_html=True)

# Check if chat toggle form was submitted
if st.request.method == "POST" and "chat_click" in st.form_submit_button:
    toggle_chat()

# Chat popup simulation
if st.session_state.chat_open:
    with st.container():
        chat_box_style = """
        <style>
            .chat-popup {
                position: fixed;
                bottom: 100px;
                right: 20px;
                width: 300px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                z-index: 9999;
            }
        </style>
        <div class="chat-popup">
        """
        st.markdown(chat_box_style, unsafe_allow_html=True)

        with st.form("chat_form"):
            user_input = st.text_input("Ask Mistral")
            submitted = st.form_submit_button("Send")
            if submitted and user_input.strip():
                response = call_mistral(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Mistral", response))

        # Display chat history
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {msg}")

        st.markdown("</div>", unsafe_allow_html=True)
