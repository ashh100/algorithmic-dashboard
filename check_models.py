import streamlit as st
import google.generativeai as genai

st.title("🤖 Model Diagnostic Tool")

# 1. Configure AI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.success("API Key found!")
except Exception as e:
    st.error(f"API Key Error: {e}")

# 2. List Available Models
st.subheader("Models available to your API Key:")

try:
    models = list(genai.list_models())
    found_any = False
    for m in models:
        # We only care about models that can 'generateContent' (write text)
        if 'generateContent' in m.supported_generation_methods:
            st.code(m.name) # This prints the EXACT name you need
            found_any = True
            
    if not found_any:
        st.warning("No text generation models found. Check your Google AI Studio settings.")
        
except Exception as e:
    st.error(f"Error listing models: {e}")