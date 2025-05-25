import streamlit as st
from streamlit_mic_recorder import mic_recorder
import tempfile
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
st.text(f"GROQ API Key Loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")

llm = ChatGroq(model="llama-3.3-70b-versatile")
model = WhisperModel("base", device="cpu", compute_type="int8")

st.title("🎤 Voice-to-Text Streamlit Agent")
st.write("Press the **Start Recording** button below and speak. When you stop, the audio will be transcribed and sent to the LLM.")

audio_data = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording")

if audio_data and "bytes" in audio_data:
    audio_bytes = audio_data["bytes"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_filename = f.name

    st.success("Recording complete. Transcribing...")

    # Transcribe audio
    segments, _ = model.transcribe(temp_filename, beam_size=5)
    transcript = " ".join([seg.text for seg in segments])

    st.subheader("📝 Transcription:")
    st.write(transcript)

    text = transcript.lower()
    if text.strip():
        st.info(f"Voice Input - {text}")
        with st.spinner("💬 Querying Groq LLM..."):
            response = llm.invoke([
                SystemMessage(content="Helpful AI Assistant"),
                HumanMessage(content=text)
            ])
            st.subheader("🤖 LLM Response:")
            st.write(response)
    else:
        st.info("Voice to Text Failed")

    os.remove(temp_filename)
