import streamlit as st
import sounddevice as sd
import tempfile
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
st.text(f"GROQ API Key Loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Initialize the Whisper model (can be: "tiny", "base", "small", "medium", "large")
model = WhisperModel("base", device="cpu", compute_type="int8")

st.title("🎤 Voice-to-Text Streamlit Agent")

duration = st.slider("Select Recording Duration (seconds)", 1, 10, 5)
st.write("Press the button below and start speaking...")

if st.button("Start Recording"):
    with st.spinner("Recording..."):
        fs = 44100  # Sample rate
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            write(f.name, fs, audio)
            st.success("Recording complete. Transcribing...")

            # Transcribe audio
            segments, info = model.transcribe(f.name, beam_size=5)
            transcript = " ".join([seg.text for seg in segments])

            st.subheader("📝 Transcription:")
            st.write(transcript)

            # Trigger simple mock actions
            text = transcript.lower()
            if text:
                st.info(f"Voice Input - {text}")
                with st.spinner("💬 Querying Groq LLM..."):
                    # response = llm.invoke([{"role": "system", "content": "Helpful AI Assistant"}, {"role": "user", "content": text}])
                    response = llm.invoke([
                        SystemMessage(content="Helpful AI Assistant"),
                        HumanMessage(content=text)
                    ])
                    st.subheader("🤖 LLM Response:")
                    st.write(response)
            else:
                st.info("Voice to Text Failed")

        # Clean up temp file
        os.remove(f.name)
