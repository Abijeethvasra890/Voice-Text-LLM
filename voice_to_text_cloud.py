import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import av
from scipy.io.wavfile import write # Make sure this is imported

load_dotenv()
st.text(f"GROQ API Key Loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Initialize the Whisper model (can be: "tiny", "base", "small", "medium", "large")
# Consider using a smaller model for Streamlit Cloud to manage resource usage
model = WhisperModel("base", device="cpu", compute_type="int8")

st.title("🎤 Voice-to-Text Streamlit Agent")

# Class to process audio frames from WebRTC
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []
        self.sample_rate = 44100  # Will be updated by the actual audio stream
        self.is_recording = False

    # Change from `recv` to `recv_queued`
    def recv_queued(self, frames: list[av.AudioFrame]) -> None:
        if self.is_recording:
            for frame in frames: # Iterate through all queued frames
                # Convert audio frame to numpy array and append
                self.audio_chunks.append(frame.to_ndarray())
                self.sample_rate = frame.sample_rate # Update sample rate based on actual stream
        # No need to return a frame with recv_queued

# Streamlit session state to manage recording and transcription
if "recording_started" not in st.session_state:
    st.session_state.recording_started = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None

st.write("Press the button below and start speaking...")

# Use streamlit-webrtc to get audio input
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

if st.button("Start/Stop Recording"):
    if webrtc_ctx.audio_processor:
        webrtc_ctx.audio_processor.is_recording = not webrtc_ctx.audio_processor.is_recording
        st.session_state.recording_started = webrtc_ctx.audio_processor.is_recording

        if not st.session_state.recording_started:
            # Recording stopped, process the audio
            if webrtc_ctx.audio_processor.audio_chunks:
                with st.spinner("Processing audio and transcribing..."):
                    # Concatenate all recorded chunks
                    recorded_audio = np.concatenate(webrtc_ctx.audio_processor.audio_chunks, axis=0)

                    # Ensure it's mono if necessary (Whisper expects mono)
                    if recorded_audio.ndim > 1:
                        recorded_audio = recorded_audio.mean(axis=1)

                    # Normalize to -1.0 to 1.0 if it's not already
                    if recorded_audio.dtype != np.float32:
                        recorded_audio = recorded_audio.astype(np.float32) / np.iinfo(recorded_audio.dtype).max

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_audio_path = f.name
                        # Save the audio to a temporary WAV file for faster-whisper
                        write(temp_audio_path, webrtc_ctx.audio_processor.sample_rate, recorded_audio)

                        st.success("Recording complete. Transcribing...")

                        # Transcribe audio
                        segments, info = model.transcribe(temp_audio_path, beam_size=5)
                        transcript = " ".join([seg.text for seg in segments])
                        st.session_state.transcript = transcript

                        st.subheader("📝 Transcription:")
                        st.write(st.session_state.transcript)

                        # Trigger simple mock actions
                        text = st.session_state.transcript.lower()
                        if text:
                            st.info(f"Voice Input - {text}")
                            with st.spinner("💬 Querying Groq LLM..."):
                                response = llm.invoke([
                                    SystemMessage(content="You are a helpful AI assistant. Provide concise and relevant answers."),
                                    HumanMessage(content=text)
                                ])
                                st.session_state.llm_response = response.content
                                st.subheader("🤖 LLM Response:")
                                st.write(st.session_state.llm_response)
                        else:
                            st.info("Voice to Text Failed")

                    # Clean up temp file
                    os.remove(temp_audio_path)
                # Clear audio chunks for next recording
                webrtc_ctx.audio_processor.audio_chunks = []
            else:
                st.warning("No audio was recorded.")
        else:
            st.info("Recording started... Press the button again to stop.")
    else:
        st.warning("WebRTC streamer is not ready. Please refresh the page if this persists.")

# Display current status
if st.session_state.recording_started:
    st.info("Recording in progress...")
else:
    st.info("Ready to record.")

if st.session_state.transcript:
    st.subheader("📝 Last Transcription:")
    st.write(st.session_state.transcript)

if st.session_state.llm_response:
    st.subheader("🤖 Last LLM Response:")
    st.write(st.session_state.llm_response)