# app.py (Streamlit Frontend)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import websockets
import asyncio
import json
import numpy as np # Needed for numpy operations on audio frames

# Streamlit page configuration
st.set_page_config(page_title="Real-time Voice-to-LLM", layout="wide")

st.title("🎤 Real-time Voice-to-LLM Agent")
st.markdown("---")

# Session state for transcript and LLM response
if "partial_transcript" not in st.session_state:
    st.session_state.partial_transcript = ""
if "llm_response_stream" not in st.session_state:
    st.session_state.llm_response_stream = ""
if "is_streaming_active" not in st.session_state:
    st.session_state.is_streaming_active = False

# Define the WebSocket URL for your backend ASR service
# IMPORTANT: If running on a different machine or deployed, replace "localhost" with the backend's IP/domain
WEBSOCKET_URL = "ws://localhost:8000/ws/audio"

class AudioSender(AudioProcessorBase):
    def __init__(self):
        self.websocket = None
        # Use a new event loop for this thread to manage async WebSocket operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.receiving_task = None # To hold the async task for receiving

    async def _connect_ws(self):
        try:
            st.info("Attempting to connect to WebSocket backend...")
            self.websocket = await websockets.connect(WEBSOCKET_URL)
            st.success("Connected to WebSocket backend!")
        except Exception as e:
            st.error(f"WebSocket connection error: {e}. Please ensure the backend is running at {WEBSOCKET_URL}")
            self.websocket = None # Ensure websocket is None on failure

    async def _send_audio(self, frame: av.AudioFrame):
        if self.websocket and not self.websocket.closed:
            try:
                # Convert audio frame to bytes
                # Ensure the audio is mono (if multi-channel) and at 16-bit PCM for efficiency
                # The backend expects 16-bit signed integers.
                audio_array = frame.to_ndarray().flatten().astype(np.int16)
                await self.websocket.send(audio_array.tobytes())
            except Exception as e:
                st.warning(f"Error sending audio: {e}. Attempting to reconnect...")
                if self.websocket:
                    await self.websocket.close()
                self.websocket = None # Reset connection to trigger reconnect

    async def _receive_ws(self):
        if self.websocket:
            try:
                while True:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    if data.get("type") == "transcript_partial":
                        st.session_state.partial_transcript = data["text"]
                        # We use st.rerun() here to update the UI
                        # Note: st.rerun() restarts the script from top, use sparingly
                        # For very high refresh rates, you might consider a more
                        # advanced Streamlit approach or simpler display.
                        st.rerun()
                    elif data.get("type") == "llm_response_chunk":
                        # Append LLM response chunks
                        st.session_state.llm_response_stream += data["text"]
                        st.rerun()
            except websockets.exceptions.ConnectionClosedOK:
                st.info("WebSocket connection closed cleanly by backend.")
            except Exception as e:
                st.error(f"Error receiving from WebSocket: {e}")
            finally:
                self.websocket = None # Ensure websocket is None on connection close/error

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Check if WebSocket is connected. If not, try to connect.
        if not self.websocket:
            self.loop.run_until_complete(self._connect_ws())
            if self.websocket and not self.receiving_task:
                # Start the background task to receive messages from WebSocket
                self.receiving_task = self.loop.create_task(self._receive_ws())

        if self.websocket and not self.websocket.closed:
            self.loop.run_until_complete(self._send_audio(frame))
        return frame # Return frame to allow potential further processing in other components


# UI for the WebRTC streamer
st.write("Click 'Start' to enable audio streaming to the backend.")
webrtc_ctx = webrtc_streamer(
    key="realtime_audio",
    mode=WebRtcMode.SENDONLY, # Only send audio
    audio_processor_factory=AudioSender,
    media_stream_constraints={"video": False, "audio": True}, # Request only audio
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Public STUN server for NAT traversal
    },
    # Manual start/stop control
    desired_playing_state=st.session_state.is_streaming_active
)

# Button to toggle streaming
if st.button("Start/Stop Audio Streaming"):
    st.session_state.is_streaming_active = not st.session_state.is_streaming_active
    st.session_state.llm_response_stream = "" # Clear previous response on new start/stop
    st.session_state.partial_transcript = "" # Clear transcript
    st.rerun() # Rerun to apply the desired_playing_state change

# Display current status
if st.session_state.is_streaming_active:
    st.info("Streaming audio in progress... Speak now!")
else:
    st.warning("Audio streaming is stopped.")
    # Removed the problematic line here.
    # We can infer readiness by checking if it's NOT playing and not active.
    if not webrtc_ctx.state.playing and not st.session_state.is_streaming_active:
        st.info("Click 'Start/Stop Audio Streaming' to begin.")

st.markdown("---")

st.subheader("📝 Partial Transcript:")
transcript_placeholder = st.empty()
transcript_placeholder.write(st.session_state.partial_transcript)

st.subheader("🤖 LLM Streaming Response:")
llm_response_placeholder = st.empty()
llm_response_placeholder.write(st.session_state.llm_response_stream)