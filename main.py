# main.py (FastAPI Backend)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import numpy as np
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Load Groq LLM
# IMPORTANT: Ensure GROQ_API_KEY is set in your environment variables
# You can set it directly in your terminal before running:
# export GROQ_API_KEY="your_groq_api_key_here" (macOS/Linux)
# $env:GROQ_API_KEY="your_groq_api_key_here" (PowerShell)
# set GROQ_API_KEY=your_groq_api_key_here (CMD)
chat_groq = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

# Initialize the Whisper model (for demonstration of batch processing on chunks)
# For a real streaming ASR, you'd use a dedicated streaming ASR client (e.g., Vosk, cloud provider SDK)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8") # Use tiny for faster demo

app = FastAPI()

@app.get("/")
async def get():
    return HTMLResponse("""
        <h1>Real-time Voice-to-LLM Backend</h1>
        <p>Connects via WebSocket from Streamlit.</p>
        <p>Ensure this backend is running before starting the Streamlit app.</p>
    """)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected!")

    audio_buffer = [] # To accumulate audio chunks
    buffer_threshold_seconds = 2 # Process every 2 seconds of audio
    sample_rate = 44100 # Ensure this matches your frontend (or desired ASR rate)

    current_transcript = ""
    llm_context_messages = [SystemMessage(content="You are a helpful AI assistant.")]

    try:
        while True:
            # Receive audio data from the client
            audio_bytes = await websocket.receive_bytes()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_buffer.append(audio_array)

            # Check if enough audio has accumulated for processing
            total_samples = sum(len(chunk) for chunk in audio_buffer)
            if total_samples >= buffer_threshold_seconds * sample_rate:
                combined_audio = np.concatenate(audio_buffer).astype(np.float32) / np.iinfo(np.int16).max
                # Clear buffer for next chunk
                audio_buffer = []

                # --- ASR Processing (simulated streaming with Whisper) ---
                # In a real streaming ASR, you'd feed `audio_bytes` directly
                # to the ASR model's stream interface and get partial results.
                segments, info = whisper_model.transcribe(combined_audio, sample_rate=sample_rate, beam_size=5)
                chunk_transcript = " ".join([seg.text for seg in segments])

                if chunk_transcript.strip():
                    current_transcript += " " + chunk_transcript.strip()
                    await websocket.send_text(json.dumps({
                        "type": "transcript_partial",
                        "text": current_transcript
                    }))
                    print(f"Partial Transcript: {current_transcript}")


                    # --- LLM Processing (Incremental) ---
                    # Only send the latest full transcript to LLM for simplicity.
                    # For complex conversations, you'd manage context more sophisticatedly.
                    llm_context_messages_for_llm = [
                        SystemMessage(content="You are a helpful AI assistant. Provide concise and relevant answers based on the user's latest input. If the user stops talking, wait for them to speak again."),
                        HumanMessage(content=current_transcript)
                    ]

                    # LLM Streaming Response
                    llm_response_generator = chat_groq.stream(llm_context_messages_for_llm)
                    full_llm_response = ""
                    for chunk in llm_response_generator:
                        if chunk.content:
                            full_llm_response += chunk.content
                            await websocket.send_text(json.dumps({
                                "type": "llm_response_chunk",
                                "text": chunk.content
                            }))
                    print(f"LLM Response (Stream): {full_llm_response}")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Closing WebSocket connection.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)