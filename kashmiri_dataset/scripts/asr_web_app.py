import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

st.set_page_config(page_title="Kashmiri ASR Studio", page_icon="🎙️", layout="centered")

BASE_DIR = Path("/home/musa/Kashmiri Project")
MODEL_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/best_model"

if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "kashmiri_dataset/experiments/native_asr_training/experiments/B-01-xlsr300m-character-ctc/step_checkpoint"

@st.cache_resource
def load_asr_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL_DIR))
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    return processor, model, device

processor, model, device = load_asr_model()

st.title("🎙️ Kashmiri Native ASR Studio")
st.markdown("Transcribe spoken **Kashmiri (کٲشُر)** in Perso-Arabic script using our fine-tuned **XLS-R 300M + Kashmiri CTC** acoustic model.")

st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Architecture**: XLS-R 300M (Wav2Vec2ForCTC)")
st.sidebar.markdown(f"**Vocabulary**: 161 Kashmiri Character Tokens")
st.sidebar.markdown(f"**Checkpoint**: Epoch 3 (`best_model/`)")
st.sidebar.markdown(f"**Validation Norm WER**: **41.06%**")
st.sidebar.markdown(f"**Validation Norm CER**: **20.32%**")
st.sidebar.markdown(f"**Inference Device**: `{device}`")

def transcribe_audio_bytes(audio_bytes, suffix=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    audio, sr = sf.read(tmp_wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / 16000.0

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits[0]

    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids)

    pad_id = processor.tokenizer.pad_token_id
    blank_ratio = (pred_ids == pad_id).float().mean().item()

    Path(tmp_in_path).unlink(missing_ok=True)
    Path(tmp_wav_path).unlink(missing_ok=True)

    return transcription, duration, blank_ratio

tab1, tab2, tab3 = st.tabs(["🎙️ Browser Mic", "⚡ System Mic (Record Directly)", "📁 Upload Audio File"])

with tab1:
    st.subheader("Record via Browser Microphone")
    st.info("⚠️ If your browser doesn't prompt for microphone permission or fails to record, use the **⚡ System Mic** tab next to this one!")
    audio_input = st.audio_input("Click to record with browser", key="browser_mic")
    if audio_input is not None:
        st.audio(audio_input)
        if st.button("Transcribe Browser Audio", type="primary", key="rec_btn"):
            with st.spinner("Decoding Kashmiri speech..."):
                text, dur, blank = transcribe_audio_bytes(audio_input.read(), suffix=".wav")
                st.success("Transcription Complete!")
                st.markdown(f"""
                <div style="direction: rtl; font-size: 32px; line-height: 1.8; padding: 20px; background-color: #1e1e2e; color: #a6e3a1; border-radius: 10px; font-family: 'Noto Naskh Arabic', sans-serif;">
                    {text if text.strip() else '<i>(No speech detected)</i>'}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Audio Duration: {dur:.2f}s | Confidence Frame Coverage: {(1.0 - blank)*100:.1f}%")

with tab2:
    st.subheader("Record via System Microphone (Direct Hardware Capture)")
    st.write("This records directly from your system microphone (e.g. EPOS ADAPT 360 headset or default mic) without browser security restrictions.")
    
    rec_sec = st.slider("Recording Duration (seconds)", min_value=2, max_value=15, value=5, step=1)
    
    if st.button("🔴 Click Here to Record from System Mic", type="primary", key="sys_rec_btn"):
        rec_placeholder = st.empty()
        rec_placeholder.warning(f"🎙️ RECORDING NOW FOR {rec_sec} SECONDS... Speak Kashmiri!")
        
        output_wav = Path(tempfile.gettempdir()) / "sys_mic_speech.wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse",
            "-i", "default",
            "-t", str(rec_sec),
            "-ar", "16000",
            "-ac", "1",
            str(output_wav)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rec_placeholder.success("✅ Recording complete! Now transcribing...")
        
        with open(output_wav, "rb") as f:
            wav_bytes = f.read()
            
        text, dur, blank = transcribe_audio_bytes(wav_bytes, suffix=".wav")
        st.audio(str(output_wav))
        st.markdown(f"""
        <div style="direction: rtl; font-size: 32px; line-height: 1.8; padding: 20px; background-color: #1e1e2e; color: #a6e3a1; border-radius: 10px; font-family: 'Noto Naskh Arabic', sans-serif;">
            {text if text.strip() else '<i>(No speech detected)</i>'}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Audio Duration: {dur:.2f}s | Confidence Frame Coverage: {(1.0 - blank)*100:.1f}%")

with tab3:
    st.subheader("Upload an existing audio file")
    uploaded_file = st.file_uploader("Upload audio file (WAV, MP3, M4A, OGG, WEBM)", type=["wav", "mp3", "m4a", "ogg", "webm"])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Transcribe Uploaded File", type="primary", key="up_btn"):
            with st.spinner("Decoding audio..."):
                ext = Path(uploaded_file.name).suffix or ".wav"
                text, dur, blank = transcribe_audio_bytes(uploaded_file.read(), suffix=ext)
                st.success("Transcription Complete!")
                st.markdown(f"""
                <div style="direction: rtl; font-size: 32px; line-height: 1.8; padding: 20px; background-color: #1e1e2e; color: #a6e3a1; border-radius: 10px; font-family: 'Noto Naskh Arabic', sans-serif;">
                    {text if text.strip() else '<i>(No speech detected)</i>'}
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Audio Duration: {dur:.2f}s | Confidence Frame Coverage: {(1.0 - blank)*100:.1f}%")
