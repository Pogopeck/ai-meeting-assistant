from faster_whisper import WhisperModel
from llama_cpp import Llama
import os

# Lazy load models (only once)
_whisper = None
_llm = None

def get_models():
    global _whisper, _llm
    if _whisper is None:
        _whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    if _llm is None:
        model_path = os.getenv("MODEL_PATH", "models/phi-3-mini.Q4_K_M.gguf")
        _llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
    return _whisper, _llm

def transcribe_audio(audio_bytes: bytes) -> str:
    whisper, _ = get_models()
    segments, _ = whisper.transcribe(audio_bytes, language="en")
    return " ".join(seg.text for seg in segments)

def extract_actions(transcript: str) -> str:
    _, llm = get_models()
    prompt = f"""<|system|>Extract clear, concise action items. If none, say 'No action items.'<|end|>
<|user|>Transcript: "{transcript}"<|end|>
<|assistant|>"""
    out = llm(prompt, max_tokens=256, stop=["<|end|>"], echo=False)
    return out["choices"][0]["text"].strip()