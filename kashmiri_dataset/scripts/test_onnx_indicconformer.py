import urllib.request
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path("kashmiri_dataset/models/indicconformer_onnx")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

files = ["config.json", "vocab.txt", "model.int8.onnx"]
base_url = "https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ks-onnx/resolve/main/"

for f in files:
    dest = MODEL_DIR / f
    if not dest.exists():
        print(f"Downloading {f}...")
        urllib.request.urlretrieve(base_url + f, dest)
        print(f"Downloaded {f} ({dest.stat().st_size / (1024*1024):.2f} MB)")

print("Initializing ONNX session with model.int8.onnx...")
sess = ort.InferenceSession(str(MODEL_DIR / "model.int8.onnx"))
print("Model Inputs:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
print("Model Outputs:")
for out in sess.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")
