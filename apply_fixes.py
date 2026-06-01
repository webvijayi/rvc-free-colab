#!/usr/bin/env python3
"""
apply_fixes.py — Make the old Rejekts / ardha27 "RVC Train" Colab notebook
(Retrieval-based Voice Conversion v2) run on a MODERN Google Colab runtime
(Python 3.12, numpy 2.x, torch 2.x, transformers 5.x) — for FREE.

The original notebook targets numba 0.56 / numpy 1.23 / fairseq and breaks on
today's Colab. This script applies the four fixes needed, in-place, to a
freshly downloaded `project-main` (the RVC v2 project the notebook unzips to
your Google Drive, e.g. /content/drive/MyDrive/project-main).

Run it ONCE, right after the notebook's "Install to Google Drive" cell, before
Preprocess. It is idempotent.

Usage (in a Colab cell):
    !pip -q install av praat-parselmouth pyworld faiss-cpu transformers soundfile pyngrok
    !python apply_fixes.py /content/drive/MyDrive/project-main

What it fixes:
  1. audio loading: PyAV's API changed (av.open mode 'rb'->'r', add_stream
     kwargs); replaced load_audio() with a version-proof ffmpeg subprocess.
  2. feature extraction: fairseq won't build on py3.12. Loads HuBERT via
     HuggingFace transformers (lengyue233/content-vec-best) instead. Also makes
     argv parsing robust (the Rejekts arg order differs from upstream RVC).
  3. tensorboard summaries: utils.plot_*_to_numpy used the removed
     np.fromstring(fig.canvas.tostring_rgb()); switched to
     np.frombuffer(fig.canvas.buffer_rgba()).
  4. (the pip line above) installs the deps the original installer skipped
     because the numba==0.56.4 build aborts on py3.12.

NOTE: Google Colab BANS the RVC *Gradio GUI* on the free tier ("disallowed
usage"), but the *training cells* run fine. Just don't open the GUI cell.
"""
import io
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/project-main"


def patch(path, fn):
    p = os.path.join(ROOT, path)
    with io.open(p, "r", encoding="utf-8") as f:
        src = f.read()
    new = fn(src)
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(new)
    print("patched:", path)


# --- Fix 1: ffmpeg-based load_audio (append override; last definition wins) ---
FFMPEG_LOADER = '''

def load_audio(file, sr):
    import subprocess, numpy as np
    file = str(file).strip().strip('"')
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file,
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except Exception as e:
        raise RuntimeError("Failed to load audio: " + str(e))
    return np.frombuffer(out, np.float32).flatten()
'''


def fix_audio(src):
    if "ffmpeg" in src and "load_audio" in src and "buffer override" in src:
        return src  # already patched
    return src + "\n# --- modern-colab ffmpeg load_audio override (buffer override) ---" + FFMPEG_LOADER


# --- Fix 3: tensorboard plot helpers (numpy2 / matplotlib3.8+) ---
def fix_utils(src):
    return src.replace(
        'np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")',
        "np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)"
        ".reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].reshape(-1)",
    )


# --- Fix 2: feature extraction via transformers (no fairseq) ---
EXTRACT_FEATURE = r'''import os
import sys
import traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

_A = sys.argv[1:]
device = _A[0]
n_part = int(_A[1])
i_part = int(_A[2])
# robust: find the experiment dir (a path) and the version regardless of arg order
exp_dir = next((x for x in _A if ("log" in x.lower() or "/" in x)), _A[-2])
version = "v1" if "v1" in _A else "v2"
is_half = any(str(x).lower() == "true" for x in _A)

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(" ".join(sys.argv))
printt("exp_dir: " + exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
os.makedirs(outPath, exist_ok=True)


def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# ContentVec / HuBERT loaded the HuggingFace way (replaces fairseq)
class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


printt("load model(s) from lengyue233/content-vec-best")
model = HubertModelWithFinalProj.from_pretrained("lengyue233/content-vec-best")
model = model.to(device)
printt("move model to %s" % device)
if is_half and device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
                if os.path.exists(out_path):
                    continue
                feats = readwave(wav_path, normalize=True)
                feats = (
                    feats.half().to(device)
                    if is_half and device not in ["mps", "cpu"]
                    else feats.to(device)
                )
                if feats.dim() == 1:
                    feats = feats.view(1, -1)
                # fairseq output_layer=N  ==  transformers hidden_states[N]
                # (hidden_states[0] is the embedding output)
                layer = 9 if version == "v1" else 12
                with torch.no_grad():
                    out = model(feats, output_hidden_states=True)
                    x = out.hidden_states[layer]
                    feats = model.final_proj(x) if version == "v1" else x
                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except Exception:
            printt(traceback.format_exc())
    printt("all-feature-done")
'''


def main():
    patch("infer/lib/audio.py", fix_audio)
    patch("infer/lib/train/utils.py", fix_utils)
    # extract_feature_print.py is fully replaced
    p = os.path.join(ROOT, "infer/modules/train/extract_feature_print.py")
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(EXTRACT_FEATURE)
    print("replaced: infer/modules/train/extract_feature_print.py")
    print("\nAll fixes applied. Now run Preprocess -> Extract -> Train Index -> Train Model.")
    print("Do NOT open the Gradio GUI cell (banned on Colab free tier).")


if __name__ == "__main__":
    main()
