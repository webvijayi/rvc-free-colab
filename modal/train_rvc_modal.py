"""
Train an RVC v2 voice model on a Modal.com cloud GPU — a free-GPU alternative to
Colab with much longer sessions (no ~90min idle disconnect, no ~12h cap).

The big idea: the entire Colab dependency saga (fairseq won't build, numpy 2.x
breakage, the transformers HuBERT swap) only exists because Colab pins you to
Python 3.12. On Modal YOU choose the interpreter, so we pin **Python 3.10** and
stock RVC-Project + fairseq + hubert_base.pt install and run UNMODIFIED.

What still needs handling on Modal (these are the non-obvious bits):
  1. Downgrade pip to 23.3.2 in its OWN build step BEFORE installing fairseq —
     modern pip rejects fairseq's deps (omegaconf 2.0.6 / hydra 1.0.7) legacy
     metadata, and it must already be downgraded when those are resolved.
  2. apt-install `clang` — Modal's standalone Python is clang-built, so distutils
     invokes clang++ to compile fairseq's C++ extensions (build-essential's g++
     alone gives: "command 'clang++' failed: No such file or directory").
  3. Build numpy<1.24 first so fairseq 0.12.2 compiles its wheel.
  4. Run as `modal run --detach` with `.spawn()` (see bottom) so the job runs
     fully server-side and survives your laptop disconnecting. `.remote()` in a
     detached app gets canceled on disconnect; `.spawn()` in a non-detached
     ephemeral app gets torn down. You need BOTH --detach AND .spawn().
  5. train.py reads logs/<exp>/filelist.txt and logs/<exp>/config.json, which the
     Gradio "Train" button normally generates. Running train.py headless you must
     write them yourself — see _build_filelist_and_config().
  6. matplotlib >= 3.8 removed FigureCanvasAgg.tostring_rgb(), so RVC crashes at
     the first TensorBoard plot. _patch_plot_utils() swaps it to buffer_rgba().

Setup (one time):
    pip install modal
    modal token new                       # sign in (Google/GitHub), approve
    modal volume create rvc-data
    modal volume put rvc-data ./my_dataset.zip dataset.zip   # your vocals zip

Run:
    modal run --detach modal/train_rvc_modal.py

Collect the result when it finishes:
    modal volume get rvc-out <ModelName>.pth ./        # final inference weight
    modal volume get rvc-out 'added_*.index' ./        # retrieval index

The dataset zip = a flat zip of clean (ideally UVR-isolated) vocal .wav files.
"""
import glob
import os
import shutil
import subprocess

import modal

# ---- configure these ----
MODEL = "MyVoice"        # model/experiment name
SR = "48k"               # "40k" (standard) or "48k" (crisper, needs clean data)
VERSION = "v2"
TOTAL_EPOCH = 200
SAVE_EVERY = 25
BATCH = 8                # 8 for ~30+min data, 4 for less
DATASET_ZIP = "dataset.zip"   # filename you `modal volume put` into rvc-data
GPU = "T4"               # T4 cheapest; A10G ~2x faster, ~2x price
# -------------------------

RVC = "/rvc"
FEAT = "256" if VERSION == "v1" else "768"
HF = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
SR_HZ = {"32k": 32000, "40k": 40000, "48k": 48000}[SR]

app = modal.App("rvc-train")
data_vol = modal.Volume.from_name("rvc-data", create_if_missing=True)
out_vol = modal.Volume.from_name("rvc-out", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    # (2) clang for fairseq C++ ext (Modal's Python is clang-built)
    .apt_install("git", "ffmpeg", "build-essential", "clang", "wget", "aria2")
    .run_commands(
        "pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        f"git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI {RVC}",
    )
    .run_commands(
        # (1) downgrade pip in its OWN step before fairseq deps are resolved
        "pip install 'pip==23.3.2' 'setuptools==69.5.1' wheel",
        # (3) numpy<1.24 so fairseq 0.12.2 builds
        "pip install 'numpy==1.23.5' cython",
        "pip install fairseq==0.12.2",  # pulls compatible omegaconf/hydra itself
        "pip install faiss-cpu==1.7.4 praat-parselmouth pyworld torchcrepe "
        "soundfile librosa==0.10.2 scipy tensorboard tqdm ffmpeg-python einops "
        "'numba==0.58.1' 'llvmlite==0.41.1' av matplotlib resampy",
    )
    .run_commands(  # bake pretrains/helpers into the image (cached across runs)
        f"cd {RVC} && mkdir -p assets/hubert assets/rmvpe assets/pretrained_v2",
        f"aria2c -x4 -o assets/hubert/hubert_base.pt -d {RVC} {HF}/hubert_base.pt",
        f"aria2c -x4 -o assets/rmvpe/rmvpe.pt -d {RVC} {HF}/rmvpe.pt",
        f"aria2c -x4 -o assets/pretrained_{VERSION}/f0G{SR}.pth -d {RVC} {HF}/pretrained_{VERSION}/f0G{SR}.pth",
        f"aria2c -x4 -o assets/pretrained_{VERSION}/f0D{SR}.pth -d {RVC} {HF}/pretrained_{VERSION}/f0D{SR}.pth",
    )
)


def run(cmd, cwd=RVC):
    print(f"\n>>> {cmd}\n", flush=True)
    if subprocess.run(cmd, shell=True, cwd=cwd).returncode != 0:
        raise RuntimeError(f"step failed: {cmd}")


def _patch_plot_utils():
    """(6) matplotlib>=3.8 removed tostring_rgb(); swap to buffer_rgba()[:,:,:3]."""
    p = f"{RVC}/infer/lib/train/utils.py"
    s = open(p).read()
    s = s.replace(
        'np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")',
        "np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)",
    ).replace(
        "data.reshape(fig.canvas.get_width_height()[::-1] + (3,))",
        "data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]",
    )
    open(p, "w").write(s)


def _build_filelist_and_config(exp):
    """(5) train.py reads filelist.txt + config.json; the GUI normally writes them."""
    from random import shuffle

    gt, feat = f"{exp}/0_gt_wavs", f"{exp}/3_feature{FEAT}"
    f0, f0nsf = f"{exp}/2a_f0", f"{exp}/2b-f0nsf"
    names = (
        {n.split(".")[0] for n in os.listdir(gt)}
        & {n.split(".")[0] for n in os.listdir(feat)}
        & {n.split(".")[0] for n in os.listdir(f0)}
        & {n.split(".")[0] for n in os.listdir(f0nsf)}
    )
    opt = [
        f"{gt}/{n}.wav|{feat}/{n}.npy|{f0}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0"
        for n in names
    ]
    mute = f"{RVC}/logs/mute"
    for _ in range(2):
        opt.append(
            f"{mute}/0_gt_wavs/mute{SR}.wav|{mute}/3_feature{FEAT}/mute.npy|"
            f"{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0"
        )
    shuffle(opt)
    open(f"{exp}/filelist.txt", "w").write("\n".join(opt))
    cfg_name = f"v1/{SR}.json" if (VERSION == "v1" or SR == "40k") else f"v2/{SR}.json"
    shutil.copy(f"{RVC}/configs/{cfg_name}", f"{exp}/config.json")


def _train_index(exp):
    import faiss
    import numpy as np

    feat_dir = f"{exp}/3_feature{FEAT}"
    big = np.concatenate(
        [np.load(os.path.join(feat_dir, n)) for n in sorted(os.listdir(feat_dir))], 0
    )
    np.random.shuffle(big)
    n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39)
    index = faiss.index_factory(int(FEAT), f"IVF{n_ivf},Flat")
    index.train(big)
    index.add(big)
    faiss.write_index(index, f"{exp}/added_IVF{n_ivf}_Flat_nprobe_1_{MODEL}_{VERSION}.index")


@app.function(image=image, gpu=GPU, volumes={"/data": data_vol, "/out": out_vol},
              timeout=60 * 60 * 23)
def train():
    ds = "/work/dataset"
    os.makedirs(ds, exist_ok=True)
    if not glob.glob(f"{ds}/*.wav"):
        import zipfile
        with zipfile.ZipFile(f"/data/{DATASET_ZIP}") as z:
            z.extractall(ds)
    assert glob.glob(f"{ds}/*.wav"), "no wavs in dataset zip"

    _patch_plot_utils()
    exp = f"{RVC}/logs/{MODEL}"
    os.makedirs(exp, exist_ok=True)

    # resume: restore prior prep/checkpoints so a restart skips the slow prep
    if os.path.isdir(f"/out/{MODEL}") and os.listdir(f"/out/{MODEL}"):
        run(f"cp -a /out/{MODEL}/. {exp}/", cwd="/")

    def has(sub):
        return os.path.isdir(f"{exp}/{sub}") and os.listdir(f"{exp}/{sub}")

    if not has("0_gt_wavs"):
        run(f"python infer/modules/train/preprocess.py '{ds}' {SR_HZ} 4 '{exp}' False 3.7")
    if not has("2a_f0"):
        run(f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 '{exp}' True")
    if not has(f"3_feature{FEAT}"):
        run(f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 '{exp}' {VERSION} True")

    os.makedirs(f"/out/{MODEL}", exist_ok=True)
    run(f"cp -a {exp}/. /out/{MODEL}/", cwd="/")
    out_vol.commit()

    _build_filelist_and_config(exp)
    run(
        f"python infer/modules/train/train.py -e {MODEL} -sr {SR} -f0 1 "
        f"-bs {BATCH} -g 0 -te {TOTAL_EPOCH} -se {SAVE_EVERY} "
        f"-pg assets/pretrained_{VERSION}/f0G{SR}.pth -pd assets/pretrained_{VERSION}/f0D{SR}.pth "
        f"-l 0 -c 0 -sw 1 -v {VERSION}"
    )
    _train_index(exp)

    run(f"cp -a {exp}/. /out/{MODEL}/", cwd="/")
    for w in glob.glob(f"{RVC}/assets/weights/{MODEL}*.pth") + glob.glob(f"{RVC}/weights/{MODEL}*.pth"):
        shutil.copy(w, f"/out/{os.path.basename(w)}")
    for idx in glob.glob(f"{exp}/added_*.index"):
        shutil.copy(idx, f"/out/{os.path.basename(idx)}")
    out_vol.commit()
    print("DONE. Artifacts on rvc-out volume:", os.listdir("/out"), flush=True)


@app.local_entrypoint()
def main():
    # (4) --detach + .spawn() => runs fully server-side, survives disconnect
    call = train.spawn()
    print(f"SPAWNED call id: {call.object_id} — runs server-side; safe to disconnect.")
