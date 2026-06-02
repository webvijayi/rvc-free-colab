# Train RVC on Modal (free GPU, no Colab time limits)

Colab's free tier is great until it isn't: the ~90-minute idle disconnect, the
~12-hour session cap, and the Gradio UI being blocked all get in the way of a
long training run. [Modal](https://modal.com) is a serverless-GPU service with a
monthly free credit that sidesteps all of that — the job runs server-side, so
you can close your laptop and it keeps going.

The nice part: the whole reason the Colab notebook needed patching (fairseq won't
build, numpy 2.x breakage, swapping HuBERT to transformers) is that **Colab pins
you to Python 3.12**. On Modal you pick the interpreter. Pin **Python 3.10** and
stock RVC-Project + fairseq + `hubert_base.pt` install and run with no patches.

The full app is [`modal/train_rvc_modal.py`](modal/train_rvc_modal.py). It runs
the stock RVC pipeline (preprocess → extract f0 + features → train → index) and
writes the finished model + index to a Modal volume.

## Setup (one time)

```bash
pip install modal
modal token new          # opens a browser; sign in (Google/GitHub) and approve
modal volume create rvc-data
modal volume put rvc-data ./my_dataset.zip dataset.zip
```

`my_dataset.zip` is a flat zip of clean vocal `.wav` files (UVR-isolated lead
vocals give the best timbre). Edit the config block at the top of the app —
`MODEL`, `SR` (`40k` or `48k`), `TOTAL_EPOCH`, `BATCH`, `GPU` — then:

```bash
modal run --detach modal/train_rvc_modal.py
```

It prints a spawned call id and a dashboard URL, then returns. Training continues
server-side. When it finishes:

```bash
modal volume get rvc-out 'MyVoice*.pth' ./      # final inference weight
modal volume get rvc-out 'added_*.index' ./     # retrieval index
```

## The Modal-specific gotchas (why the app looks the way it does)

None of these are in `apply_fixes.py` because that script is for Colab. They show
up specifically when you build an RVC image on Modal and run training headless:

1. **Downgrade pip before fairseq.** fairseq 0.12.2 depends on `omegaconf==2.0.6`
   and `hydra-core==1.0.7`, whose legacy metadata modern pip rejects. Downgrade
   pip to `23.3.2` in its **own** build step so it's already old when those deps
   resolve. Doing it in the same `pip install` line as the packages is too late.

2. **`apt install clang`.** Modal's standalone Python is built with clang, so
   distutils calls `clang++` to compile fairseq's C++ extensions. With only
   `build-essential` (g++) you get
   `command 'clang++' failed: No such file or directory`.

3. **`numpy<1.24` first.** fairseq 0.12.2 needs it to build its wheel.

4. **`--detach` *and* `.spawn()` together.** This is the one that actually keeps
   training alive after you disconnect. `.remote()` in a detached app gets
   canceled when the local caller drops; `.spawn()` in a non-detached (ephemeral)
   app gets torn down when the run ends. You need both: launch with
   `modal run --detach` and call `train.spawn()` in the entrypoint.

5. **Generate `filelist.txt` + `config.json` yourself.** `train.py` reads both
   from `logs/<model>/`, but it's the Gradio "Train" button that normally writes
   them. Running headless you have to build the filelist (pairing
   `0_gt_wavs` / `3_feature768` / `2a_f0` / `2b-f0nsf`, plus two `mute` rows) and
   copy the matching `configs/v2/<sr>.json`. See `_build_filelist_and_config()`.

6. **matplotlib ≥ 3.8 plot fix.** Same one as Colab — `tostring_rgb()` is gone,
   so RVC crashes at the first TensorBoard spectrogram plot. `_patch_plot_utils()`
   swaps it to `buffer_rgba()[:, :, :3]`.

## Cost

A ~35-minute dataset trains to 200 epochs in roughly 5–6 hours on a T4 (about
$3–4 of the monthly free credit). An A10G is ~2× faster for ~2× the price, so
total cost is similar — pick T4 to stretch the credit, A10G if you're impatient.

## Resuming

The app caches preprocess/extract output to the `rvc-out` volume and skips those
steps on a restart, so re-running after an interruption goes straight back to
training instead of redoing the slow prep.
