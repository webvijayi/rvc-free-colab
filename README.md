# rvc-free-colab

Getting the old Rejekts "RVC Train" Colab notebook to actually train again on a current Colab runtime, without paying for anything.

I went to train a voice model with the well-known [ardha27 / Rejekts RVC notebook](https://github.com/ardha27/AI-Song-Cover-RVC) and hit a wall: it was written back when Colab ran an older Python with numba 0.56, numpy 1.23 and fairseq. Colab has since moved to Python 3.12, numpy 2.x, torch 2.x and transformers 5.x, and the notebook now falls over in four different spots before it can finish a single epoch. None of the breakage is in the RVC training itself — it's all dependency rot and a couple of removed library APIs.

This repo is the small set of patches that fix it. It still runs on the free T4. You do not need Colab Pro or compute units.

## How to use it

After the notebook's **Install to Google Drive** cell finishes, add one cell and run it:

```python
!pip -q install av praat-parselmouth pyworld faiss-cpu transformers soundfile pyngrok
!wget -q https://raw.githubusercontent.com/webvijayi/rvc-free-colab/main/apply_fixes.py
!python apply_fixes.py /content/drive/MyDrive/project-main
```

Then run the rest in order: Preprocess, Extract Features, Train Index, Train Model. Skip the "Open the GUI" cell (more on that below). That's the whole change.

## What was actually broken

**Audio loading.** The installer quietly dies at `numba==0.56.4` (it won't build on 3.12), so `av` never gets installed and preprocessing throws `ModuleNotFoundError: av`. Install `av` and you hit the next thing: modern PyAV changed its API, so the notebook's `av.open(file, 'rb')` and `add_stream(channels=1)` calls fail too. Rather than chase the PyAV version, I rewrote `load_audio()` to shell out to ffmpeg, which is already on every Colab box and doesn't care about library churn.

**Feature extraction.** This one loads HuBERT through fairseq, and fairseq simply will not pip-install on Python 3.12 anymore (the omegaconf/hydra pins conflict and the build fails). The maintained RVC forks all dropped it, so I did the same: load the same weights through HuggingFace transformers using `lengyue233/content-vec-best`. For a v2 model you want `hidden_states[12]` (the 768-dim layer); v1 uses `hidden_states[9]` plus the final projection. One gotcha specific to the Rejekts notebook — it passes the args to `extract_feature_print.py` in a different order than upstream RVC, which made it read the experiment folder as `"0"`, so the replacement figures out the path and the v1/v2 flag from the args directly instead of by position.

**Training crashes after epoch 1.** Losses log fine, then it dies in `utils.plot_spectrogram_to_numpy` when it tries to draw the TensorBoard preview. The culprit is `np.fromstring(fig.canvas.tostring_rgb())` — `np.fromstring` is gone in numpy 2.x and `tostring_rgb()` was removed in matplotlib 3.8. Swapped both for `np.frombuffer(fig.canvas.buffer_rgba())` and dropped the alpha channel.

**Missing packages.** `pyworld` and `praat-parselmouth` are also casualties of the aborted installer — they're in the pip line above.

## One thing to know about Colab

Colab blocks the RVC Gradio web UI on the free tier. If you launch it you'll get "runtime disconnected — disallowed usage." That restriction is about the interactive UI and tunnel, not the training. The headless training cells run fine, so just don't open the GUI cell. Losses print to the cell and to `logs/<model>/train.log`, and TensorBoard over ngrok is optional. If you do get disconnected, reconnect and remount Drive — your project and progress are on Drive.

Heads up: training writes a fair amount to Drive (checkpoints, the spectrogram cache). If your Drive is near full you'll see truncated-file errors mid-training, so clear some space first or point the project at Colab's local disk.

## What's in here

`apply_fixes.py` does all four patches in place against a downloaded `project-main`, and it's safe to re-run. The replacement `extract_feature_print.py` is embedded inside it. There are no model weights and no audio in this repo — just the code and notes.

## Credits

- The original notebook: [ardha27/AI-Song-Cover-RVC](https://github.com/ardha27/AI-Song-Cover-RVC)
- [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- HuBERT weights: [lengyue233/content-vec-best](https://huggingface.co/lengyue233/content-vec-best)
- transformers-based RVC references I leaned on: [esnya/hf-rvc](https://github.com/esnya/hf-rvc) and [Applio](https://github.com/IAHispano/Applio)

MIT licensed.
