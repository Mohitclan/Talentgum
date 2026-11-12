# Annotating the class recordings on an AWS GPU

Goal: run `ras segment` over all class recordings (about 600 × 45-60 min) on one GPU instance and get, for every
class, a JSON with each song attempt (song, start, end, confidence) plus one audio clip per attempt.
The script is `scripts/segment_batch.py`: several recordings at once on one GPU, skips finished ones, writes
`clips.csv` at the end. Method and output format: [CLASS_RECORDINGS.md](CLASS_RECORDINGS.md).

Measured so far: on an Intel Mac CPU one 53 min class took 40 min. The GPU time is **not measured yet**;
step 4 measures it on one class before the full run.

## 1. Check the instance

```bash
nvidia-smi                 # GPU model, driver, memory. No output = no NVIDIA driver: use a Deep Learning AMI
df -h /                    # free disk: videos are ~200 MB each (600 → ~120 GB); audio only is ~4x smaller
nproc; free -g
```

Use an Ubuntu Deep Learning AMI (driver already installed). Give the root/EBS volume enough room for the
inputs plus ~10 % for outputs.

## 2. Put the code, songs and recordings on the instance

From the laptop (replace `KEY.pem` and `ubuntu@HOST`):

```bash
cd "/Users/adminpsi/Desktop/Talent Gum"
rsync -av --exclude .venv --exclude 'data/private/videos' --exclude 'data/private/clips' --exclude node_modules \
  -e "ssh -i KEY.pem" Talentgum/ ubuntu@HOST:~/Talentgum/
```

`data/private/songs_segment/` (the six songs, file name = song id) goes along with it. It is git-ignored, so a
`git clone` alone would miss it.

Recordings, laid out as one folder per student: `classes/<student>/<class>.mp4`.

- **Already in S3:** on the instance `aws s3 sync s3://BUCKET/classes /data/classes`
  (the instance role needs read access to the bucket).
- **On a laptop/drive:** upload only the audio, which is ~4x smaller and all the script needs:

  ```bash
  # on the laptop: copy the audio track out of every video, keeping the folder layout
  cd /path/to/classes && find . -name '*.mp4' | while read -r f; do
    mkdir -p "../classes_audio/$(dirname "$f")"
    ffmpeg -nostdin -v error -n -i "$f" -vn -c:a copy "../classes_audio/${f%.mp4}.m4a"; done
  rsync -av -e "ssh -i KEY.pem" ../classes_audio/ ubuntu@HOST:/data/classes/
  ```

## 3. Install (on the instance)

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 tmux
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env   # gets Python 3.12 (the AMI's is too old/new)
cd ~/Talentgum
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.2.2" "torchaudio==2.2.2" --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -e ".[transkun]"
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True
```

(Alternative: the repo's Docker image, `docker build -t ras-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .`,
then run the commands below with `docker run --gpus all -e RAS_DEVICE=cuda -v ~/Talentgum:/work -v /data:/data -w /work ras-gpu python ...`.)

## 4. Time one class first

```bash
cd ~/Talentgum
RAS_DEVICE=cuda .venv/bin/python scripts/segment_batch.py /data/classes \
  --songs data/private/songs_segment --out /data/out --workers 1 --limit 1
```

The last line gives seconds per audio minute. Total time ≈ that × 30 000 (600 classes × ~50 min) ÷ workers'
speed-up. While it runs, `watch -n2 nvidia-smi` in a second terminal shows GPU use and memory.

## 5. Full run

```bash
tmux new -s seg          # keeps running if the SSH connection drops (detach: Ctrl-b d, back: tmux a -t seg)
RAS_DEVICE=cuda .venv/bin/python scripts/segment_batch.py /data/classes \
  --songs data/private/songs_segment --out /data/out --workers 3 2>&1 | tee /data/segment.log
```

- `--workers`: start at 2-3. Raise it while GPU use (`nvidia-smi`) is well under 90 % and GPU memory and RAM have room;
  lower it on "CUDA out of memory". Each worker also needs ~2-3 GB of RAM.
- Stopped or crashed? Run the same command again: finished classes are skipped and saved transcripts are reused.
  Failed files are in `/data/out/failures.csv` and are retried on the next run.
- `--no-clips` writes only the JSON (no audio clips) if disk is tight; clips can be cut later from the JSON.
- Student folder names are replaced by hashed ids; `--keep-names` keeps them.

## 6. Bring the results back and stop paying

```bash
rsync -av -e "ssh -i KEY.pem" ubuntu@HOST:/data/out/ "/Users/adminpsi/Desktop/Talent Gum/Talentgum/data/private/classes_out/"
```

Then **stop or terminate the instance** and delete the uploaded recordings from it: they are children's
classes (see Privacy in CLASS_RECORDINGS.md).

## 7. What is left for people

The GPU run replaces listening through 600 hours. People still do two small jobs:

1. **Check the flagged clips.** `clips.csv` lists every clip; rows with `status = needs_review` (and the reason)
   need a teacher to confirm the song or reject it. On the first sample 12 of 15 clips were flagged, so expect
   this share to drop only after step 2.
2. **Label 5-10 classes by hand** (song, start, end of each attempt) and score the splitter with
   `ras segment-eval /data/out/<student_id>/<class>.json labels.json`. Use the result to tune the thresholds in
   `SegmentConfig` (`ras/segment/find.py`), then re-run: transcripts are reused, so a re-run takes minutes, not hours
   (delete the class JSONs first so they are not skipped, keep the `.notes.json` files).
