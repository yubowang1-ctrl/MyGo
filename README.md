# MyGO

## Download AudioSet Database
To download, use the script `scripts/download_audio_segments.py`. This script downloads audio segments from YouTube based on the AudioSet dataset. 

Example:

```bash
python3 scripts/download_audio_segments.py \
    --csvfile "data/audioset/unbalanced_train_segments_batch_0.csv" \
    --workers 42 \
    --skip-header \
    --outdir "external_drive/audioset/unbalanced_train_segments/"
```

You can terminate the script at any time (Ctrl+C), and it will save progress to a stats file. When you rerun the script with the same parameters, it will resume from where it left off. The stats file is saved in the output directory with the name `.stats.json`.