# MyGO

## Download AudioSet Database
To download, use the script `scripts/download_audio_segments.py`. This script downloads audio segments from YouTube based on the AudioSet dataset. 

Sample command:
python3 scripts/download_audio_segments.py 
--csvfile "data/audioset/unbalanced_train_segments_batch_0.csv" # source csv file
--workers 42 # Number of parallel download workers, adjust based on system performance
--skip-header # skip the header row in the csv file
--outdir "external_drive/audioset/unbalanced_train_segments/" # output directory

You can terminate the script at any time (Ctrl+C), and it will save progress to a stats file. When you rerun the script with the same parameters, it will resume from where it left off. The stats file is saved in the output directory with the name `.stats.json`.