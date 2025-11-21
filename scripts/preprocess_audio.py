#!/usr/bin/env python3
"""
Simple CLI script to load an .m4a audio file and run the TF-based AudioPreprocessor

Usage examples:
  python scripts/preprocess_audio.py audio.m4a --out spec.npy
  python scripts/preprocess_audio.py audio.m4a --sr None --mono False

This script will:
 - load audio using librosa (optionally preserve original sr with --sr None)
 - prepare waveform as 1D (mono) or 2D (channels, time)
 - call `utils.audio_process.AudioPreprocessor` (which resamples to 48000 by default)
 - print the resulting spectrogram shape and optionally save as .npy
"""

import argparse
import sys
from pathlib import Path
import tensorflow as tf
import librosa
import numpy as np

from utils.audio_process import AudioPreprocessor, AudioSTFTConfig


def load_m4a_as_waveform(path, sr=None, mono=True):
    """
    Load .m4a using librosa and return waveform and sample rate.

    - If `mono` is True: returns 1-D numpy array (samples,)
    - If `mono` is False: returns 2-D numpy array (channels, samples)
    - If `sr` is None: preserve original file sample rate
    """
    y, sr_loaded = librosa.load(path, sr=sr, mono=mono)
    y = np.asarray(y, dtype=np.float32)

    # librosa.load with mono=False typically returns shape (n_channels, n_samples)
    if not mono:
        if y.ndim == 1:
            # single channel
            wav = y[np.newaxis, :]
        elif y.ndim == 2:
            # assume (channels, samples) — if it's (samples, channels), transpose
            if y.shape[0] < y.shape[1]:
                wav = y
            else:
                wav = y.T
        else:
            raise ValueError("Unsupported waveform ndim: %s" % (y.ndim,))
    else:
        wav = y.flatten()

    return wav, sr_loaded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="input .m4a file path")
    p.add_argument("--sr", default=None,
                   help="librosa load sr (int) or None to preserve original. Default: None")
    p.add_argument("--mono", action="store_true", help="force mono load (default False)")
    p.add_argument("--out", default=None, help="optional output .npy path to save spectrogram")

    # STFT / preprocessing params (optional overrides)
    p.add_argument("--window_size", type=int, default=None, help="STFT window_size (n_fft/frame_length)")
    p.add_argument("--stride", type=int, default=None, help="STFT stride (frame_step/hop_length)")
    p.add_argument("--fmin", type=float, default=None, help="min frequency to keep (Hz)")
    p.add_argument("--fmax", type=float, default=None, help="max frequency to keep (Hz)")

    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(2)

    # parse sr arg: allow string 'None'
    sr_arg = None if str(args.sr).lower() == "none" else int(args.sr) if args.sr is not None else None

    waveform, sr_loaded = load_m4a_as_waveform(str(inp), sr=sr_arg, mono=args.mono)

    print(f"Loaded: {inp}  sample_rate={sr_loaded}  waveform.shape={waveform.shape}")

    cfg = AudioSTFTConfig(target_sr=48000)
    pre = AudioPreprocessor(cfg)

    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    sample_rate_tf = tf.constant(sr_loaded, dtype=tf.int32)

    spec = pre(waveform_tf,
               sample_rate_tf,
               window_size=args.window_size,
               stride=args.stride,
               fmin=args.fmin,
               fmax=args.fmax)

    # spec is a tf.Tensor of shape [H, W, 2]
    try:
        spec_np = spec.numpy()
    except Exception:
        # If running in graph mode or TF < 2 behavior, evaluate in a session-like way
        spec_np = np.array(spec)

    print("Spec shape:", spec_np.shape)

    if args.out:
        outp = Path(args.out)
        np.save(outp, spec_np)
        print(f"Saved spectrogram to {outp}")


if __name__ == "__main__":
    main()
