#!/usr/bin/env python3
"""Download audio segments from YouTube using yt-dlp with parallel workers and progress tracking.

CSV format expected (per line):
  video_id,start_time,end_time,label

Features:
  - Parallel downloads using multiple yt-dlp workers
  - Progress saved to JSON file (resumable)
  - Error classification: unavailable videos vs rate-limit errors
  - Tracks: downloaded, failed, unavailable, pending, and retried videos
  - On startup, automatically resumes from last checkpoint
  - On exit, prints session statistics

Usage:
  python3 scripts/download_audio_segments_v2.py path/to/file.csv --outdir downloads --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import typing as t
from dataclasses import dataclass
from multiprocessing import Pool, Manager, cpu_count, Lock
from pathlib import Path

import tqdm


@dataclass
class RowRecord:
    """Represents a single CSV row."""
    video_id: str
    start_time: str
    end_time: str
    label: str


@dataclass
class DownloadStats:
    """Statistics for a download session."""
    processed: int = 0
    downloaded: int = 0
    failed: int = 0
    unavailable: int = 0
    pending: int = 0
    retried: int = 0


class ProgressTracker:
    """Manages progress state saved to JSON."""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.state = {
            "downloaded": [],
            "failed": [],
            "unavailable": [],
            "pending": [],
        }
        self.load()

    def load(self) -> None:
        """Load progress from file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as fh:
                    self.state = json.load(fh)
                print(f"Loaded progress from {self.progress_file}")
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")

    def save(self) -> None:
        """Save progress to file."""
        try:
            with open(self.progress_file, "w") as fh:
                json.dump(self.state, fh, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress file: {e}")

    def mark_downloaded(self, vid: str) -> None:
        """Mark video as successfully downloaded."""
        self.state["downloaded"].append(vid)
        self._remove_from_others(vid)
        self.save()

    def mark_unavailable(self, vid: str) -> None:
        """Mark video as unavailable (404, private, etc.)."""
        self.state["unavailable"].append(vid)
        self._remove_from_others(vid)
        self.save()

    def mark_failed(self, vid: str) -> None:
        """Mark video as failed (will retry later)."""
        if vid not in self.state["failed"]:
            self.state["failed"].append(vid)
        self._remove_from_others(vid)
        self.save()

    def mark_pending(self, vid: str) -> None:
        """Mark video as pending (not yet processed)."""
        if vid not in self.state["pending"]:
            self.state["pending"].append(vid)
        self._remove_from_others(vid)
        self.save()

    def _remove_from_others(self, vid: str) -> None:
        """Remove video from other lists."""
        for key in ["downloaded", "failed", "unavailable", "pending"]:
            if isinstance(self.state[key], list):
                self.state[key] = [v for v in self.state[key] if v != vid]

    def is_downloaded(self, vid: str) -> bool:
        return vid in self.state["downloaded"]

    def is_unavailable(self, vid: str) -> bool:
        return vid in self.state["unavailable"]

    def is_failed(self, vid: str) -> bool:
        return vid in self.state["failed"]

    def get_pending(self) -> t.List[str]:
        return list(self.state.get("pending", []))

    def get_stats(self) -> DownloadStats:
        return DownloadStats(
            downloaded=len(self.state["downloaded"]),
            unavailable=len(self.state["unavailable"]),
            failed=len(self.state["failed"]),
            pending=len(self.state["pending"]),
        )


class RateLimitErrorLog:
    """Logs rate-limit errors separately for retry."""

    def __init__(self, error_file: str):
        self.error_file = error_file
        self.errors = {}
        self.load()

    def load(self) -> None:
        """Load error log from file."""
        if os.path.exists(self.error_file):
            try:
                with open(self.error_file, "r") as fh:
                    self.errors = json.load(fh)
            except Exception as e:
                print(f"Warning: Could not load error log: {e}")

    def save(self) -> None:
        """Save error log to file."""
        try:
            with open(self.error_file, "w") as fh:
                json.dump(self.errors, fh, indent=2)
        except Exception as e:
            print(f"Warning: Could not save error log: {e}")

    def log_rate_limit(self, vid: str, error_msg: str) -> None:
        """Log a rate-limit or transient error."""
        self.errors[vid] = {
            "error": error_msg,
            "timestamp": time.time(),
        }
        self.save()

    def clear_rate_limit(self, vid: str) -> None:
        """Clear rate-limit error for a video."""
        if vid in self.errors:
            del self.errors[vid]
            self.save()


def normalize_time(ts: str) -> str:
    """Convert time to a consistent format."""
    ts = ts.strip()
    if not ts:
        return ts
    if ":" in ts:
        return ts
    try:
        f = float(ts)
        return str(int(f))
    except Exception:
        return ts


def build_url(video_id: str) -> str:
    """Build a YouTube URL from video ID or accept full URL."""
    video_id = video_id.strip()
    if video_id.startswith("http://") or video_id.startswith("https://"):
        return video_id
    if "youtube.com" in video_id or "youtu.be" in video_id:
        return video_id
    return f"https://www.youtube.com/watch?v={video_id}"


def classify_error(stderr: str) -> t.Tuple[str, str]:
    """Classify error as 'unavailable' or 'rate_limit' and return (category, reason)."""
    stderr_lower = stderr.lower()

    unavailable_keywords = [
        "404",
        "video not found",
        "private video",
        "age restricted",
        "not available",
        "removed",
        "deleted",
    ]
    for kw in unavailable_keywords:
        if kw in stderr_lower:
            return "unavailable", f"Video unavailable: {kw}"

    rate_limit_keywords = [
        "429",
        "too many requests",
        "rate limit",
        "throttled",
        "connection reset",
        "temporarily unavailable",
    ]
    for kw in rate_limit_keywords:
        if kw in stderr_lower:
            return "rate_limit", f"Rate limit error: {kw}"

    # Default: treat unknown error as transient
    return "rate_limit", f"Transient error"


def load_stats_from_disk(stats_file: str) -> t.Dict[str, t.List[str]]:
    """Load stats_dict (status -> list of video_ids) from disk."""
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r") as fh:
                data = json.load(fh)
            print(f"Loaded stats from {stats_file}")
            return {
                "downloaded": set(data.get("downloaded", set())),
                "failed": set(data.get("failed", set())),
                "unavailable": set(data.get("unavailable", set())),
            }
        except Exception as e:
            print(f"Warning: Could not load stats file: {e}")

    return {"downloaded": set(), "failed": set(), "unavailable": set()}


def split_into_batches(
    rows: t.List[RowRecord],
    outdir: str,
    num_batches: int,
    original_csvfile: str,
) -> None:
    """Split rows into num_batches equal CSV files and save them.
    
    Batch files are named: {outdir}/batch_0.csv, batch_1.csv, ...
    """
    if num_batches <= 0:
        print(f"Error: --split-batches must be > 0")
        return
    
    if len(rows) == 0:
        print("No rows to split.")
        return
    
    # Calculate batch size
    batch_size = (len(rows) + num_batches - 1) // num_batches  # Ceiling division
    
    os.makedirs(outdir, exist_ok=True)
    
    # Split and save batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(rows))
        batch_rows = rows[start_idx:end_idx]
        
        if not batch_rows:
            continue
        
        batch_file = os.path.join(outdir, f"batch_{batch_idx}.csv")
        with open(batch_file, "w", newline="") as fh:
            writer = csv.writer(fh)
            # Write header
            writer.writerow(["video_id", "start_time", "end_time", "label"])
            # Write rows
            for row in tqdm.tqdm(batch_rows, desc=f"Writing batch {batch_idx}"):
                # Remove surrounding quotes from label if present
                clean_label = row.label.strip()
                if clean_label.startswith('"') and clean_label.endswith('"'):
                    clean_label = clean_label[1:-1]
                writer.writerow([row.video_id, row.start_time, row.end_time, clean_label])
        
        print(f"✓ Batch {batch_idx}: {len(batch_rows)} videos → {batch_file}")
    
    print(f"\nTotal: {len(rows)} videos split into {num_batches} batches")


def download_worker(
    row: RowRecord,
    outdir: str,
    ytdlp_path: str,
    extra_args: t.List[str],
    dry_run: bool,
    stats_dict: t.Dict,
    stats_lock: t.Any,
) -> t.Tuple[str, str, str]:
    """Download a single audio segment. Returns (video_id, status, message).
    Status is one of: 'success', 'unavailable', 'rate_limit'.
    Updates shared stats_dict atomically with lock, appending video_id to appropriate list.
    After successful download, re-encodes audio to 64kbps using ffmpeg.
    """
    vid = row.video_id.strip()
    start = row.start_time.strip()
    end = row.end_time.strip()

    if not vid or not start or not end:
        return vid, "failed", "Incomplete row"

    start_n = normalize_time(start)
    end_n = normalize_time(end)
    section = f"*{start_n}-{end_n}"
    url = build_url(vid)
    out_template = os.path.join(outdir, "%(id)s.%(ext)s")

    cmd = [
        ytdlp_path,
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "m4a",
        "--download-sections",
        section,
        "-o",
        out_template,
    ]
    if extra_args:
        cmd += extra_args
    cmd += [url]

    if dry_run:
        print("DRY-RUN:", " ".join(shlex.quote(a) for a in cmd))
        with stats_lock:
            if vid not in stats_dict["downloaded"]:
                stats_dict["downloaded"].append(vid)
        return vid, "success", "Dry-run completed"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Successfully downloaded; now re-encode to 64kbps
            audio_file = os.path.join(outdir, f"{vid}.m4a")
            if os.path.exists(audio_file):
                temp_file = os.path.join(outdir, f"{vid}_temp.m4a")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    audio_file,
                    "-b:a",
                    "64k",
                    "-y",
                    temp_file,
                ]
                try:
                    ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
                    if ffmpeg_result.returncode == 0:
                        # Replace original with bitrate-limited version
                        os.replace(temp_file, audio_file)
                    else:
                        # Ffmpeg failed, but yt-dlp download succeeded; keep original
                        print(f"⚠ {vid}: ffmpeg bitrate conversion failed, keeping original")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                except subprocess.TimeoutExpired:
                    print(f"⚠ {vid}: ffmpeg timeout, keeping original download")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            
            with stats_lock:
                if vid not in stats_dict["downloaded"]:
                    stats_dict["downloaded"].append(vid)
            return vid, "success", "Downloaded and converted to 64kbps"
        else:
            # Classify the error
            stderr = result.stderr + result.stdout
            category, reason = classify_error(stderr)
            with stats_lock:
                if category == "unavailable":
                    if vid not in stats_dict["unavailable"]:
                        stats_dict["unavailable"].append(vid)
                else:
                    if vid not in stats_dict["failed"]:
                        stats_dict["failed"].append(vid)
            return vid, category, reason
    except subprocess.TimeoutExpired:
        with stats_lock:
            if vid not in stats_dict["failed"]:
                stats_dict["failed"].append(vid)
        return vid, "rate_limit", "Timeout (likely rate-limited)"
    except Exception as e:
        with stats_lock:
            if vid not in stats_dict["failed"]:
                stats_dict["failed"].append(vid)
        return vid, "rate_limit", f"Exception: {str(e)}"
    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download audio segments from YouTube with parallel workers and progress tracking."
    )
    parser.add_argument(
        "--csvfile",
        nargs="?",
        help="Path to CSV file",
    )
    parser.add_argument("--outdir", "-o", default="downloads", help="Output directory")
    parser.add_argument("--ytdlp", default="yt-dlp", help="Path to yt-dlp executable")
    parser.add_argument("--skip-header", action="store_true", help="Skip first CSV header row")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=f"Number of parallel yt-dlp workers (default 2, max {cpu_count()})",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument to append to each yt-dlp call",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between worker batches (default 0.1)",
    )
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Stats file to save/load session stats (default: {outdir}/.stats.json)",
    )
    parser.add_argument(
        "--split-batches",
        type=int,
        default=None,
        help="Split pending rows into N equal CSV batch files (no download, just split)",
    )
    

    args = parser.parse_args()

    # Validate workers
    max_workers = cpu_count() or 2
    # workers = max(1, min(args.workers, max_workers))
    workers = max(1, args.workers)
    if args.workers > max_workers:
        print(f"Warning: --workers={args.workers} exceeds CPU count ({max_workers}), using {workers}")

    os.makedirs(args.outdir, exist_ok=True)

    stats_file = args.stats_file or os.path.join(args.outdir, ".stats.json")

    # Load persisted stats from disk
    persisted_stats = load_stats_from_disk(stats_file)
    rows: t.List[RowRecord] = []
    with open(args.csvfile, newline="") as fh:
        reader = csv.reader(fh)
        if args.skip_header:
            next(reader, None)
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) >= 3:
                # Handle labels that contain commas (they may have been split by csv.reader)
                # Expected format: [video_id, start_time, end_time, label, ...]
                # If more than 4 fields, fields[3:] are parts of the label
                video_id = row[0]
                start_time = row[1]
                end_time = row[2]
                label = ",".join(row[3:]) if len(row) > 3 else ""
                rows.append(RowRecord(video_id, start_time, end_time, label))

    # Filter: skip downloaded and unavailable; only process new videos and rate-limited retries
    rows_to_process: t.List[RowRecord] = []
    total_num_rows = len(rows)
    for row in rows:
        # Skip if already downloaded
        if row.video_id in persisted_stats["downloaded"]:
            continue
        # Skip if marked unavailable
        if row.video_id in persisted_stats["unavailable"]:
            continue
        # Include: either new or failed (rate-limit retry)
        rows_to_process.append(row)

    if not rows_to_process:
        print("No rows to process (all already handled).")
        return

    # Handle --split-batches mode: split pending rows into N CSV files and exit
    if args.split_batches:
        split_into_batches(rows_to_process, args.outdir, args.split_batches, args.csvfile)
        return

    print(f"Processing {len(rows_to_process)} video(s) with {workers} worker(s)...")
    print(f"  - Downloaded so far: {len(persisted_stats['downloaded'])}")
    print(f"  - Unavailable: {len(persisted_stats['unavailable'])}")
    print(f"  - Will retry (failed): {len([v for v in rows_to_process if v.video_id in persisted_stats['failed']])}")
    
    stats = DownloadStats(processed=len(rows_to_process))

    # Create shared state for worker processes
    manager = Manager()
    stats_dict = manager.dict()
    stats_dict["downloaded"] = manager.list(persisted_stats["downloaded"])
    stats_dict["failed"] = manager.list(persisted_stats["failed"])
    stats_dict["unavailable"] = manager.list(persisted_stats["unavailable"])
    stats_lock = manager.Lock()

    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        # Convert manager.list to regular list for serialization
        final_stats = {
            "downloaded": list(stats_dict["downloaded"]),
            "failed": list(stats_dict["failed"]),
            "unavailable": list(stats_dict["unavailable"]),
        }
        save_stats_to_disk(stats_file, final_stats)
        
        print("\n\nInterrupted by user.")
        print_session_summary(
            stats.processed,
            len(stats_dict["downloaded"]),
            len(stats_dict["unavailable"]),
            len(stats_dict["failed"]),
            stats.processed - len(stats_dict["downloaded"]) - len(stats_dict["unavailable"]) - len(stats_dict["failed"]),
        )
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Process rows with worker pool
    try:
        with Pool(processes=workers) as pool:
            results = []
            for i, row in enumerate(rows_to_process):
                result = pool.apply_async(
                    download_worker,
                    (row, args.outdir, args.ytdlp, args.extra_arg, args.dry_run, stats_dict, stats_lock),
                )
                results.append((row, result))

                # Add delay between batches
                if args.delay and (i + 1) % workers == 0:
                    time.sleep(args.delay)

            # Collect results (stats already updated by workers via shared dict)
            for row, result in results:
                try:
                    vid, status, message = result.get(timeout=600)
                    if status == "success":
                        print(f"✓ {vid}: {message}")
                    elif status == "unavailable":
                        print(f"✗ {vid}: {message} (unavailable)")
                    elif status == "rate_limit":
                        print(f"⚠ {vid}: {message} (will retry)")
                except KeyboardInterrupt:
                    # Re-raise to trigger signal handler
                    raise
                except Exception as e:
                    print(f"⚠ {row.video_id}: Exception during download: {e}")

    except KeyboardInterrupt:
        # Convert manager.list to regular list for serialization
        final_stats = {
            "downloaded": list(stats_dict["downloaded"]),
            "failed": list(stats_dict["failed"]),
            "unavailable": list(stats_dict["unavailable"]),
        }
        save_stats_to_disk(stats_file, final_stats)
        
        print("\n\nInterrupted by user.")
        print_session_summary(
            stats.processed,
            len(stats_dict["downloaded"]),
            len(stats_dict["unavailable"]),
            len(stats_dict["failed"]),
            total_num_rows - len(stats_dict["downloaded"]) - len(stats_dict["unavailable"]) - len(stats_dict["failed"]),
        )
        sys.exit(0)

    # Final summary and save stats to disk
    final_stats = {
        "downloaded": list(stats_dict["downloaded"]),
        "failed": list(stats_dict["failed"]),
        "unavailable": list(stats_dict["unavailable"]),
    }
    save_stats_to_disk(stats_file, final_stats)
    
    print_session_summary(
        stats.processed,
        len(stats_dict["downloaded"]),
        len(stats_dict["unavailable"]),
        len(stats_dict["failed"]),
        stats.processed - len(stats_dict["downloaded"]) - len(stats_dict["unavailable"]) - len(stats_dict["failed"]),
    )


def print_session_summary(
    total: int,
    downloaded: int,
    unavailable: int,
    failed: int,
    pending: int,
) -> None:
    """Print session statistics."""
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Total processed:     {total}")
    print(f"Downloaded:          {downloaded}")
    print(f"Unavailable:         {unavailable}")
    print(f"Failed (will retry): {failed}")
    print(f"Pending:             {pending}")
    print("=" * 60)


def save_stats_to_disk(stats_file: str, stats_dict: t.Dict[str, t.List[str]]) -> None:
    """Save stats_dict to disk as JSON."""
    try:
        with open(stats_file, "w") as fh:
            json.dump(stats_dict, fh, indent=2)
        print(f"\nSaved stats to {stats_file}")
    except Exception as e:
        print(f"Warning: Could not save stats file: {e}")


if __name__ == "__main__":
    main()
