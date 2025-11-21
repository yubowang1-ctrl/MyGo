"""
Audio preprocessing utilities.

Functionality (Step 1):

- Time amplitude waveform -> Time-frequency spectrogram

- Unify sampling rate to target_sr (default 48000 Hz)

- STFT + band truncation (default 30~16000 Hz)

- Automatically copy mono to 2 channels

- Output shape [H, W, 2] (freq_bins, time_frames, channels)

"""

from dataclasses import dataclass
from typing import Tuple, Optional

import tensorflow as tf


# ====================== 配置类 ======================

@dataclass
class AudioSTFTConfig:
    target_sr: int = 48000         # 目标采样率
    window_size: int = 2048        # STFT frame_length / n_fft
    stride: int = 512              # STFT frame_step / hop_length
    fmin: float = 30.0             # 最低频率
    fmax: float = 16000.0          # 最高频率
    num_channels: int = 2          # 最终输出通道数（这里固定 2）


# ====================== 工具函数 ======================

def ensure_sample_rate(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor,
    target_sr: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    确保 waveform 的采样率为 target_sr，不一致时使用 tf.signal.resample 重采样。

    参数：
        waveform: [T] 或 [C, T] 的 float32 Tensor
        sample_rate: 标量 int32/int64 Tensor（原始采样率）
        target_sr: int，目标采样率

    返回：
        resampled_waveform: [T'] 或 [C, T']，float32
        target_sr_tensor: 标量 int32（就是 target_sr）
    """
    sample_rate = tf.cast(sample_rate, tf.int32)
    target_sr = tf.cast(target_sr, tf.int32)

    def _same_sr():
        return waveform, target_sr

    def _resample():
        # 使用 tf.cond 安全地处理动态 rank：确保形状为 [C, T]
        def _expand():
            return tf.expand_dims(waveform, 0), tf.constant(1, dtype=tf.int32)

        def _identity():
            return waveform, tf.constant(2, dtype=tf.int32)

        x, orig_rank = tf.cond(
            tf.equal(tf.rank(waveform), 1),
            _expand,
            _identity,
        )

        # 检查 rank 合法性（1 或 2）
        tf.debugging.assert_greater_equal(orig_rank, 1)
        tf.debugging.assert_less_equal(orig_rank, 2)

        orig_len = tf.shape(x)[-1]
        rate_ratio = tf.cast(target_sr, tf.float32) / tf.cast(sample_rate, tf.float32)
        new_len = tf.cast(tf.round(tf.cast(orig_len, tf.float32) * rate_ratio), tf.int32)

        # 沿时间轴重采样
        x_resampled = tf.signal.resample(x, new_len, axis=-1)  # [C, T']

        # 若原本为 rank==1，则 squeeze 回去
        x_resampled = tf.cond(
            tf.equal(orig_rank, 1),
            lambda: tf.squeeze(x_resampled, axis=0),
            lambda: x_resampled,
        )

        return x_resampled, target_sr

    return tf.cond(tf.equal(sample_rate, target_sr), _same_sr, _resample)


# ====================== 核心 STFT 函数 ======================

def stft_spectrogram(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor,
    cfg: AudioSTFTConfig,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    时间振幅 -> 时间-频率 spectrogram（幅度谱）。

    参数：
        waveform: [T] 或 [C, T] 的 float32 Tensor
        sample_rate: 标量 int32/int64 Tensor（原始采样率）
        cfg: AudioSTFTConfig，提供默认参数（target_sr, window_size, stride, fmin, fmax）
        window_size: 可覆盖 cfg.window_size
        stride: 可覆盖 cfg.stride
        fmin: 可覆盖 cfg.fmin
        fmax: 可覆盖 cfg.fmax

    返回：
        spec: [H, W, 2] 的 float32 Tensor
              H: 频率 bins（截取 fmin~fmax 后）
              W: 时间帧数
              2: 通道数
        freqs_sel: [H] 的频率轴（Hz）
    """
    # ---------- 动态参数覆盖默认配置 ----------
    window_size = window_size or cfg.window_size
    stride = stride or cfg.stride
    fmin = fmin if fmin is not None else cfg.fmin
    fmax = fmax if fmax is not None else cfg.fmax

    # ---------- 确保采样率为 target_sr ----------
    waveform, sample_rate = ensure_sample_rate(
        waveform, sample_rate, cfg.target_sr
    )

    # ---------- 统一成 [C, T] ----------
    rank = tf.rank(waveform)
    waveform = tf.cond(
        tf.equal(rank, 1),
        lambda: tf.expand_dims(waveform, 0),
        lambda: waveform,
    )
    # 验证 rank 在 1 或 2
    tf.debugging.assert_greater_equal(rank, 1, message="waveform must be rank 1 or 2")
    tf.debugging.assert_less_equal(rank, 2, message="waveform must be rank 1 or 2")
    # waveform: [C, T]
    num_channels = tf.shape(waveform)[0]

    # ---------- STFT: [C, T] -> [C, W, F] ----------
    stfts = tf.signal.stft(
        signals=waveform,
        frame_length=window_size,
        frame_step=stride,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )  # [C, W, F]

    # ---------- 计算频率轴 & 截取频段 ----------
    # STFT 输出频率 bins 数量 = frame_length // 2 + 1
    n_fft = window_size
    num_freq_bins = n_fft // 2 + 1
    sr_f = tf.cast(sample_rate, tf.float32)

    freqs = tf.linspace(
        0.0,
        sr_f / 2.0,
        num_freq_bins,
    )  # [F]

    freq_mask = tf.logical_and(freqs >= fmin, freqs <= fmax)
    freqs_sel = tf.boolean_mask(freqs, freq_mask)  # [H]

    # stfts: [C, W, F] -> [C, F, W] -> 频率截取 -> [C, H, W]
    stfts = tf.transpose(stfts, perm=[0, 2, 1])          # [C, F, W]
    stfts = tf.boolean_mask(stfts, freq_mask, axis=1)    # [C, H, W]

    # ---------- 取幅度谱 ----------
    mags = tf.abs(stfts)  # [C, H, W]

    # ---------- 通道处理：单声道复制，>2 通道取前 2 ----------
    def expand_to_two_channels(m: tf.Tensor) -> tf.Tensor:
        ch0 = m[0:1, :, :]                   # [1, H, W]
        chs = tf.concat([ch0, ch0], axis=0)  # [2, H, W]
        return chs

    def take_first_two(m: tf.Tensor) -> tf.Tensor:
        return m[:2, :, :]                   # [2, H, W]

    mags_2 = tf.cond(
        tf.equal(num_channels, 1),
        lambda: expand_to_two_channels(mags),
        lambda: take_first_two(mags),
    )  # [2, H, W]

    # ---------- 最终规格： [H, W, 2] ----------
    spec = tf.transpose(mags_2, perm=[1, 2, 0])  # [H, W, 2]

    return tf.cast(spec, tf.float32), freqs_sel


# ====================== 高层封装：预处理模块 ======================

class AudioPreprocessor(tf.Module):
    """
    高层预处理模块：
        - 统一采样率到 cfg.target_sr
        - STFT + 频段截取
        - 单声道复制到 2 通道
        - 输出 [H, W, 2] spectrogram

    后续你可以在这里继续加：
        - log-mel 变换
        - 归一化
        - 固定 H, W 的裁剪 / 填充
        - patch 编码接口等
    """

    def __init__(self, cfg: AudioSTFTConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.cfg = cfg

    @tf.function
    def __call__(
        self,
        waveform: tf.Tensor,
        sample_rate: tf.Tensor,
        *,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> tf.Tensor:
        """
        调用方式：
            spec = preproc(waveform, sample_rate)
        或带参数：
            spec = preproc(waveform, sample_rate,
                           window_size=4096,
                           stride=1024,
                           fmin=50.0,
                           fmax=18000.0)

        参数：
            waveform: [T] 或 [C, T] 的 float32 Tensor
            sample_rate: 标量 int32/int64 Tensor（原始采样率）
            window_size / stride / fmin / fmax: 可选，覆盖 cfg 中的默认值

        返回：
            spec: [H, W, 2] 的 float32 Tensor
        """
        spec, _ = stft_spectrogram(
            waveform=waveform,
            sample_rate=sample_rate,
            cfg=self.cfg,
            window_size=window_size,
            stride=stride,
            fmin=fmin,
            fmax=fmax,
        )
        return spec


# ====================== 简单使用示例（可选） ======================

if __name__ == "__main__":
    # 仅示例：用 librosa 读一个音频，丢给 AudioPreprocessor
    import numpy as np
    import librosa

    # 替换成你自己的音频路径
    path = "example.m4a"

    # 读成 mono waveform + 原始采样率
    y, sr = librosa.load(path, sr=None, mono=True)  # y: np.ndarray [T]

    waveform_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    sample_rate_tf = tf.constant(sr, dtype=tf.int32)

    cfg = AudioSTFTConfig(
        target_sr=48000,
        window_size=2048,
        stride=512,
        fmin=30.0,
        fmax=16000.0,
    )

    preproc = AudioPreprocessor(cfg)

    # 可以用默认参数
    spec = preproc(waveform_tf, sample_rate_tf)
    print("Default spec shape:", spec.shape)

    # 也可以动态覆盖参数
    spec2 = preproc(
        waveform_tf,
        sample_rate_tf,
        window_size=4096,
        stride=1024,
        fmin=50.0,
        fmax=18000.0,
    )
    print("Overridden spec shape:", spec2.shape)
