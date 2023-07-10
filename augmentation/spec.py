import torch
import random
import numpy as np
import torchaudio
from torchaudio import transforms


class SpectrogramToDB(object):
    """
    change the spectrogram from the power/amplitude scale to the decibel scale
    """

    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10. if stype == "power" else 20.
        self.amin = 1e-10
        self.ref_value = 1
        self.db_multiplier = np.log10(np.maximum(self.amin, self.ref_value))

    def __call__(self, spec):
        spec_db = self.multiplier * torch.log10(torch.clamp(spec, min=self.amin))
        spec_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            spec_db = torch.max(spec_db, spec_db.new_full((1,), spec_db.max() - self.top_db))
        return spec_db


def tfm_spectro(
        sig,
        sr=16000,
        to_db_scale=False,
        n_fft=1024,
        ws=None,
        hop=None,
        f_min=0.0,
        f_max=-80,
        pad=0,
        n_mels=128
):
    # reshape signal for torchaudio to generate the spectrogram
    print(sig.reshape(1, -1))
    mel = transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels, n_fft=n_fft,
                                    win_length=ws, hop_length=hop, f_min=f_min, f_max=f_max, pad=pad, ) \
        (sig.reshape(1, -1))
    if to_db_scale: mel = SpectrogramToDB(stype='magnitude', top_db=f_max)(mel)
    return mel


def _freq_mask(spec, F=25, num_masks=1, replace_with_zero=False):
    """
    """
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f: return cloned
        mask_end = random.randrange(f_zero, f_zero + f)
        if replace_with_zero:
            cloned[0][f_zero: mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def _time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        if t_zero == t_zero + t: return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if replace_with_zero:
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned


class SpecDomainAugmentation(torch.nn.Module):
    """
    Augmentation speech data in spectrogram domain
    """

    def __init__(
            self,
            freq_mask_f=25,
            freq_mask_num_masks=1,
            freq_mask_repl_with_zero=False,
            time_mask_f=25,
            time_mask_num_masks=1,
            time_mask_repl_with_zero=False,
    ):
        super().__init__()
        self.freq_mask_f = freq_mask_f
        self.freq_mask_num_masks = freq_mask_num_masks
        self.freq_mask_repl_with_zero = freq_mask_repl_with_zero
        self.time_mask_f = time_mask_f
        self.time_mask_num_masks = time_mask_num_masks
        self.time_mask_repl_with_zero = time_mask_repl_with_zero

    def forward(self, spec):
        spec = _freq_mask(spec, self.time_mask_f, self.freq_mask_num_masks, self.freq_mask_repl_with_zero)
        spec = _time_mask(spec, self.time_mask_f, self.time_mask_num_masks, self.time_mask_repl_with_zero)
        return spec


if __name__ == "__main__":
    audio, sample_rate = torchaudio.load("../datasets/slurp/audio/slurp_real/audio-1502200272.flac")
    spectro = tfm_spectro(audio, ws=512, hop=256, n_mels=128, to_db_scale=True, f_max=8000, f_min=-80)

    aug = SpecDomainAugmentation()
    print(aug.forward(spectro))
