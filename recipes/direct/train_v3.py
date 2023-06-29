import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd

class SLUWithTranDec(sb.Brain):
    """
    modules: Transformer, seq_lin, ctc_lin, env_corrupt
    """
    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        torch.Tensor or Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # NOTE: WAV AUG HERE!!!!!!!!!!!!!!!!!
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            # Default: add noise
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)
                # tokens_bos_lens = torch.cat([tokens_bos_lens, tokens_bos_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # ASR encoder forward pass
        with torch.no_grad():
            ASR_encoder_out = self.hparams.asr_model.encode_batch(
                wavs.detach(), wav_lens
            )

        enc_out, pred = self.modules.Transformer(
            ASR_encoder_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )
        e_in = self.hparams.output_emb(tokens_bos)



if __name__ == "__main__":
    model = SLUWithTranDec()