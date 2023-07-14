import torch
import speechbrain as sb
from speechbrain.pretrained.interfaces import Pretrained

show_results_every = 100  # plots results every N iterations
run_opts = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# class PipelineSLUTask(sb.pretrained.interfaces.Pretrained):
#     HPARAMS_NEEDED = [
#         "slu_enc",
#         "output_emb",
#         "dec",
#         "seq_lin",
#         "env_corrupt",
#         "tokenizer",
#     ]
#     MODULES_NEEDED = [
#         "slu_enc",
#         "output_emb",
#         "dec",
#         "seq_lin",
#         "env_corrupt",
#     ]
#
#     def __init__.py(self, *args, **kwargs):
#         super().__init__.py(*args, **kwargs)
#         pass
#
#     def encode_file(self, path):
#
#         tokens_bos = torch.tensor([[0]]).to(self.device)
#         # tokens = torch.tensor([], dtype=torch.int64).to(self.device)
#
#         waveform = self.load_audio(path)
#         wavs = waveform.unsqueeze(0)
#         wav_lens = torch.tensor([1.0])
#         # Fake a batch:
#         # batch = waveform.unsqueeze(0)
#         rel_length = torch.tensor([1.0])
#         with torch.no_grad():
#             rel_lens = rel_length.to(self.device)
#             # ASR encoder forward pass
#             ASR_encoder_out = self.hparams.asr_model.encode_batch(
#                 wavs.detach(), wav_lens
#             )
#
#             # SLU forward pass
#             encoder_out = self.hparams.slu_enc(ASR_encoder_out)
#             e_in = self.hparams.output_emb(tokens_bos)
#             # print(e_in.shape)
#             # print(encoder_out.shape)
#             # print(wav_lens.shape)
#             h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)
#
#             # Output layer for seq2seq log-probabilities
#             logits = self.hparams.seq_lin(h)
#             p_seq = self.hparams.log_softmax(logits)
#
#             # Compute outputs
#             # if (
#             #     stage == sb.Stage.TRAIN
#             #     and self.batch_count % show_results_every != 0
#             # ):
#             #     return p_seq, wav_lens
#             # else:
#             p_tokens, scores = self.hparams.beam_searcher(encoder_out, wav_lens)
#             return p_seq, wav_lens, p_tokens
#
#         # return ASR_encoder_out
#
#     def decode(self, p_seq, wav_lens, predicted_tokens):
#         # tokens_eos = torch.tensor([[0]]).to(self.device)
#         # tokens_eos_lens = torch.tensor([0]).to(self.device)
#
#         # Decode token terms to words
#         predicted_semantics = [
#             self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
#             for utt_seq in predicted_tokens
#         ]
#         return predicted_semantics


class EncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contain the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer

    def transcribe_file(self, path, **kwargs):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)
