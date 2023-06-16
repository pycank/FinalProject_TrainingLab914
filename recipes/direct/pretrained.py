import torch
import speechbrain as sb

show_results_every = 100  # plots results every N iterations
run_opts = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

class PipelineSLUTask(sb.pretrained.interfaces.Pretrained):
    HPARAMS_NEEDED = [
        "slu_enc",
        "output_emb",
        "dec",
        "seq_lin",
        "env_corrupt",
        "tokenizer",
    ]
    MODULES_NEEDED = [
        "slu_enc",
        "output_emb",
        "dec",
        "seq_lin",
        "env_corrupt",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def encode_file(self, path):

        tokens_bos = torch.tensor([[0]]).to(self.device)
        # tokens = torch.tensor([], dtype=torch.int64).to(self.device)

        waveform = self.load_audio(path)
        wavs = waveform.unsqueeze(0)
        wav_lens = torch.tensor([1.0])
        # Fake a batch:
        # batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        with torch.no_grad():
            rel_lens = rel_length.to(self.device)
            # ASR encoder forward pass
            ASR_encoder_out = self.hparams.asr_model.encode_batch(
                wavs.detach(), wav_lens
            )

            # SLU forward pass
            encoder_out = self.hparams.slu_enc(ASR_encoder_out)
            e_in = self.hparams.output_emb(tokens_bos)
            # print(e_in.shape)
            # print(encoder_out.shape)
            # print(wav_lens.shape)
            h, _ = self.hparams.dec(e_in, encoder_out, wav_lens)

            # Output layer for seq2seq log-probabilities
            logits = self.hparams.seq_lin(h)
            p_seq = self.hparams.log_softmax(logits)

            # Compute outputs
            # if (
            #     stage == sb.Stage.TRAIN
            #     and self.batch_count % show_results_every != 0
            # ):
            #     return p_seq, wav_lens
            # else:
            p_tokens, scores = self.hparams.beam_searcher(encoder_out, wav_lens)
            return p_seq, wav_lens, p_tokens

        # return ASR_encoder_out

    def decode(self, p_seq, wav_lens, predicted_tokens):
        # tokens_eos = torch.tensor([[0]]).to(self.device)
        # tokens_eos_lens = torch.tensor([0]).to(self.device)

        # Decode token terms to words
        predicted_semantics = [
            self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in predicted_tokens
        ]
        return predicted_semantics
