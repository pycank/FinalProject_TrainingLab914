import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast

from recipes.direct.dataio_prepare import dataio_prepare
from recipes.direct.prepare_SLURP import prepare_SLURP_2


def _decode_ids(tokenizer, utt_seq):
    try:
        semantic = tokenizer.decode_ids(utt_seq).split(" ")
        return semantic
    except Exception:
        print(f"ERROR: unable to decode utt_seq={utt_seq}")

class SLUWithTranDec(sb.Brain):
    """
    SLU wav2vec with transformer
    Todo: impl transformer

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
        # tokens_bos, tokens_bos_lens = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # NOTE: WAV AUG HERE!!!!!!!!!!!!!!!!!
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            # Default: add noise
            # if hasattr(self.hparams, "env_corrupt"):
            #     wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
            #     wavs = torch.cat([wavs, wavs_noise], dim=0)
            #     wav_lens = torch.cat([wav_lens, wav_lens])
            #     tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)
            #     tokens_bos_lens = torch.cat([tokens_bos_lens, tokens_bos_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

            #  encoder forward pass
            # wav: (batch_size, len)
            # wav_lens: (batch_size)
            wav2vec2_out = self.modules.wav2vec2(wavs, wav_lens)
            # wav2vec2_out: (batch_size, .., 768)

            # print(f"wav2vec2_out: {wav2vec2_out.shape}")
            # print(f"tokens_bos: {tokens_bos.shape}")
            # print(f"wav_lens: {wav_lens.shape}")
            # SLU forward pass
            # enc_out == size wav2vec2_out
            # pred: (batch_size, .., 58)
            # tokens_bos: (batch_size, ..padded)
            # wav_lens: [0-1., ....] (batch_size)
            enc_out, pred = self.hparams.Transformer(
                wav2vec2_out, tokens_eos, wav_lens, pad_idx=0 # self.hparams.pad_index
            )

            # output layer for seq2seq log-probabilities
            pred = self.hparams.seq_lin(pred)
            p_seq = self.hparams.log_softmax(pred)

            # Compute outputs
            if (
                    stage == sb.Stage.TRAIN
                    and self.batch_count % show_results_every != 0
            ):
                return p_seq, wav_lens
            else:
                p_tokens, scores = self.hparams.beam_searcher(
                    wav2vec2_out, wav_lens
                )
                return p_seq, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""
        if (
            stage == sb.Stage.TRAIN
            and self.batch_count % show_results_every != 0
        ):
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        a = p_seq.shape
        b = tokens_eos.shape
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
            self.batch_count % show_results_every == 0
        ):
            predicted_semantics = []
            # Decode token terms to words
            predicted_semantics = [
                _decode_ids(self.hparams.tokenizer, utt_seq)
                # tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]

            target_semantics = [wrd.split(" ") for wrd in batch.semantics]

            self.log_outputs(predicted_semantics, target_semantics)

            if stage != sb.Stage.TRAIN:
                self.wer_metric.append(
                    ids, predicted_semantics, target_semantics
                )
                self.cer_metric.append(
                    ids, predicted_semantics, target_semantics
                )

            if stage == sb.Stage.TEST:
                # write to "predictions.jsonl"
                with jsonlines.open(
                    self.hparams["output_folder"] + "/predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            _dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                            if not isinstance(_dict, dict):
                                _dict = {
                                    "scenario": "none",
                                    "action": "none",
                                    "entities": [],
                                }
                        except SyntaxError:  # need this if the output is not a valid dictionary
                            _dict = {
                                "scenario": "none",
                                "action": "none",
                                "entities": [],
                            }
                        _dict["file"] = self.id_to_file[ids[i]]
                        writer.write(_dict)

        return loss


    def log_outputs(self, predicted_semantics, target_semantics):
        """ TODO: log these to a file instead of stdout """
        for i in range(len(target_semantics)):
            try:
                print(" ".join(predicted_semantics[i]).replace("|", ","))
                print(" ".join(target_semantics[i]).replace("|", ","))
                print("")
            except Exception:
                print(f"Error: SLU.log_outputs: predicted_semantics_len={len(predicted_semantics)}, id={i}, value={predicted_semantics[i]}")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()
        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()
        self.batch_count += 1
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_count = 0

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "wave2vec_lr": old_lr_wav2vec2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec2_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.wav2vec2_optimizer.zero_grad(set_to_none)
        self.optimizer.zero_grad(set_to_none)

if __name__ == "__main__":

    show_results_every = 1  # plots results every N iterations
    hparams_file = f"/home/kryo/Desktop/FinalProject_TrainingLab914/recipes/direct/hparams/train_v5.yaml"
    overrides = {
        "working_dir": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/working",
        "data_folder": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/slurp/audio",
        "skip_prep": True,
    }
    run_opts = {
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Load hyperparameters file with command-line overrides
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    # sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_SLURP_2,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "train_splits": hparams["train_splits"],
            "slu_type": "direct",
            "skip_prep": hparams["skip_prep"],
            "sampling": hparams["sampling"] if "sampling" in hparams.keys() else None,
        },
    )

    # # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Load prev checkpoint if possible
    hparams["checkpointer"].recover_if_possible()

    model = SLUWithTranDec(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # print_slu_details(model)

    # adding objects to trainer:
    model.tokenizer = tokenizer

    with torch.autograd.detect_anomaly():
        model.fit(
            model.hparams.epoch_counter,
            train_set,
            valid_set,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # # Test
    print("Creating id_to_file mapping...")
    id_to_file = {}
    import pandas as pd
    df = pd.read_csv(hparams["csv_test"])
    for i in range(len(df)):
        id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

    model.id_to_file = id_to_file

    model.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    model.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
