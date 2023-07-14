"""
Recipe for "direct" (speech -> semantics) SLU with ASR-based transfer learning.

Customize from train.py
- Using transformer instead of GNU+Attention
"""

import sys
import torch
import pandas as pd
import speechbrain as sb
import jsonlines
import ast
from dataio_prepare import dataio_prepare
from prepare_SLURP import prepare_SLURP_2
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


class SLUDirectTrans(sb.Brain):
    def __init__(self, *args, **kwargs):
        super(SLUDirectTrans, self).__init__(*args, **kwargs)
        # assert [
        #             'Transformer',
        #             'asr_model',
        #             'seq_lin',
        #             'log_softmax',
        #             'valid_search',
        #             'test_search',
        #             'seq_cost',
        #             'tokenizer',
        #             'output_folder',
        #             'cer_computer',
        #             'error_rate_computer',
        #             'epoch_counter',
        #             'train_logger',
        #             'wer_file',
        #         ] in self.hparams.keys()

    def compute_forward(self, batch, stage):
        """
        Forward computations from the waveform batches to the output probabilities.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            # Default: add noise
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)
                tokens_bos_lens = torch.cat([tokens_bos_lens, tokens_bos_lens])
                tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
                tokens_eos_lens = torch.cat([tokens_eos_lens, tokens_eos_lens])
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # ASR encoder forward pass
        with torch.no_grad():
            ASR_enc_out = self.hparams.asr_model.encode_batch(
                wavs.detach(), wav_lens
            )

        # SLU forward pass
        tfm_enc_out, tfm_dec_out = self.hparams.Transformer(
            ASR_enc_out, tokens_bos
        )

        # Output layer for seq2seq log-probabilities
        logits = self.hparams.seq_lin(tfm_dec_out)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        searcher = self.hparams.test_search if stage == sb.Stage.TEST else self.hparams.valid_search
        if (
                stage == sb.Stage.TRAIN
                and self.batch_count % show_results_every != 0
        ):
            return p_seq, wav_lens
        else:
            # tfm_enc_out or ASR_enc_out
            p_tokens, scores = searcher(tfm_enc_out, wav_lens)
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
        tokens, tokens_lens = batch.tokens

        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )

        # (No ctc loss)
        loss = loss_seq

        if (stage != sb.Stage.TRAIN) or (
                self.batch_count % show_results_every == 0
        ):
            # Decode token terms to words
            predicted_semantics = [
                self.hparams.tokenizer.decode_ids(utt_seq).split(" ")
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
                        # self.hparams["output_folder"] + "/predictions.jsonl", mode="a"
                        self.hparams.output_folder + "/predictions.jsonl", mode="a"
                ) as writer:
                    for i in range(len(predicted_semantics)):
                        try:
                            _dict = ast.literal_eval(
                                " ".join(predicted_semantics[i]).replace(
                                    "|", ","
                                )
                            )
                            try:
                                is_valid_semantic = isinstance(_dict, dict) \
                                    and ["scenario", "action", "entities"] not in list(_dict.keys()) \
                                    and all([['type', 'filler'] == list(arr.keys()) for arr in _dict["entities"]])
                            except Exception as e:
                                print(e)
                                is_valid_semantic = False
                            if not is_valid_semantic:
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
                        try:
                            _dict["file"] = self.id_to_file[ids[i]]
                            writer.write(_dict)
                        except Exception:
                            print(f"Not found: {ids[i]} in {self.id_to_file}")

        return loss

    def log_outputs(self, predicted_semantics, target_semantics):
        """
        TODO: log these to a file instead of stdout
        """
        for i in range(len(target_semantics)):
            try:
                print("")
                print(" ".join(predicted_semantics[i]).replace("|", ","))
                print(" ".join(target_semantics[i]).replace("|", ","))
                print("")
                # print one
                break
            except Exception:
                print(f"Error: SLU.log_outputs: predicted_semantics_len={len(predicted_semantics)}, id={i}, value={predicted_semantics[i]}")

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
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
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
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


def print_slu_details(slu: SLUDirectTrans):
    def show_module_list_parameters(module_list):
        n_params = 0
        for attr_name in module_list:
            module = module_list[attr_name]
            print(f"> {attr_name}: {module._get_name()}")
            for name, param in module.named_parameters():
                print(f"\t{name.ljust(40)}: {param.shape}")
                n_params = n_params + torch.numel(param)
        return n_params

    print("ASR Encoder: ")
    n_asr_enc_params = show_module_list_parameters(slu.hparams.asr_model.mods)
    print("==========")
    print("Trainable:")
    n_trainable_params = show_module_list_parameters(slu.hparams.modules)
    print("==========")
    print(f"Pretrained ASR Encoder parameters: {n_asr_enc_params}")
    print(f"Trainable parameters: {n_trainable_params}")


if __name__ == "__main__":
    show_results_every = 100  # plots results every N iterations

    # load hparams result
    # hparams_file = f"/home/kryo/Desktop/FinalProject_TrainingLab914/recipes/direct/hparams/train_v1f.yaml"
    # overrides = {
    #     "working_dir": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/working",
    #     "data_folder": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/slurp/audio",
    #     "asr_model": {
    #         "run_opts": {
    #             "device": "cuda" if torch.cuda.is_available() else "cpu"
    #         }
    #     },
    #     "skip_prep": True,
    # }
    # run_opts = {
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

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
            "sampling": hparams["sampling"],
        },
    )

    # # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Load prev checkpoint if possible
    hparams["checkpointer"].recover_if_possible()

    # # Brain class initialization
    slu_brain = SLUDirectTrans(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    print_slu_details(slu_brain)

    # # adding objects to trainer:
    slu_brain.tokenizer = tokenizer

    # Training
    slu_brain.fit(
        slu_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Test
    print("TESTING...")
    print("Creating id_to_file mapping...")
    id_to_file = {}
    df = pd.read_csv(hparams["csv_test"])
    for i in range(len(df)):
        id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

    slu_brain.id_to_file = id_to_file

    slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
    slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
