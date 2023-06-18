import torch
import sys

from dataio_prepare import dataio_prepare
from prepare_SLURP import prepare_SLURP_2
from pipeline_SLU import SLU

import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import pandas as pd
import speechbrain as sb

show_results_every = 100  # plots results every N iterations
# hparams_file = f"./results/better_tokenizer/1986/hyperparams.yaml"
# overrides = {}
# run_opts = {
#     "device": "cuda" if torch.cuda.is_available() else "cpu"
# }

# Load hyperparameters file with command-line overrides
hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

# multi-gpu (ddp) save data preparation
run_on_main(
    prepare_SLURP_2,
    kwargs={
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["output_folder"],
        "train_splits": hparams["train_splits"],
        "slu_type": "direct",
        "skip_prep": hparams["skip_prep"],
    },
)

# We download and pretrain the tokenizer
run_on_main(hparams["pretrainer"].collect_files)
hparams["pretrainer"].load_collected(device=run_opts["device"])

# Load prev checkpoint if possible
hparams["checkpointer"].recover_if_possible()


# If --distributed_launch then
# create ddp_group with the right communication protocol
sb.utils.distributed.ddp_init_group(run_opts)

sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
)


# # here we create the datasets objects as well as tokenization and encoding
(train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)


# # We download and pretrain the tokenizer
run_on_main(hparams["pretrainer"].collect_files)
hparams["pretrainer"].load_collected(device=run_opts["device"])

# Load prev checkpoint if possible
hparams["checkpointer"].recover_if_possible()

# # Brain class initialization
slu_brain = SLU(
    modules=hparams["modules"],
    opt_class=hparams["opt_class"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

# # adding objects to trainer:
slu_brain.tokenizer = tokenizer

# # Training
slu_brain.fit(
    slu_brain.hparams.epoch_counter,
    train_set,
    valid_set,
    train_loader_kwargs=hparams["dataloader_opts"],
    valid_loader_kwargs=hparams["dataloader_opts"],
)

# # Test
print("Creating id_to_file mapping...")
id_to_file = {}
df = pd.read_csv(hparams["csv_test"])
for i in range(len(df)):
    id_to_file[str(df.ID[i])] = df.wav[i].split("/")[-1]

slu_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test_real.txt"
slu_brain.evaluate(test_set, test_loader_kwargs=hparams["dataloader_opts"])
