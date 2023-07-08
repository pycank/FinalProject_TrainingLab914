#!/usr/bin/env/python3
import sys
import io
import os
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import sentencepiece as sp


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        print(f"Config: {hparams}")

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # sb.create_experiment_directory(
    #     experiment_directory=hparams["output_folder"],
    #     hyperparams_to_save=hparams_file,
    #     overrides=overrides,
    # )

    # prepare data
    df = pd.read_csv(hparams['csv_path'])
    lst = df[hparams['col_name']].to_list()

    # Train tokenizer
    print("Start training...")
    model = io.BytesIO()
    sp.SentencePieceTrainer.train(
        user_defined_symbols=hparams['user_defined_symbols'],
        sentence_iterator=iter(lst),
        model_type=hparams['type'],
        model_writer=model,
        vocab_size=hparams['vocab_size'],
        bos_id=hparams['bos_index'],
        eos_id=hparams['eos_index'],
    )

    flag = ''
    if hparams['bos_index'] != -1:
        flag = flag + 'b'
    if hparams['eos_index'] != -1:
        flag = flag + 'e'
    if len(flag):
        flag = flag + 'id_'
    model_file_path = os.path.join(
        hparams['tokenizer_folder'],
        f"tokenizer_{hparams['vocab_size']}{hparams['type']}_{flag}{hparams['version']}.ckpt"
    )

    print()
    with open(model_file_path, 'wb') as f:
        f.write(model.getvalue())
        print(f"Saved: {model_file_path}")

    print()
    # Directly load the model from serialized model.
    spm = sp.SentencePieceProcessor(model_proto=model.getvalue())
    if hparams['print_vocab']:
        print("Vocab:")
        print([spm.id_to_piece(id) for id in range(spm.get_piece_size())])

    print()
    if hparams['try_encode_text']:
        print(f"Try encode: '{hparams['try_encode_text']}'")
        encoded = spm.encode(hparams['try_encode_text'])
        print(encoded)
        print([spm.id_to_piece(id) for id in encoded])