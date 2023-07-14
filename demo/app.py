from types import SimpleNamespace
import matplotlib.pyplot as plt
import streamlit as st
import librosa
import torchaudio
import pandas as pd
import numpy as np
import os
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import Pretrained, EndToEndSLU
from speechbrain.utils.distributed import run_on_main
import torch
import speechbrain as sb
import jsonlines
import ast
import io
import json


show_results_every = 100


class SLUTask(Pretrained):
    """
    Customize EndToEndSLU pretrained implementation
    """
    HPARAMS_NEEDED = ["tokenizer", "asr_model"]
    MODULES_NEEDED = [
        "slu_enc",
        "beam_searcher",
    ]

    def __init__(
            self,
            modules, hparams, *args,
            MODULES_NEEDED_MAPPER=None, **kwargs
    ):
        if MODULES_NEEDED_MAPPER:
            assert set(MODULES_NEEDED_MAPPER.keys()).issubset(set(self.MODULES_NEEDED))
            for key in MODULES_NEEDED_MAPPER.keys():
                value = MODULES_NEEDED_MAPPER[key]
                if value:
                    hparams[key] = hparams[value]
                else:
                    hparams[key] = None
        super().__init__(modules, hparams, *args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.asr_model = self.hparams.asr_model
        self.slu_enc = self.hparams.slu_enc
        self.beam_searcher = self.hparams.beam_searcher

    def decode_file(self, path):
        """Maps the given audio file to a string representing the
        semantic dictionary for the utterance.

        Arguments
        ---------
        path : str
            Path to audio file to decode.

        Returns
        -------
        str
            The predicted semantics.
        """
        waveform = self.load_audio(path)
        waveform = waveform.to(self.device)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.decode_batch(batch, rel_length)
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

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
        ASR_encoder_out = self.asr_model.encode_batch(wavs.detach(), wav_lens)
        if self.slu_enc:
            # encoder_out = self.mods.slu_enc(ASR_encoder_out)
            encoder_out = self.slu_enc(ASR_encoder_out)
            return encoder_out
        return ASR_encoder_out

    def decode_batch(self, wavs, wav_lens):
        """Maps the input audio to its semantics

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
            Each waveform in the batch decoded.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores = self.beam_searcher(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full decoding - note: no gradients through decoding"""
        return self.decode_batch(wavs, wav_lens)

run_opts = {
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_model(model_cfg):
    config_path = model_cfg['config_path']
    config_overrides = {
        "asr_model": {
            "run_opts": {
                "device": run_opts['device']
            }
        },
        "working_dir": "../datasets",
        "data_folder": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/slurp/audio",
        "output_folder": "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/working/out",
        "skip_prep": True,
        **model_cfg['config_overrides'],
    }
    with open(config_path) as fin:
        hparams = load_hyperpyyaml(fin, config_overrides, overrides_must_match=False)

    # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Load prev checkpoint if possible
    loaded_ckpt = hparams["checkpointer"].recover_if_possible(device=run_opts["device"])

    if not loaded_ckpt:
        raise Exception(f"Load ckpt failed: {model_cfg}")

    # Pretrained class initialization
    slu_brain = SLUTask(
        hparams['modules'],
        hparams,
        MODULES_NEEDED_MAPPER=model_cfg['mapper']
    )

    return slu_brain

def plot_wave(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()

def format_semantic(semantic):
    semantic = str.replace(semantic, "|", ",")
    semantic = str.replace(semantic, "'", '"')
    try:
        semantic = json.loads(semantic)
        return json.dumps(semantic, indent=2)
    except Exception:
        return semantic


def upload_audio():
    audio_bytes = None
    uploaded_file = st.file_uploader("Choose a audio file")
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
    return audio_bytes


def pick_audio():
    audio_bytes = None

    file_path = st.selectbox(
        "Pick file from test set",
        options=df['wav']
    )
    if file_path:
        st.code(
            f"{format_semantic(df[df['wav'] == file_path]['semantics'].values[0])}",
            language="json"
        )
        st.text(f"Transcript: {df[df['wav'] == file_path]['transcript'].values[0]}")

    if file_path is not None:
        file_path = str.replace(file_path, "$data_root/", "../datasets/slurp/audio/")
        wav, sr = librosa.load(file_path, sr=16000)
        audio_file = open(file_path, 'rb')
        audio_bytes = audio_file.read()
    return audio_bytes


if __name__ == "__main__":
    TEST_CSV_PATH = "/home/kryo/Desktop/FinalProject_TrainingLab914/datasets/working/out/test-type=direct.csv"
    MODULE_LIST_HPARAMS = "../loader/hparams/model_list.yaml"
    VERSIONS = {
        "v1": {
            "config_path": "../recipes/direct/hparams/train.yaml",
            "config_overrides": {
                "save_folder": "../datasets/working/out/save/v1"
            },
            "mapper": {}
        },
        "v1t3": {
            "config_path": "../recipes/direct/hparams/train_v1t3.yaml",
            "config_overrides": {
                "save_folder": "../datasets/working/out/save/v1t3"
            },
            "mapper": {}
        },
        "v1f": {
            "config_path": "../recipes/direct/hparams/train_v1f.yaml",
            "config_overrides": {
                "save_folder": "../datasets/working/out/save/v1f"
            },
            "mapper": {
                "beam_searcher": "test_search",
                "slu_enc": None
            }
        }
    }

    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_file_dir, MODULE_LIST_HPARAMS)

    model_cfg_list = VERSIONS
    # cli_overrides = {}
    # with open(config_path) as fin:
    #     model_list_params = load_hyperpyyaml(fin, cli_overrides, overrides_must_match=False)
    #     model_cfg_list = model_list_params['models']

    st.write("""
    # SLU demo
    """)

    st.header("Test set")
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(TEST_CSV_PATH)
    df = st.session_state['df']

    st.write(df)

    # =========================================================
    st.divider()

    st.header("Inference")

    versions = st.multiselect(
        "Pick versions",
        options=model_cfg_list.keys()
    )
    if 'models' not in st.session_state:
        st.session_state.models = {}
    remove_models = set(st.session_state.models.keys()) - set(versions)
    for version in remove_models:
        del st.session_state.models[version]
    for version in versions:
        if version not in st.session_state.models.keys():
            model = load_model(model_cfg_list[version])
            st.session_state.models[version] = model
    models = st.session_state.models
    # print(models)

    inp_type = st.select_slider(
        "Input type:",
        options=[
            "audio from test set",
            "upload audio"
        ]
    )

    audio_bytes = upload_audio() if inp_type == "upload audio" else pick_audio()

    if audio_bytes:
        audio_input = io.BytesIO(audio_bytes)
        audio_input, _ = librosa.load(audio_input, sr=None)
        # print(audio_input)
        st.pyplot(
            plot_wave(
                np.array(audio_input).astype(float),
                sr=16000
            ),
        )
        st.audio(audio_bytes, format='audio/flac')

        if st.button("inference"):
            if not len(models):
                st.write("Select model version first!")
            else:
                for k in models:
                    st.write(k)
                    model = models[k]

                    wavs = torch.tensor(np.array([audio_input]))
                    wav_lens = torch.tensor(np.array([len(audio_input)]))

                    predicted_semantics = model.decode_batch(wavs, wav_lens)
                    st.code(format_semantic(' '.join(predicted_semantics[0])), language="json")
