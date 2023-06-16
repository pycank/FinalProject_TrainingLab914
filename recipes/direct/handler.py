from typing import Dict, List, Any
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from recipes.pretrained import PipelineSLUTask

class EndpointHandler():
    def __init__(self, path=""):
        hparams_file = f"{path}/better_tokenizer/1986/hyperparams.yaml"
        overrides = {}
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        run_opts = {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        # We download and pretrain the tokenizer
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

        self.pipeline = PipelineSLUTask(
            modules=hparams['modules'],
            hparams=hparams,
            run_opts=run_opts
        )

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
       data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
      Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        # pseudo
        # self.model(input)
        data = data.get("inputs", data)
        print(data)
        ps, wl, pt = self.pipeline.encode_file(data)
        print(ps)
        print(wl)
        print(pt)
        return self.pipeline.decode(ps, wl, pt)
