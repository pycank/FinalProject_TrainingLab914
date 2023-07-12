import inspect
from collections import OrderedDict

import torch
from speechbrain.nnet.containers import LengthsCapableSequential, Sequential
from speechbrain.nnet.linear import Linear
from speechbrain.utils.callchains import lengths_arg_exists


def _get_by_key(key, _dict: dict):
    if key in _dict.keys():
        return _dict[key]
    else:
        return None


class InjectUtils:
    def __init__(self):
        self.inject_before_dict = {}
        self.inject_after_dict = {}

    def inject(self, key, instance, pos: str = "before"):
        assert instance is not None
        assert pos == "before" or pos == "after"
        assert key

        match pos:
            case "before":
                if key not in self.inject_before_dict.keys():
                    self.inject_before_dict[key] = instance
                else:
                    raise Exception(f"Name {key} in list!")
            case "after":
                if key not in self.inject_after_dict.keys():
                    self.inject_after_dict[key] = instance
                else:
                    raise Exception(f"Name {key} in list!")

    def get_before(self, key):
        return _get_by_key(key, self.inject_before_dict)

    def get_after(self, key):
        return _get_by_key(key, self.inject_after_dict)


class InjectableLengthsCapableSequential(LengthsCapableSequential):
    """Sequential model that can take ``lengths`` in the forward method.

    This is useful for Sequential models that include RNNs where it is
    important to avoid padding, or for some feature normalization layers.

    Unfortunately, this module is not jit-able because the compiler doesn't
    know ahead of time if the length will be passed, and some layers don't
    accept the length parameter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inj_utils = InjectUtils()

    def inject(self, inject_layer, before_layer=None, after_layer=None):
        assert (before_layer is None) != (after_layer is None)

        if before_layer is not None:
            self.inj_utils.inject(before_layer, inject_layer, pos="before")
        if after_layer is not None:
            self.inj_utils.inject(after_layer, inject_layer, pos="after")
        return

    def append(self, *args, **kwargs):
        """Add a layer to the list of layers, inferring shape if necessary.
        """
        # Add lengths arg inference here.
        super().append(*args, **kwargs)
        latest_forward_method = list(self.values())[-1].forward
        self.takes_lengths.append(lengths_arg_exists(latest_forward_method))

    def forward(self, x, lengths=None):
        """Applies layers in sequence, passing only the first element of tuples.

        In addition, forward the ``lengths`` argument to all layers that accept
        a ``lengths`` argument in their ``forward()`` method (e.g. RNNs).

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.
        lengths : torch.Tensor
            The relative lengths of each signal in the tensor.
        """
        keys = self.keys()
        for layer, give_lengths in zip(self.values(), self.takes_lengths):
            key = iter(keys)
            print(key)
            before = self.inj_utils.get_before(key)
            after = self.inj_utils.get_after(key)
            if before:
                x = before(x)
            if give_lengths:
                x = layer(x, lengths=lengths)
            else:
                x = layer(x)
            if after:
                x = after(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

if __name__ == "__main__":
    ilcs = InjectableLengthsCapableSequential(
        a=Linear(5, input_size=10)
    )
    ilcs.inject(lambda x: print("X"), before_layer="a")
    print(ilcs.forward(torch.rand([1, 10])))
