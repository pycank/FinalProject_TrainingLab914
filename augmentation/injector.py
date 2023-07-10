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

# def lcs_injector(
#         lcs: LengthsCapableSequential,
#         layer,
#         *args,
#         layer_name=None,
#         pos: int = 0,
#         **kwargs
# ):
#     """
#     Custom hook to inject new layer in LengthsCapableSequential container
#     Note that this function only for appended layer which does not change output size. E.g.: augmentation
#     """
#     # Compute layer_name
#     if layer_name is None:
#         layer_name = str(len(lcs))
#     elif layer_name in lcs:
#         raise Exception(f"Layer {layer_name} is existed!")
#         # index = 0
#         # while f"{layer_name}_{index}_injected" in lcs:
#         #     index += 1
#         # layer_name = f"{layer_name}_{index}_injected"
#
#     # Finally, append the layer.
#     try:
#         lcs.add_module(layer_name, layer)
#     except TypeError:
#         raise ValueError(
#             "Must pass `input_shape` at initialization and use "
#             "modules that take `input_shape` to infer shape when "
#             "using `append()`."
#         )
#
#     latest_forward_method = list(lcs.values())[-1].forward
#     lcs.takes_lengths.append(lengths_arg_exists(latest_forward_method))
#
# if __name__ == "__main__":
#     lcs = LengthsCapableSequential(
#         b=Linear(10, input_size=10),
#         a=Linear(5, input_size=10)
#     )
#     new_layer = Linear(10, input_size=3)
#     # lcs.add_module()
#     # print(lcs.insert(2, Linear(10, input_size=10)))
#     # print(lcs(torch.Tensor(10)))
#     # Get the current modules
#     modules = lcs._modules
#     print(lcs._modules)
#
#     modules_list = list(modules.items())
#     modules_name = list(modules.keys())
#
#     # Create a new OrderedDict from the modified list
#     id = 1
#     # modules_ordered = OrderedDict()
#     i = 0
#     for module, name in zip(modules_list, modules_name):
#         if i == id:
#             modules.update({f"aaaa": new_layer})
#         i = i+1
#         _, m = module
#         modules.update({f"{name}": m})
#
#     # lcs._modules = modules_ordered
#     print(lcs._modules)
#     print(lcs(torch.Tensor(10)))

