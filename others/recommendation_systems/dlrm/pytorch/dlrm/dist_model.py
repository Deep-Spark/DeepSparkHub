"""Distributed version of DLRM model

In order to code the hybrid decomposition, the model code needs to be restructured. I don't know a clean enough
way to do it in the serialized model.py. So even a lot of codes will be duplicated between the 2 files, I still
believe it is easier and cleaner to just implement a distributed version from scratch instead of reuse the same
file.

The model is broken into 2 parts:
    - Bottom model: embeddings and bottom MLP
    - Top model: interaction and top MLP
The all to all communication will happen between bottom and top model

"""

import copy
import math

from absl import logging

import torch
from torch import nn

import dlrm.nn
from dlrm.nn.functional import dotBasedInteract
from dlrm.utils import distributed as dist

try:
    from apex import mlp
except ImportError:
    logging.warning("APEX MLP is not availaible!")
    _USE_APEX_MLP = False
else:
    _USE_APEX_MLP = True


def get_criteo_device_mapping(num_gpus=4):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. 26 embeddings will be distributed across among all the devices. 0, 9, 19, 20, 21
    are the large ones, 20GB each.

    Args:
        num_gpus (int): Default 4.

    Returns:
        device_mapping (dict):
    """
    device_mapping = {'bottom_mlp': 0}  # bottom_mlp must be on the first GPU for now.
    if num_gpus == 4:
        device_mapping.update({
            'embedding' : [
                [0, 1],
                [7, 8, 9, 10, 11, 12, 13, 20],
                [14, 15, 16, 17, 18, 19, 22, 23],
                [21, 24, 25, 2, 3, 4, 5, 6]],
            'vectors_per_gpu' : [3, 8, 8, 8]})
    elif num_gpus == 8:
        device_mapping.update({
            'embedding' : [
                [],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
                [20, 22, 23, 24],
                [21, 25]],
            'vectors_per_gpu' : [1, 4, 4, 4, 4, 4, 4, 2]})
    elif num_gpus == 16:
        device_mapping.update({
            'embedding' : [
                [],
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
                [10, 11],
                [12, 13],
                [14, 15],
                [16, 17],
                [18, 19],
                [20],
                [21, 22],
                [23],
                [24],
                [25]],
            'vectors_per_gpu' : [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1]})
    else:
        raise NotImplementedError

    return device_mapping


class DlrmBottom(nn.Module):
    """Bottom model of DLRM

    Embeddings and bottom MLP of DLRM. Only joint embedding is supported in this version.

    Args:
        num_numerical_features (int): Number of dense features fed into bottom MLP
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        bottom_mlp_sizes (list): A list of integer indicating bottom MLP layer sizes. Last bottom MLP layer
            must be embedding_dim. Default None, not create bottom embedding on current device.
        embedding_dim (int): Length of embedding vectors. Default 128
        hash_indices (bool): If True, hashed_index = index % categorical_feature_size. Default False
        device (torch.device): where to create the embedding. Default "cuda"
        use_embedding_ext (bool): If True, use embedding extension.
    """
    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes=None, embedding_dim=128,
                 hash_indices=False, device="cuda", use_embedding_ext=True):
        super(DlrmBottom, self).__init__()
        if bottom_mlp_sizes is not None and embedding_dim != bottom_mlp_sizes[-1]:
            raise TypeError("The last bottom MLP layer must have same size as embedding.")

        self._embedding_dim = embedding_dim
        self._hash_indices = hash_indices
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        # Create bottom MLP
        if bottom_mlp_sizes is not None:
            if _USE_APEX_MLP:
                self.bottom_mlp = mlp.MLP([num_numerical_features] + bottom_mlp_sizes).to(device)
            else:
                bottom_mlp_layers = []
                input_dims = num_numerical_features
                for output_dims in bottom_mlp_sizes:
                    bottom_mlp_layers.append(
                        nn.Linear(input_dims, output_dims))
                    bottom_mlp_layers.append(nn.ReLU(inplace=True))
                    input_dims = output_dims
                self.bottom_mlp = nn.Sequential(*bottom_mlp_layers).to(device)
                for module in self.bottom_mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(
                            module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                        nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))
        else:
            # An empty list with Module property makes other code eaiser. For example, can call parameters()
            # and return empty iterator intead of having a contidion to skip it.
            self.bottom_mlp = torch.nn.ModuleList()

        # Create joint embedding
        if categorical_feature_sizes:
            logging.warning("Combined all categorical features to single embedding table.")
            if not use_embedding_ext:
                self.joint_embedding = dlrm.nn.BuckleEmbedding(categorical_feature_sizes, embedding_dim, device)
                for cat, size in enumerate(categorical_feature_sizes):
                    module = self.joint_embedding
                    nn.init.uniform_(
                        module.embedding.weight.data[module.offsets[cat]:module.offsets[cat + 1]],
                        -math.sqrt(1. / size),
                        math.sqrt(1. / size))
            else:
                self.joint_embedding = dlrm.nn.JointSparseEmbedding(
                    categorical_feature_sizes, embedding_dim, device)
                for cat, size in enumerate(categorical_feature_sizes):
                    module = self.joint_embedding
                    nn.init.uniform_(
                        module.weights.data[module.offsets[cat]:module.offsets[cat + 1]],
                        -math.sqrt(1. / size),
                        math.sqrt(1. / size))
        else:
            self.joint_embedding = torch.nn.ModuleList()

    def forward(self, numerical_input, categorical_inputs):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]

        Returns:
            Tensor: Concatenated bottom mlp and embedding output in shape [batch, 1 + #embedding, embeddding_dim]
        """
        batch_size = numerical_input.size()[0]

        bottom_output = []

        # Reshape bottom mlp to concatenate with embeddings
        if self.bottom_mlp:
            # [b, 1, embeddding_dim]
            bottom_output.append(self.bottom_mlp(numerical_input).view(batch_size, 1, -1))
            # print(f"bottom_output:{bottom_output.size()}")

        # print(f"self._hash_indices:{self._hash_indices}")
        if self._hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[:, cat] %= size
                logging.log_first_n(
                    logging.WARNING, F"Hashed indices out of range.", 1)

        # NOTE: It doesn't transpose input
        if self.num_categorical_features > 0:
            bottom_output.append(self.joint_embedding(categorical_inputs).to(numerical_input.dtype))

        if len(bottom_output) == 1:
            cat_bottom_out = bottom_output[0]
        else:
            cat_bottom_out = torch.cat(bottom_output, dim=1)
        
        # [b, 1+num_categorical_features, embeddding_dim]
        return cat_bottom_out

    # pylint:disable=missing-docstring
    @property
    def num_categorical_features(self):
        return len(self._categorical_feature_sizes)

    def extra_repr(self):
        s = F"hash_indices={self._hash_indices}"
        return s
    # pylint:enable=missing-docstring


class DlrmTop(nn.Module):
    """Top model of DLRM

    Interaction and top MLP of DLRM.

    Args:
        top_mlp_sizes (list): A list of integers indicating top MLP layer sizes.
        num_interaction_inputs (int): Number of input vectors to interaction, equals to #embeddings + 1 (
            bottom mlp)
        embedding_dim (int): Length of embedding vectors. Default 128
        interaction_op (string): Type of interactions. Default "dot"
    """
    def __init__(self, top_mlp_sizes, num_interaction_inputs, embedding_dim=128, interaction_op="dot"):
        super(DlrmTop, self).__init__()
        self._interaction_op = interaction_op

        if interaction_op == "dot":
            num_interactions = (num_interaction_inputs * (num_interaction_inputs - 1)) // 2 + embedding_dim
        elif interaction_op == "cat":
            num_interactions = num_interaction_inputs * embedding_dim
        else:
            raise TypeError(F"Unknown interaction {interaction_op}.")

        # Create Top MLP
        top_mlp_layers = []
        input_dims = num_interactions + 1  # pad 1 to be multiple of 8
        if _USE_APEX_MLP:
            top_mlp_layers.append(mlp.MLP([input_dims] + top_mlp_sizes[:-1]))
            top_mlp_layers.append(nn.Linear(top_mlp_sizes[-2], top_mlp_sizes[-1]))
        else:
            for output_dims in top_mlp_sizes[:-1]:
                top_mlp_layers.append(nn.Linear(input_dims, output_dims))
                top_mlp_layers.append(nn.ReLU(inplace=True))
                input_dims = output_dims
            top_mlp_layers.append(nn.Linear(input_dims, top_mlp_sizes[-1]))

        self.top_mlp = nn.Sequential(*top_mlp_layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))

        # Set corresponding weight of padding to 0
        if not _USE_APEX_MLP:
            nn.init.zeros_(self.top_mlp[0].weight[:, -1].data)
        else:
            nn.init.zeros_(self.top_mlp[0].weights[0][:, -1].data)

    # pylint:disable=missing-docstring
    def extra_repr(self):
        s = F"interaction_op={self._interaction_op}"
        return s
    # pylint:enable=missing-docstring

    def forward(self, bottom_output):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]

            bottom_output:[b, 1+num_categorical_features, emb_dim]
        """
        # The first vector in bottom_output is from bottom mlp
        bottom_mlp_output = bottom_output.narrow(1, 0, 1).squeeze()
        if self._interaction_op == "dot": 
            interaction = torch.bmm(bottom_output, torch.transpose(bottom_output, 1, 2))
            tril_indices_row, tril_indices_col = torch.tril_indices(
                interaction.shape[1], interaction.shape[2], offset=-1)
            interaction_flat = interaction[:, tril_indices_row, tril_indices_col]

            # concatenate dense features and interactions
            zero_padding = torch.zeros(
                bottom_output.shape[0], 1, dtype=bottom_output.dtype, device=bottom_output.device)

            #[b, (1+num_categorical_features)*num_categorical_features/2+emb_dim]
            interaction_output = torch.cat((bottom_mlp_output, interaction_flat, zero_padding), dim=1)

        elif self._interaction_op == "cat":
            interaction_output = bottom_output
        else:
            raise NotImplementedError

        top_mlp_output = self.top_mlp(interaction_output)

        # [b]
        return top_mlp_output


class BottomToTop(torch.autograd.Function):
    """Switch from model parallel to data parallel

    Wrap the communication of doing from bottom model in model parallel fashion to top model in data parallel

    TODO (haow): Current implementation assumes all the gpu gets same number of vectors from bottom model. May need
        to change it to a more generalized solution.
    """

    @staticmethod
    def forward(ctx, local_bottom_outputs, batch_size_per_gpu, vector_dim, vectors_per_gpu):
        """
        Args:
            ctx : Pytorch convention
            local_bottom_outputs (Tensor): Concatenated output of bottom model
            batch_size_per_gpu (int):
            vector_dim (int):
            vectors_per_gpu (int): Note, bottom MLP is considered as 1 vector

        Returns:
            slice_embedding_outputs (Tensor): Patial output from bottom model to feed into data parallel top model
        """
        ctx.world_size = torch.distributed.get_world_size()
        ctx.batch_size_per_gpu = batch_size_per_gpu
        ctx.vector_dim = vector_dim
        ctx.vectors_per_gpu = vectors_per_gpu

        # # Buffer shouldn't need to be zero out. If not zero out buffer affecting accuracy, there must be a bug.
        # bottom_output_buffer = [torch.empty(
        #     batch_size_per_gpu, n * vector_dim,
        #     device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype) for n in vectors_per_gpu]

        # torch.distributed.all_to_all(bottom_output_buffer, list(local_bottom_outputs.split(batch_size_per_gpu, dim=0)))
        # slice_bottom_outputs = torch.cat(bottom_output_buffer, dim=1).view(batch_size_per_gpu, -1, vector_dim)


        import numpy as np
        vector_total = np.array(vectors_per_gpu).sum()
        world_size = dist.get_world_size()

        bottom_output_buffer = torch.empty(
            batch_size_per_gpu*vector_total*vector_dim,
            device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype)

        input_splits = [int(local_bottom_outputs.view(-1).size()[0]/world_size)]*world_size
        output_splits = [int(batch_size_per_gpu*n*vector_dim) for n in vectors_per_gpu]

        torch.distributed.all_to_all_single(bottom_output_buffer, local_bottom_outputs.view(-1), output_splits, input_splits)
        slice_bottom_outputs = torch.cat([n.view(batch_size_per_gpu, -1, vector_dim) for n in list(bottom_output_buffer.split(output_splits))],dim=1)

        return slice_bottom_outputs

    @staticmethod
    def backward(ctx, grad_slice_bottom_outputs):
        rank = dist.get_rank()

        # grad_local_bottom_outputs = torch.empty(
        #     ctx.batch_size_per_gpu * ctx.world_size, ctx.vectors_per_gpu[rank] * ctx.vector_dim,
        #     device=grad_slice_bottom_outputs.device,
        #     dtype=grad_slice_bottom_outputs.dtype)
        # # All to all only takes list while split() returns tuple
        # grad_local_bottom_outputs_split = list(grad_local_bottom_outputs.split(ctx.batch_size_per_gpu, dim=0))

        # split_grads = [t.contiguous() for t  in (grad_slice_bottom_outputs.view(ctx.batch_size_per_gpu, -1).split(
        #     [ctx.vector_dim * n for n in ctx.vectors_per_gpu], dim=1))]

        # torch.distributed.all_to_all(grad_local_bottom_outputs_split, split_grads)

        grad_local_bottom_outputs = torch.empty(
            ctx.vectors_per_gpu[rank] * ctx.vector_dim * ctx.world_size, ctx.batch_size_per_gpu,
            device=grad_slice_bottom_outputs.device,
            dtype=grad_slice_bottom_outputs.dtype)

        grad_input = torch.transpose(grad_slice_bottom_outputs.view(ctx.batch_size_per_gpu, -1), 0, 1).contiguous()
        grad_input_splits = [ctx.vector_dim * n for n in ctx.vectors_per_gpu]
   
        grad_output_splits = [ctx.vectors_per_gpu[rank] * ctx.vector_dim] * ctx.world_size

        torch.distributed.all_to_all_single(grad_local_bottom_outputs, grad_input, grad_output_splits, grad_input_splits)

        grad_local_bottom_outputs = grad_local_bottom_outputs.view(ctx.world_size, -1, ctx.batch_size_per_gpu).permute(0,2,1).contiguous()
        grad_local_bottom_outputs = grad_local_bottom_outputs.view(ctx.world_size*ctx.batch_size_per_gpu, -1)

        return grad_local_bottom_outputs.view(grad_local_bottom_outputs.shape[0], -1, ctx.vector_dim), None, None, None

bottom_to_top = BottomToTop.apply

class DistDlrm():
    """Wrapper of top and bottom model

    To make interface simpler, this wrapper class is created to have bottom and top model in the same class.
    """

    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
                 embedding_dim, world_num_categorical_features, interaction_op="dot",
                 hash_indices=False, device="cuda", use_embedding_ext=True):
        super(DistDlrm, self).__init__()
        self.embedding_dim = embedding_dim
        self.bottom_model = DlrmBottom(
            num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, embedding_dim,
            hash_indices=hash_indices, device=device, use_embedding_ext=use_embedding_ext)

        num_interaction_inputs = world_num_categorical_features + 1
        self.top_model = DlrmTop(top_mlp_sizes, num_interaction_inputs,
                                 embedding_dim=embedding_dim, interaction_op=interaction_op).to(device)

    def __call__(self, numerical_input, categorical_inputs):
        """Single GPU forward"""
        assert dist.get_world_size() == 1  # DONOT run this in distributed mode
        bottom_out = self.bottom_model(numerical_input, categorical_inputs)
        top_out = self.top_model(bottom_out)

        return top_out

    def __repr__(self):
        s = F"{self.__class__.__name__}{{\n"
        s += repr(self.bottom_model)
        s += "\n"
        s += repr(self.top_model)
        s += "\n}\n"
        return s

    def to(self, *args, **kwargs):
        self.bottom_model.to(*args, **kwargs)
        self.top_model.to(*args, **kwargs)
