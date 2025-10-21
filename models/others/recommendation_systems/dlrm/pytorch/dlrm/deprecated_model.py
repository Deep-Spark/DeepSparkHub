"""Reimplementation of Facebook's DLRM model"""
import copy
import json
import math

from absl import logging

import torch
from torch import nn

import dlrm.nn
from dlrm.cuda_ext import dotBasedInteract

CRITEO_LARGE_EMBEDDING_IDS = [0, 9, 19, 20, 21]

class DlrmBase(nn.Module):
    """Base class of DLRM model

    Base class of DLRM model. There are several possible implementation of embeddings. Base model abstract embedding
    related things including creation and forward, and does the remaining (intitlization, interaction, MLP, etc.).

    Args:
        num_numerical_features (int): Number of dense features fed into bottom MLP
        categorical_feature_sizes (list): A list of integer indicating number of features of each embedding table
        bottom_mlp_sizes (list): A list of integer indicating bottom MLP layer sizes. Last bottom MLP layer
            must be embedding_dim
        top_mlp_sizes (list): A list of integers indicating top MLP layer sizes.
        embedding_dim (int): Length of embedding vectors. Default 32
        interaction_op (string): Type of interactions. Default "dot"
        self_interaction (bool): Default False.
        hash_indices (bool): If True, hashed_index = index % categorical_feature_size. Default False.
    """
    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
                 embedding_dim=32, interaction_op="dot", self_interaction=False, hash_indices=False):
        super(DlrmBase, self).__init__()
        if embedding_dim != bottom_mlp_sizes[-1]:
            raise TypeError("The last bottom MLP layer must have same size as embedding.")

        self._embedding_dim = embedding_dim
        self._interaction_op = interaction_op
        self._self_interaction = self_interaction
        self._hash_indices = hash_indices
        self._categorical_feature_sizes = copy.copy(categorical_feature_sizes)

        # Interactions are among outputs of all the embedding tables and bottom MLP, total number of
        # (num_embedding_tables + 1) vectors with size embdding_dim. ``dot`` product interaction computes dot product
        # between any 2 vectors. ``cat`` interaction concatenate all the vectors together.
        # Output of interaction will have shape [num_interactions, embdding_dim].
        num_interaction_inputs = len(categorical_feature_sizes) + 1
        if interaction_op == "dot":
            if self_interaction:
                raise NotImplementedError
            num_interactions = (num_interaction_inputs * (num_interaction_inputs - 1)) // 2 + embedding_dim
        elif interaction_op == "cat":
            num_interactions = num_interaction_inputs * embedding_dim
        else:
            raise TypeError(F"Unknown interaction {interaction_op}.")

        self.embeddings = nn.ModuleList()
        self._create_embeddings(self.embeddings, embedding_dim, categorical_feature_sizes)

        # Create bottom MLP
        bottom_mlp_layers = []
        input_dims = num_numerical_features
        for output_dims in bottom_mlp_sizes:
            bottom_mlp_layers.append(
                nn.Linear(input_dims, output_dims))
            bottom_mlp_layers.append(nn.ReLU(inplace=True))
            input_dims = output_dims
        self.bottom_mlp = nn.Sequential(*bottom_mlp_layers)

        # Create Top MLP
        top_mlp_layers = []
        input_dims = num_interactions + 1  # pad 1 to be multiple of 8
        for output_dims in top_mlp_sizes[:-1]:
            top_mlp_layers.append(nn.Linear(input_dims, output_dims))
            top_mlp_layers.append(nn.ReLU(inplace=True))
            input_dims = output_dims
        # last Linear layer uses sigmoid
        top_mlp_layers.append(nn.Linear(input_dims, top_mlp_sizes[-1]))
        top_mlp_layers.append(nn.Sigmoid())
        self.top_mlp = nn.Sequential(*top_mlp_layers)

        self._initialize_mlp_weights()

    def _interaction(self, bottom_mlp_output, embedding_outputs, batch_size):
        """Interaction

        "dot" interaction is a bit tricky to implement and test. Break it out from forward so that it can be tested
        independently.

        Args:
            bottom_mlp_output (Tensor):
            embedding_outputs (list): Sequence of tensors
            batch_size (int):
        """
        concat = torch.cat([bottom_mlp_output] + embedding_outputs, dim=1)
        if self._interaction_op == "dot" and not self._self_interaction:
            concat = concat.view((batch_size, -1, self._embedding_dim))
            if concat.dtype == torch.half:
                interaction_output = dotBasedInteract(concat, bottom_mlp_output)
            else:  # Legacy path
                interaction = torch.bmm(concat, torch.transpose(concat, 1, 2))
                tril_indices_row, tril_indices_col = torch.tril_indices(
                    interaction.shape[1], interaction.shape[2], offset=-1)
                interaction_flat = interaction[:, tril_indices_row, tril_indices_col]

                # concatenate dense features and interactions
                zero_padding = torch.zeros(
                    concat.shape[0], 1, dtype=concat.dtype, device=concat.device)
                interaction_output = torch.cat((bottom_mlp_output, interaction_flat, zero_padding), dim=1)

        elif self._interaction_op == "cat":
            interaction_output = concat
        else:
            raise NotImplementedError

        return interaction_output

    def _initialize_mlp_weights(self):
        """Initializing weights same as original DLRM"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, 0., math.sqrt(2. / (module.in_features + module.out_features)))
                nn.init.normal_(module.bias.data, 0., math.sqrt(1. /  module.out_features))

        # Explicitly set weight corresponding to zero padded interaction output. They will
        # stay 0 throughout the entire training. An assert can be added to the end of the training
        # to prove it doesn't increase model capacity but just 0 paddings.
        nn.init.zeros_(self.top_mlp[0].weight[:, -1].data)

    # pylint:disable=missing-docstring
    def _create_embeddings(self, embeddings, embedding_dim, categorical_feature_sizes):
        raise NotImplementedError

    @property
    def num_categorical_features(self):
        return len(self._categorical_feature_sizes)


    def forward(self, numerical_input, categorical_inputs):
        raise NotImplementedError

    def extra_repr(self):
        s = (F"interaction_op={self._interaction_op}, self_interaction={self._self_interaction}, "
             F"hash_indices={self._hash_indices}")
        return s
    # pylint:enable=missing-docstring

    @classmethod
    def from_json(cls, json_str, **kwargs):
        """Create from json str"""
        obj_dict = json.loads(json_str)
        return cls(**obj_dict, **kwargs)

class Dlrm(DlrmBase):
    """Reimplement Facebook's DLRM model

    Original implementation is from https://github.com/facebookresearch/dlrm.

    """
    def __init__(self, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
                 embedding_dim=32, interaction_op="dot", self_interaction=False, hash_indices=False,
                 base_device="cuda", mem_donor_devices=None):
        # Running everything on gpu by default
        self._base_device = base_device
        self._embedding_device_map = [base_device for _ in range(len(categorical_feature_sizes))]
        if mem_donor_devices is not None:
            for i, large_embedding_id in enumerate(CRITEO_LARGE_EMBEDDING_IDS):
                self._embedding_device_map[large_embedding_id] = mem_donor_devices[i]

        super(Dlrm, self).__init__(
            num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes,
            embedding_dim, interaction_op, self_interaction, hash_indices)

    def _create_embeddings(self, embeddings, embedding_dim, categorical_feature_sizes):
        # Each embedding table has size [num_features, embedding_dim]
        for i, num_features in enumerate(categorical_feature_sizes):
            # Allocate directly on GPU is much faster than allocating on CPU then copying over
            embedding_weight = torch.empty((num_features, embedding_dim), device=self._embedding_device_map[i])
            embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False, sparse=True)

            # Initializing embedding same as original DLRM
            nn.init.uniform_(
                embedding.weight.data,
                -math.sqrt(1. / embedding.num_embeddings),
                math.sqrt(1. / embedding.num_embeddings))

            embeddings.append(embedding)

    def set_devices(self, base_device, mem_donor_devices=None):
        """Set devices to run the model

        Put small embeddings and MLPs on base device. Put 5 largest embeddings on mem_donar_devices. 5 largest
        embeddings are [0, 9, 19, 20, 21].

        Args:
            base_device (string);
            mem_donor_devices (list): list of 5 strings indicates where to run 5 largest embeddings. Default None.
        """
        self._base_device = base_device
        self.bottom_mlp.to(base_device)
        self.top_mlp.to(base_device)
        self._embedding_device_map = [base_device for _ in range(self.num_categorical_features)]

        if mem_donor_devices is not None:
            if len(mem_donor_devices) != 5:
                raise ValueError(F"Must specify 5 mem_donor_devices, got {len(mem_donor_devices)}.")

            for i, large_embedding_id in enumerate(CRITEO_LARGE_EMBEDDING_IDS):
                self._embedding_device_map[large_embedding_id] = mem_donor_devices[i]

        for embedding_id, device in enumerate(self._embedding_device_map):
            logging.info("Place embedding %d on device %s", embedding_id, device)
            self.embeddings[embedding_id].to(device)

    def forward(self, numerical_input, categorical_inputs):
        """

        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]
        """
        batch_size = numerical_input.size()[0]
        # TODO(haow): Maybe check batch size of sparse input

        # Put indices on the same device as corresponding embedding
        device_indices = []
        for embedding_id, embedding in enumerate(self.embeddings):
            device_indices.append(categorical_inputs[embedding_id].to(self._embedding_device_map[embedding_id]))

        bottom_mlp_output = self.bottom_mlp(numerical_input)

        # embedding_outputs will be a list of (26 in the case of Criteo) fetched embeddings with shape
        # [batch_size, embedding_size]
        embedding_outputs = []
        for embedding_id, embedding in enumerate(self.embeddings):
            if self._hash_indices:
                device_indices[embedding_id] %= embedding.num_embeddings
                logging.log_first_n(
                    logging.WARNING, F"Hashed indices out of range.", 1)
            embedding_outputs.append(embedding(device_indices[embedding_id]).to(self._base_device))

        interaction_output = self._interaction(bottom_mlp_output, embedding_outputs, batch_size)

        top_mlp_output = self.top_mlp(interaction_output)

        return top_mlp_output

class DlrmJointEmbedding(DlrmBase):
    """DLRM uses one hot embedding only

    If all embeddings are one hot, they can be easily combined and will have better performance.
    """

    def _create_embeddings(self, embeddings, embedding_dim, categorical_feature_sizes):
        """Combine all one hot embeddings as one"""
        logging.warning("Combined all categorical features to single embedding table.")
        embeddings.append(dlrm.nn.BuckleEmbedding(categorical_feature_sizes, embedding_dim))

        for cat, size in enumerate(categorical_feature_sizes):
            module = embeddings[0]  # Only one embedding module in the ModuleList
            nn.init.uniform_(
                module.embedding.weight.data[module.offsets[cat]:module.offsets[cat + 1]],
                -math.sqrt(1. / size),
                math.sqrt(1. / size))

    def forward(self, numerical_input, categorical_inputs):
        """
        Args:
            numerical_input (Tensor): with shape [batch_size, num_numerical_features]
            categorical_inputs (Tensor): with shape [num_categorical_features, batch_size]
        """
        batch_size = numerical_input.size()[0]
        bottom_mlp_output = self.bottom_mlp(numerical_input)

        # Change indices based on hash_shift
        # It would be more efficient to change on the data loader side. But in order to keep the interface consistent
        # with the base Dlrm model, it is handled here.
        if self._hash_indices:
            for cat, size in enumerate(self._categorical_feature_sizes):
                categorical_inputs[cat] %= size
                logging.log_first_n(
                    logging.WARNING, F"Hashed indices out of range.", 1)

        # self._interaction takes list of tensor as input. So make this single element list
        # categorical_inputs is transposed here only to keep interface consistent with base model,
        # which makes it easy to test. Will change them to be the best performing version.
        # TODO(haow): Remove transpose.
        embedding_outputs = [self.embeddings[0](categorical_inputs.t()).view(batch_size, -1)]

        interaction_output = self._interaction(bottom_mlp_output, embedding_outputs, batch_size)

        top_mlp_output = self.top_mlp(interaction_output)

        return top_mlp_output
