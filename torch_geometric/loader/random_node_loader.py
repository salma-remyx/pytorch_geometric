import math
from typing import Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from torch_geometric.loader.random_node_sampling import UniformNodeSampler


class RandomNodeLoader(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        sampling (str, optional): The mini-batch construction scheme. If set to
            :obj:`"partition"`, a single shuffled permutation of the nodes is
            split into :obj:`num_parts` chunks, so every node appears in exactly
            one mini-batch per epoch. If set to :obj:`"uniform"`, each
            mini-batch is instead an independent (i.i.d.) uniform draw of nodes
            -- the Random Node Sampling (RNS) scheme of `"Implicit
            Regularization of Mini-Batch Training in Graph Neural Networks"
            <https://arxiv.org/abs/2605.22480v1>`_, whose i.i.d. mini-batches
            yield lower-variance gradients and act as an implicit regularizer.
            (default: :obj:`"partition"`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        sampling: str = 'partition',
        **kwargs,
    ):
        self.data = data
        self.num_parts = num_parts

        if isinstance(data, HeteroData):
            edge_index, node_dict, edge_dict = to_homogeneous_edge_index(data)
            self.node_dict, self.edge_dict = node_dict, edge_dict
        else:
            edge_index = data.edge_index

        self.edge_index = edge_index
        self.num_nodes = data.num_nodes
        self.sampling = sampling

        batch_size = math.ceil(self.num_nodes / num_parts)
        if sampling == 'uniform':  # Random Node Sampling (RNS):
            # `batch_sampler` is mutually exclusive with `batch_size`/`shuffle`,
            # which the i.i.d. uniform draws subsume.
            kwargs.pop('shuffle', None)
            super().__init__(
                range(self.num_nodes),
                batch_sampler=UniformNodeSampler(
                    self.num_nodes, batch_size, num_steps=num_parts,
                    generator=kwargs.pop('generator', None)),
                collate_fn=self.collate_fn,
                **kwargs,
            )
        elif sampling == 'partition':
            super().__init__(
                range(self.num_nodes),
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown sampling scheme '{sampling}' "
                             f"(expected 'partition' or 'uniform')")

    def collate_fn(self, index):
        if not isinstance(index, Tensor):
            index = torch.tensor(index)

        if isinstance(self.data, Data):
            return self.data.subgraph(index)

        elif isinstance(self.data, HeteroData):
            node_dict = {
                key: index[(index >= start) & (index < end)] - start
                for key, (start, end) in self.node_dict.items()
            }
            return self.data.subgraph(node_dict)
