r"""Uniform Random Node Sampling (RNS) for mini-batch GNN training.

This module implements the *Random Node Sampling* scheme analyzed in
`"Implicit Regularization of Mini-Batch Training in Graph Neural Networks"
<https://arxiv.org/abs/2605.22480v1>`_. RNS draws every mini-batch as a fresh
set of uniformly sampled nodes and trains on its induced subgraph.

This differs from a fixed *partition* of the nodes (the default behavior of
:class:`~torch_geometric.loader.RandomNodeLoader`), in which a single shuffled
permutation is split into ``num_parts`` chunks so that each node appears in
exactly one mini-batch per epoch. Under RNS the mini-batches are instead
independent (i.i.d.) uniform draws. The paper shows -- via backward error
analysis of graph mini-batch SGD -- that this i.i.d. uniform sampling produces
mini-batches whose expected loss is closer to the full-graph loss and whose
per-batch gradients have *lower variance*, acting as an implicit regularizer
and matching or outperforming structure-aware samplers on most benchmarks.
"""
from typing import Iterator, List, Optional

import torch


class UniformNodeSampler:
    r"""A batch sampler that yields independent, uniformly-sampled batches of
    node indices, realizing the Random Node Sampling (RNS) scheme.

    Each yielded batch is an i.i.d. uniform draw of :obj:`batch_size` distinct
    node indices from :obj:`{0, ..., num_nodes - 1}`. It is intended to be
    passed as the :obj:`batch_sampler` of a
    :class:`torch.utils.data.DataLoader` whose :obj:`collate_fn` materializes
    the induced subgraph of the sampled nodes.

    Args:
        num_nodes (int): The total number of nodes to sample from.
        batch_size (int): The number of nodes drawn for each mini-batch.
        num_steps (int, optional): The number of mini-batches per epoch. If
            set to :obj:`None`, defaults to :obj:`ceil(num_nodes / batch_size)`
            so that an epoch sees a comparable number of nodes to a single full
            pass. (default: :obj:`None`)
        generator (torch.Generator, optional): A random number generator used
            to draw the samples, for reproducibility. (default: :obj:`None`)
    """
    def __init__(
        self,
        num_nodes: int,
        batch_size: int,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if num_nodes <= 0:
            raise ValueError(f"'num_nodes' must be positive (got {num_nodes})")
        if batch_size <= 0:
            raise ValueError(
                f"'batch_size' must be positive (got {batch_size})")

        self.num_nodes = num_nodes
        # A mini-batch can hold at most every node exactly once:
        self.batch_size = min(batch_size, num_nodes)
        if num_steps is None:
            num_steps = (num_nodes + self.batch_size - 1) // self.batch_size
        if num_steps <= 0:
            raise ValueError(f"'num_steps' must be positive (got {num_steps})")
        self.num_steps = num_steps
        self.generator = generator

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_steps):
            # Uniform sampling *without* replacement within a batch, but
            # *independent* across batches -- the defining property of RNS:
            perm = torch.randperm(self.num_nodes, generator=self.generator)
            yield perm[:self.batch_size].tolist()
