import torch

from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader, UniformNodeSampler
from torch_geometric.testing import get_random_edge_index


def test_uniform_node_sampler():
    sampler = UniformNodeSampler(num_nodes=100, batch_size=25, num_steps=4)
    assert len(sampler) == 4

    batches = list(sampler)
    assert len(batches) == 4
    for batch in batches:
        assert len(batch) == 25
        assert len(set(batch)) == 25  # i.i.d. *without* replacement per batch.
        assert min(batch) >= 0 and max(batch) < 100

    # Independent draws across batches should (almost surely) differ:
    assert any(batches[0] != other for other in batches[1:])


def test_random_node_loader_uniform_sampling():
    data = Data()
    data.x = torch.randn(100, 16)
    data.node_id = torch.arange(100)
    data.edge_index = get_random_edge_index(100, 100, 500)
    data.edge_attr = torch.randn(500, 8)

    # RNS mode wires `UniformNodeSampler` into the existing call site:
    loader = RandomNodeLoader(data, num_parts=4, sampling='uniform')
    assert loader.sampling == 'uniform'
    assert len(loader) == 4

    for batch in loader:
        # Each batch is the induced subgraph of a uniform node sample:
        assert batch.num_nodes == 25
        assert batch.node_id.min() >= 0
        assert batch.node_id.max() < 100
        assert batch.edge_index.size(1) == batch.edge_attr.size(0)
        assert torch.allclose(batch.x, data.x[batch.node_id])
        batch.validate()

    # Independent uniform draws differ across epochs, unlike a fixed partition:
    first = torch.cat([b.node_id for b in loader])
    second = torch.cat([b.node_id for b in loader])
    assert not torch.equal(first, second)


def test_random_node_loader_default_is_partition():
    data = Data()
    data.x = torch.randn(40, 8)
    data.node_id = torch.arange(40)
    data.edge_index = get_random_edge_index(40, 40, 120)

    loader = RandomNodeLoader(data, num_parts=4)
    assert loader.sampling == 'partition'

    # A partition covers every node exactly once per epoch:
    covered = torch.cat([batch.node_id for batch in loader])
    assert torch.equal(covered.sort().values, torch.arange(40))
