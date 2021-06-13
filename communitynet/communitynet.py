# Standard Library
from copy import deepcopy
from multiprocessing import cpu_count, Pool as ProcessPool

# Third Party
import torch
import torch.nn as nn
from torch_geometric.data import Data


def extract_communities(graph):
    return [Data(), ...]


def build_intercommunity_graph(graph):
    return ic_edge_index, ic_edge_attr


class EnsembleGNN(nn.Module):
    def __init__(self, base_gnns, num_jobs):
        super(EnsembleGNN, self).__init__()

        self.base_gnns = nn.ModuleList(base_gnns)
        self.num_jobs = num_jobs if num_jobs != -1 else cpu_count()

    def reset_parameters(self):
        for gnn in self.base_gnns:
            gnn.reset_parameters()

    def forward(self, graph):
        couples = zip(graph.communities, self.base_gnns)

        if not self.num_jobs:
            outputs = [gnn(community) for community, gnn in couples]
        else:
            pool = ProcessPool(processes=self.num_jobs)
            outputs = pool.map(lambda c, gnn: gnn(c), couples)
            pool.close()
            pool.join()

        return torch.stack(outputs)


class CommunityNet(nn.Module):
    def __init__(self, base_gnn, output_gnn, num_communities, num_jobs=-1):
        super(CommunityNet, self).__init__()

        self.ensemble_gnn = EnsembleGNN(
            base_gnns=[deepcopy(base_gnn) for _ in range(num_communities)],
            num_jobs=num_jobs
        )
        self.output_gnn = output_gnn

    def reset_parameters(self):
        self.ensemble_gnn.reset_parameters()
        self.output_gnn.reset_parameters()

    def forward(self, graph):
        intercommunity_graph = Data(
            x=self.ensemble_gnn(graph),
            edge_index=graph.intercommunity_edge_index,
            edge_attr=graph.intercommunity_edge_attr
        )
        return self.output_gnn(intercommunity_graph)
