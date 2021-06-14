# Standard Library
from copy import deepcopy
from multiprocessing import cpu_count, Pool as ProcessPool

# Third Party
import torch
import torch.nn as nn
from torch_geometric.data import Data


def _extract_communities(graph):
    communities = []
    for community in graph.communities:
        node_to_c_node = {n: i for i, n in enumerate(community)}
        community_edge_indices, community_edge_attrs = [], []
        for i, edge in enumerate(graph.edge_index.t()):
            is_within_community = sum(edge == n for n in community).bool().all()
            if not is_within_community:
                continue

            c_edge = torch.tensor(
                [node_to_c_node[n] for n in edge],
                dtype=torch.long
            )
            community_edge_indices.append(c_edge)
            community_edge_attrs.append(graph.edge_attr[i])

        c_edge_index = torch.stack(community_edge_indices).t().contiguous()
        c_edge_attr = torch.stack(community_edge_attrs)

        community_graph = Data(
            x=graph.x[community],
            edge_index=c_edge_index,
            edge_attr=c_edge_attr
        )
        communities.append(community_graph)

    return communities


def _build_intercommunity_graph(graph):
    return ic_edge_index, ic_edge_attr # TODO


class EnsembleGNN(nn.Module):
    def __init__(self, base_gnns, num_jobs):
        super(EnsembleGNN, self).__init__()

        self.base_gnns = nn.ModuleList(base_gnns)
        self.num_jobs = num_jobs if num_jobs != -1 else cpu_count()

    def reset_parameters(self):
        for gnn in self.base_gnns:
            gnn.reset_parameters()

    def forward(self, graph):
        couples = zip(_extract_communities(graph), self.base_gnns)

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
        ic_edge_index, ic_edge_attr = _build_intercommunity_graph(graph)
        intercommunity_graph = Data(
            x=self.ensemble_gnn(graph),
            edge_index=ic_edge_index,
            edge_attr=ic_edge_attr
        )
        return self.output_gnn(intercommunity_graph)
