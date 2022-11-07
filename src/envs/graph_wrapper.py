import networkx as nx
import networkx.classes.reportviews as rv
from .types import *


class GraphWrapper:

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph
        pass

    def get_nodes(self) -> rv.NodeView:
        return self._graph.nodes

    def get_all_edges(self) -> rv.OutEdgeView:
        return self._graph.edges

    def get_out_edges(self, i: Node) -> rv.OutEdgeView:
        return self._graph.edges(i)

    def get_init_acc(self, n: Node) -> rv.NodeView:
        return self._graph.nodes[n]['accInit']

    def get_edge_time(self, o: Node, d: Node) -> Time:
        return self._graph.edges[o, d]['time']

    def node_list(self) -> [Node]:
        return list(self._graph)

    def size(self) -> int:
        return len(self._graph)

    def set_edge_time(self, o: Node, d: Node, t: Time):
        self._graph.edges[o, d]['time'] = t
