import networkx as nx
import networkx.classes.reportviews as rv
from src.misc.types import *


class GraphWrapper:

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph
        pass

    def get_all_edges(self) -> rv.OutEdgeView:
        return self._graph.edges

    def get_out_edges(self, i: Node) -> rv.OutEdgeView:
        return self._graph.edges(i)

    def node_list(self) -> [Node]:
        return list(self._graph)

    def size(self) -> int:
        return len(self._graph)

