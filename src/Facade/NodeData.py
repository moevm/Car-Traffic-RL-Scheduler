from dataclasses import dataclass

@dataclass
class NodeData:
    node_id: str
    paths: list
    start_nodes_ids: list
    path_length_meters: list
    path_length_edges: list
    routes_ids: list