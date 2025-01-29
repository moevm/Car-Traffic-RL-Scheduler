from dataclasses import dataclass

@dataclass
class NodeData:
    node_id: str
    last_path: list
    start_nodes_ids: list
    path_length_meters: list
    path_length_edges: list
    last_route_id: int

@dataclass
class SimulationParams:
    duration: int
    initialization_delay: int
    iterations: int
    intensities: list
    poisson_generators_edges: list
