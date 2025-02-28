from dataclasses import dataclass
import enum

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
    DURATION: int
    INIT_DELAY: int
    ITERATIONS: int
    PART_OF_THE_PATH: float
    CHECK_TIME: int
    intensities: list[float]
    poisson_generators_edges: list[str]
    turned_off_traffic_lights: list[str]


class NodeColor(enum.Enum):
    white = 0
    grey = 1
    black = 2

@dataclass
class NodePair:
    node_from: str
    node_to: str