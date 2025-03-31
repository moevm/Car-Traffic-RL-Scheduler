import enum

from dataclasses import dataclass, field


@dataclass
class NodeData:
    node_id: str = ""
    last_paths: list = field(default_factory=list[list[str]])
    start_nodes_ids: list = field(default_factory=list[str])
    path_length_meters: list = field(default_factory=list[float])
    path_length_edges: list = field(default_factory=list[int])
    last_routes_ids: list = field(default_factory=list[list[str]])
    start_nodes_ids_counter: dict = field(default_factory=dict[str, int])


@dataclass
class SimulationParams:
    DURATION: int
    INIT_DELAY: int
    ITERATIONS: int
    PART_OF_THE_PATH: float
    CHECK_TIME: int
    CPU_SCALE: int
    intensities: list[float]
    poisson_generators_edges: list[str]
    turned_off_traffic_lights: list[str]


@dataclass
class NodePair:
    node_from: str
    node_to: str
