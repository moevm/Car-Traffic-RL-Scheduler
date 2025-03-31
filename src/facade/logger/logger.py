import enum

from abc import ABC, abstractmethod
from progress.bar import Bar


class Message(enum.Enum):
    init_incidence_matrix = "Construction of incidence matrix..."
    init_restore_path_matrix = "Construction of restore-path matrix..."
    last_target_nodes_data = "Last target nodes data:"
    target_nodes_data = "Target nodes data:"
    search_for_valid_edges = "Search for valid edges:"
    find_way_back_paths = "Search for return paths:"
    init_restore_path_matrix_for_cycles = "Construction of restore-path matrix for start nodes of cycles..."
    find_all_routes = "Finding all routes from extreme and poisson generators nodes..."
    stabilization_of_initial_traffic = "Stabilization of initial traffic..."
    training_started = "Training started..."

class Logger(ABC):
    def __init__(self, logger_type: str):
        self._logger_type = logger_type
        self._bar = Bar("Processing")

    def print_info(self, text_message: Message):
        print(f"{self._logger_type} {text_message.value}")

    def init_progress_bar(self, text_message: Message, length_of_bar: int) -> None:
        self.print_info(text_message)
        self._bar = Bar("Processing", max=length_of_bar)

    def step_progress_bar(self) -> None:
        self._bar.next()

    def destroy_progress_bar(self) -> None:
        self._bar.finish()
