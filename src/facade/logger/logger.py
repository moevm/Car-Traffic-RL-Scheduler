from abc import ABC, abstractmethod
from progress.bar import Bar
import enum

class Message(enum.Enum):
    init_incidence_matrix = "Construction of incidence matrix..."
    init_restore_path_matrix = "Construction of restore-path matrix..."
    last_target_nodes_data = "Last target nodes data:"
    target_nodes_data = "Target nodes data:"

class Logger(ABC):
    def __init__(self, logger_type):
        self._logger_type = logger_type
        self._bar = Bar("Processing")

    def init_progress_bar(self, text_message, length_of_bar):
        print(f"{self._logger_type} {text_message.value}")
        self._bar = Bar("Processing", max=length_of_bar)

    def step_progress_bar(self):
        self._bar.next()

    def destroy_progress_bar(self):
        self._bar.finish()