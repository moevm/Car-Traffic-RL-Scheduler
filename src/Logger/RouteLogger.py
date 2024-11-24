from Logger.Logger import Logger
from prettytable import PrettyTable

class RouteLogger(Logger):
    def __init__(self):
        super().__init__("[RouteInfo]")

    def print_routes_data_info(self, message, target_nodes_data):
        number_of_displayed_target_nodes = 5
        first_column_size = 14
        second_column_size = 46
        max_size = 11
        print(f"{self._logger_type} {message.value}")
        table = PrettyTable()
        table.field_names = ["target node id", "number of paths, that ends in this target node"]
        if len(target_nodes_data) > max_size:
            for target_node_data in target_nodes_data[:number_of_displayed_target_nodes]:
                table.add_row([target_node_data.node_id, len(target_node_data.start_nodes_ids)])
            table.add_row(["." * first_column_size, "." * second_column_size])
            for target_node_data in target_nodes_data[-number_of_displayed_target_nodes:]:
                table.add_row([target_node_data.node_id, len(target_node_data.start_nodes_ids)])
        else:
            for target_node_data in target_nodes_data:
                table.add_row([target_node_data.node_id, len(target_node_data.start_nodes_ids)])
        print(f"{table}\n")
