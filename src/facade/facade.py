import json
import traci

from facade.environment.traffic_lights_env import TrafficLightsEnv
from facade.generation.route_generator import RouteGenerator
from facade.generation.transport_generator import TransportGenerator
from facade.logger.logger import *
from facade.net import Net
from facade.structures import SimulationParams
from stable_baselines3 import A2C


class Facade:
    def __init__(self, sumo_config: str, simulation_parameters_file: str):
        self.__SUMO_CONFIG = sumo_config
        self.__NET_CONFIG = self.__extract_net_config()
        self.__simulation_parameters_file = simulation_parameters_file
        self.__simulation_params = self.__get_simulation_params_from_file()
        self.__net = Net(self.__NET_CONFIG,
                         self.__simulation_params.poisson_generators_edges,
                         self.__simulation_params.CPU_SCALE
                         )
        self.__net.parallel_make_restore_path_matrix()
        self.__net.parallel_find_way_back()
        self.__net.parallel_find_routes()
        self.__route_generator = RouteGenerator(self.__net)
        self.__last_target_nodes_data = []
        self.__transport_generator = TransportGenerator(self.__simulation_params.intensities,
                                                        self.__simulation_params.poisson_generators_edges,
                                                        self.__simulation_params.DURATION, self.__net.get_edges(),
                                                        self.__simulation_params.PART_OF_THE_PATH)
        self.__step = 0
        self.__traffic_logger = Logger("[TrafficInfo]")
        self.__learning_logger = Logger("[LearningInfo]")

    def __extract_net_config(self) -> str:
        slash_position = self.__SUMO_CONFIG.rfind('/')
        extension_position = self.__SUMO_CONFIG.rfind('.sumocfg')
        net_name = f"{self.__SUMO_CONFIG[slash_position + 1:extension_position]}.net.xml"
        return self.__SUMO_CONFIG[:slash_position + 1] + net_name

    def __get_simulation_params_from_file(self) -> SimulationParams:
        with open(self.__simulation_parameters_file, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        simulation_params = SimulationParams(**data)
        return simulation_params

    def __generate_initial_traffic(self) -> None:
        while self.__step < self.__simulation_params.ITERATIONS * self.__simulation_params.INIT_DELAY:
            self.__transport_generator.clean_vehicles_data()
            if self.__step % self.__simulation_params.INIT_DELAY == 0:
                self.__route_generator.make_routes()
                self.__last_target_nodes_data = self.__route_generator.get_last_target_nodes_data()
                self.__transport_generator.generate_transport(self.__last_target_nodes_data)
            self.__step += 1
            traci.simulationStep()
        self.__route_generator.print_all_routes_data_info()
        self.__traffic_logger.print_info(Message.stabilization_of_initial_traffic)
        while self.__step < self.__simulation_params.DURATION:
            self.__transport_generator.clean_vehicles_data()
            if (self.__step % self.__simulation_params.CHECK_TIME == 0 and
                    self.__transport_generator.have_vehicles_passed_part_of_path_in_total()):
                break
            self.__step += 1
            traci.simulationStep()

    def __generate_main_traffic(self) -> None:
        schedule = self.__transport_generator.generate_schedule_for_poisson_flow(self.__step + 1)
        turned_on_traffic_lights_ids = self.__net.get_turned_on_traffic_lights_ids()
        unique_phases_states = set()
        for tls_id in turned_on_traffic_lights_ids:
            all_programs = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = all_programs.getPhases()
            for phase in phases:
                if phase.state not in unique_phases_states:
                    unique_phases_states.add(phase.state)
        unique_phases_states = list(unique_phases_states)
        unique_phases_states_dict = {unique_phases_states[i]: i for i in range(len(unique_phases_states))}
        env = TrafficLightsEnv(turned_on_traffic_lights_ids, schedule, self.__route_generator,
                               self.__transport_generator, self.__step, unique_phases_states_dict)
        self.__learning_logger.print_info(Message.training_started)
        model = A2C(policy='MultiInputPolicy', env=env, device='cuda',
                    tensorboard_log='./a2c_traffic_lights_tensorboard', n_steps=50)
        model.learn(total_timesteps=self.__simulation_params.DURATION, progress_bar=True,
                    tb_log_name='first_run')
        for tls_id in turned_on_traffic_lights_ids:
            all_programs = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            print(tls_id, all_programs)

    def execute(self) -> None:
        sumo_cmd = ["sumo-gui", "-c", self.__SUMO_CONFIG]
        traci.start(sumo_cmd)
        self.__net.turn_off_traffic_lights(self.__simulation_params.turned_off_traffic_lights)
        self.__generate_initial_traffic()
        self.__generate_main_traffic()
        traci.close()
