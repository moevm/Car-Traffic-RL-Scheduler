import sys
import random
import os
import json
import xml.etree.ElementTree as ET

from sys import argv


def get_grid_cli() -> dict:
    with open("base-cli-params/grid-config.json", "r") as file:
        base_cli = json.load(file)
    return base_cli


def get_spider_cli() -> dict:
    with open("base-cli-params/spider-config.json", "r") as file:
        base_cli = json.load(file)
    return base_cli


def get_random_cli() -> dict:
    with open("base-cli-params/rand-config.json", "r") as file:
        base_cli = json.load(file)
    return base_cli


def get_netgenerate_cli(cli: dict) -> dict:
    netgenerate_cli = {}
    for key, value in cli.items():
        if type(cli[key]) == dict:
            if type(cli[key]["low"]) == str:
                netgenerate_cli[key] = random.choice(list(cli[key].values()))
            elif type(cli[key]["low"]) == float:
                netgenerate_cli[key] = random.uniform(cli[key]["low"], cli[key]["high"])
            else:
                netgenerate_cli[key] = random.randint(cli[key]["low"], cli[key]["high"])
    return netgenerate_cli


def init_cli(argv_list: list) -> dict:
    cli_dict = {
        "-g": get_grid_cli(),
        "--grid": get_grid_cli(),
        "-s": get_spider_cli(),
        "--spider": get_spider_cli(),
        "-r": get_random_cli(),
        "--rand": get_random_cli()
    }
    for net_type in cli_dict.keys():
        for arg in argv_list:
            if net_type == arg:
                base_cli = cli_dict[net_type]
                break
    return base_cli


def substitute_args(base_cli: dict, cli: dict, argv_list: list) -> None:
    for i in range(len(argv_list)):
        if argv_list[i] == "--number-networks":
            cli["--number-networks"] = int(argv_list[i + 1])
            continue
        elif argv_list[i] in ["-g", "--grid", "-s", "--spider", "-r", "--rand"]:
            continue
        elif "-" in argv_list[i]:
            pos = argv_list[i].rfind("-")
            main_key = argv_list[i][:pos]
            sub_key = argv_list[i][pos + 1:]
            if sub_key not in ["low", "high"]:
                print("Error: invalid arg value.")
                sys.exit()
            cli[main_key] = base_cli[main_key]
            try:
                base_cli_type = type(base_cli[main_key]["low"])
                cli[main_key][sub_key] = base_cli_type(argv_list[i + 1])
            except IndexError:
                print("Error: invalid args sequence.")
                sys.exit()
            except ValueError:
                print("Error: invalid arg value.")
                sys.exit()
    return


def assign_grid_attach_params(base_cli: dict, netgenerate_cli: dict) -> None:
    grid_params_dict = {}
    grid_params = ["--grid.x-length", "--grid.y-length", "--grid.length"]
    for param in grid_params:
        grid_params_dict[param] = (base_cli[param]["low"] + base_cli[param]["high"]) / 2
    for param in grid_params:
        if param in netgenerate_cli.keys():
            grid_params_dict[param] = netgenerate_cli[param]
    for param in grid_params_dict.keys():
        pos = param.rfind('l')
        netgenerate_cli[param[:pos] + "attach-length"] = grid_params_dict[param]
    return


def assign_spider_attach_params(base_cli: dict, netgenerate_cli: dict) -> None:
    param = "--spider.space-radius"
    spider_attach_dict = {param: (base_cli[param]["low"] +
                                  base_cli[param]["high"]) / 2}
    if param in netgenerate_cli.keys():
        spider_attach_dict[param] = netgenerate_cli[param]
    netgenerate_cli["--spider.attach-length"] = spider_attach_dict[param]
    return


def make_sumocfg(net_name, base_filename):
    configuration = ET.Element("configuration")
    input_element = ET.SubElement(configuration, "input")
    ET.SubElement(input_element, "net-file", value=f"{net_name}.net.xml")
    tree = ET.ElementTree(configuration)
    tree.write(f"configs/{base_filename}-configs/{net_name}.sumocfg")


def handle_args(argv_list: list) -> None:
    base_cli = init_cli(argv_list)
    cli = {list(base_cli.items())[0][0]: "true"}
    substitute_args(base_cli, cli, argv_list)
    match list(cli.items())[0][0]:
        case "-g":
            base_filename = "grid"
        case "-s":
            base_filename = "spider"
        case "-r":
            base_filename = "rand"
    for i in range(cli["--number-networks"]):
        netgenerate_cli = get_netgenerate_cli(cli.copy())
        netgenerate_cli[list(cli.items())[0][0]] = "true"
        if base_filename == "grid":
            assign_grid_attach_params(base_cli, netgenerate_cli)
        elif base_filename == "spider":
            assign_spider_attach_params(base_cli, netgenerate_cli)
        command = "netgenerate "
        for key, value in netgenerate_cli.items():
            command += f"{str(key)} {str(value)} "
        net_name = f"{base_filename + str(i)}"
        command += f"--tls.guess true --tls.guess.threshold 0 -o configs/{base_filename}-configs/{net_name}.net.xml"
        print(command)
        os.system(command)
        make_sumocfg(net_name, base_filename)


if __name__ == "__main__":
    handle_args(argv[1:])
