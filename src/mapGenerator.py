import sys, random, os, json
from sys import argv


def get_grid_cli():
    with open('grid-config.json', 'r') as file:
        cli = json.load(file)
    return cli

def get_spider_cli():
    with open('spider-config.json', 'r') as file:
        cli = json.load(file)
    return cli

def get_random_cli():
    with open('rand-config.json', 'r') as file:
        cli = json.load(file)
    return cli

def get_netgenerate_cli(cli):
    cli.pop("--number-networks")
    net_types = ["-g", "-s", "-r"]
    for net_type in net_types:
        if net_type in cli.keys():
            cli.pop(net_type)
            break
    netgenerate_cli = {}
    for key, value in cli.items():
        if type(cli[key]["low"]) == str:
            netgenerate_cli[key] = random.choice(list(cli[key].values()))
        elif type(cli[key]["low"]) == float:
            netgenerate_cli[key] = random.uniform(cli[key]["low"], cli[key]["high"])
        else:
            netgenerate_cli[key] = random.randint(cli[key]["low"], cli[key]["high"])
    return netgenerate_cli

def init_cli(argv_list):
    cli = {}
    grid_names = ["-g", "--grid"]
    spider_names = ["-s", "--spider"]
    random_names = ["-r", "--rand"]
    for grid_name in grid_names:
        if grid_name in argv_list:
            cli = get_grid_cli()
    for spider_name in spider_names:
        if spider_name in argv_list:
            cli = get_spider_cli()
    for random_name in random_names:
        if random_name in argv_list:
            cli = get_random_cli()
    return cli

def substitute_args(cli, argv_list):
    for i in range(len(argv_list)):
        if argv_list[i] == "--number-networks":
            cli["--number-networks"] = int(argv_list[i + 1])
            continue
        elif argv_list[i] in ["-g", "--grid", "-s", "--spider", "-r", "--rand"]:
            continue
        elif "-" in argv_list[i]:
            pos = argv_list[i].rfind('-')
            key = argv_list[i][:pos]
            value = argv_list[i][pos+1:]
            if value not in ["low", "high"]:
                print("Error: invalid arg value.")
                sys.exit()
            try:
                cli[key][value] = int(argv_list[i + 1])
            except IndexError:
                print("Error: invalid args sequence.")
                sys.exit()
            except ValueError:
                print("Error: invalid arg value.")
                sys.exit()

def handle_args(argv_list):
    cli = init_cli(argv_list)
    substitute_args(cli, argv_list)
    number_networks = cli["--number-networks"]
    match list(cli.items())[0][0]:
        case "-g":
            base_filename = "grid"
        case "-s":
            base_filename = "spider"
        case "-r":
            base_filename = "rand"
    for i in range(number_networks):
        netgenerate_cli = get_netgenerate_cli(cli.copy())
        netgenerate_cli[list(cli.items())[0][0]] = "true"
        command = 'netgenerate '
        for key, value in netgenerate_cli.items():
            command += str(key) + ' ' + str(value) + ' '
        command += '-o ' + base_filename + str(i) + '.net.xml'
        os.system(command)
        
if __name__ == "__main__":
    handle_args(argv[1:])
