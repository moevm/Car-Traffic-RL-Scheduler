from sys import argv
from Facade.Facade import Facade

if __name__ == "__main__":
    facade = Facade(argv[1])
    facade.execute()