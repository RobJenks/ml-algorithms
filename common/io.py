import sys
import os.path


def read_file(path):
    with open(os.path.join(sys.path[0], path), 'r') as file:
        return file.read()
