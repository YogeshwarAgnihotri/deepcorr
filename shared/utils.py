import os

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_path_throw_error(path):
    if not os.path.exists(path):
        raise IOError("Error: path %s does not exist." % path)
