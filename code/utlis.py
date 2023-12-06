import os

def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_path_throw_error(path):
    if not os.path.exists(path):
        raise IOError("Error: path %s does not exist." % path)

def create_run_folder(path_for_saving_run, run_name):
    run_path = os.path.join(path_for_saving_run, run_name)
    check_create_path(run_path)
    return run_path