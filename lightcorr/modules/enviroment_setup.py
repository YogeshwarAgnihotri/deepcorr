import sys
import os

from shared.utils import (
    StreamToLogger, 
    setup_logger, 
    create_run_folder,
)

from modules.config_utlis import (
    config_checks_training, 
    load_config,
)

def setup_environment(args):
    config = load_config(args.config_path)
    config_checks_training(config)

    run_folder_path = create_run_folder(
        config['run_folder_path'], args.run_name
    )
    output_file_path = os.path.join(run_folder_path, "training_log.txt")
    logger = setup_logger('TrainingLogger', output_file_path)
    sys.stdout = StreamToLogger(logger, sys.stdout)

    return config, run_folder_path