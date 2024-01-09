import os
import logging
import yaml
import datetime
import pandas 
import numpy as np

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_path_throw_error(path):
    if not os.path.exists(path):
        raise IOError("Error: path %s does not exist." % path)

def format_time(seconds):
    """Format time in seconds to days, hours, minutes, and seconds."""
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s"

class StreamToLogger:
    """
    Custom stream object that redirects writes to both a logger and the original stdout.
    """
    def __init__(self, logger, orig_stdout):
        self.logger = logger
        self.orig_stdout = orig_stdout
        self.linebuf = ''  # Buffer to accumulate lines until a newline character

    def write(self, message):
        """
        Write the message to logger and the original stdout.
        Only log messages when a newline character is encountered.
        """
        self.linebuf += message
        if '\n' in message:
            self.flush()

    def flush(self):
        """Flush the stream by logging the accumulated line and clearing the buffer."""
        if self.linebuf.rstrip():
            self.logger.info(self.linebuf.rstrip())
        self.orig_stdout.write(self.linebuf)
        self.linebuf = ''

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(message)s')  # Only include the message in logs

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    return logger

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def create_run_folder(base_path):
    """
    Creates a directory for the current run, named with the current date and time.

    :param base_path: Base directory where the run folder will be created.
    :return: Path of the created run folder.
    """
    run_name = f"Date_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    run_folder_path = os.path.join(base_path, "runs", run_name)
    os.makedirs(run_folder_path, exist_ok=True)
    return run_folder_path

def export_dataframe_to_csv(df, file_name, folder_path, create_folder_if_not_exist=True):
    """
    Exports a given DataFrame to a CSV file in the specified folder.

    :param df: Pandas DataFrame to be exported.
    :param file_name: Name of the CSV file (including .csv extension).
    :param folder_path: Path to the folder where the CSV file will be saved.
    :param create_folder: If True, creates the folder if it doesn't exist. Default is True.
    """
    if create_folder_if_not_exist:
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    # Define the full file path
    file_path = os.path.join(folder_path, file_name)

    # Export the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    print(f"Data exported to {file_path}")

def save_array_to_file(array, file_name):
    """
    Save a NumPy array to a text file.

    :param array: NumPy array to be saved.
    :param file_name: Name of the file where the array will be saved.
    """
    with open(file_name, 'w') as file:
        # Write the shape of the array for reference
        file.write(f"Array Shape: {array.shape}\n\n")

        # Use numpy array2string to handle the formatting
        # Set the threshold to np.inf to ensure the whole array is printed
        array_str = np.array2string(array, threshold=np.inf)
        file.write(array_str)

def save_plot_to_path(fig, file_name, save_path):
    create_path(os.path.dirname(save_path))
    # Save the figure to the specified file
    fig.savefig(os.path.join(save_path, file_name), bbox_inches='tight')