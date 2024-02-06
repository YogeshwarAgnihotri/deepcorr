import subprocess
import itertools
from tqdm import tqdm

# Define the range of parameters you want to test
train_ratios = [0.7, 0.8, 0.9]
negative_samples = [1, 10, 50, 100, 199, 300]
load_all_data_options = [False, True]

# Generate all combinations of parameters
parameter_combinations = list(itertools.product(train_ratios, negative_samples, load_all_data_options))

# Path to your main script
script_path = '/home/yagnihotri/projects/corr/treecorr/main.py'

# Iterate over each combination and run the script
for params in tqdm(parameter_combinations, desc="Running parameter combinations"):
    train_ratio, negative_sample, load_all_data = params

    # Prepare the command to run your script with the current set of parameters
    command = [
        'python', script_path,
        '--train_ratio', str(train_ratio),
        '--negative_samples', str(negative_sample),
    ]

    # Add the --load_all_data flag only if it's set to True
    if load_all_data:
        command.append('--load_all_data')

    # Run the command
    subprocess.run(command)

# tqdm will automatically update the progress bar as each iteration of the loop completes.
