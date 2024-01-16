#!/bin/bash

# List of configuration files and corresponding run names
declare -a configs=(
    "/home/yagnihotri/projects/corr/lightcorr/config/simple_grid.yaml"
    "/home/yagnihotri/projects/corr/lightcorr/config/second_grid.yaml"
    "/home/yagnihotri/projects/corr/lightcorr/config/best_grid.yaml"
)
declare -a names=("simple_grid_run_5000_1000_split" "second_grid_run_5000_1000_split" "best_grid_run_5000_1000_split")

# Length of the configs array
len=${#configs[@]}

# Loop through the configs and names arrays
for (( i=0; i<$len; i++ )); do
    config_path=${configs[$i]}
    run_name=${names[$i]}

    # Run the Python script with the current configuration and run name
    python /home/yagnihotri/projects/corr/lightcorr/main.py --config_path $config_path --run_name $run_name
done
