import numpy as np
import tqdm
import os
import pickle

from shared.utils import create_path, save_array_to_file

def generate_flow_pairs_to_memmap(dataset,train_index,test_index,flow_size, memmap_saving_path, negetive_samples):
    print(f"\nGenerating all flow pairs (including {negetive_samples} negative flow pairs for each flow pair in dataset)...")
    all_samples=len(train_index)

    # if it dosent exit create the path
    create_path(memmap_saving_path)

    # creating paths for memmap files
    labels_path = os.path.join(memmap_saving_path, "training_labels")
    l2s_path = os.path.join(memmap_saving_path, "training_flow_pairs")
    labels_test_path = os.path.join(memmap_saving_path, "test_labels")
    l2s_test_path = os.path.join(memmap_saving_path, "test_flow_pairs")
    
    # Memmap creation
    labels = np.memmap(labels_path, dtype=np.float32, mode='w+', shape=(all_samples*(negetive_samples+1),1))
    #labels=np.zeros((all_samples*(negetive_samples+1),1))
    l2s = np.memmap(l2s_path, dtype=np.float32, mode='w+', shape=(all_samples*(negetive_samples+1),8,flow_size,1))
    #l2s=np.zeros((all_samples*(negetive_samples+1),8,flow_size,1))

    index=0
    random_ordering=[]+train_index
    for i in tqdm.tqdm(train_index, desc="Training data"):
        #[]#list(lsh.find_k_nearest_neighbors((Y_train[i]/ np.linalg.norm(Y_train[i])).astype(np.float64),(50)))

        #Saving True Pair
        #The *1000 and /1000 for normalization? 
        # "There/here"[0] are the interpacket delays
        l2s[index,0,:,0] = minimum_padding(np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
        l2s[index,1,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0, min_length=flow_size)
        l2s[index,2,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
        l2s[index,3,:,0] = minimum_padding(np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0, min_length=flow_size)

        # "There/here"[1] are the packet sizes
        l2s[index,4,:,0] = minimum_padding(np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
        l2s[index,5,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
        l2s[index,6,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
        l2s[index,7,:,0] = minimum_padding(np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0, min_length=flow_size)

        if index % (negetive_samples+1) !=0:
            #print(index, len(nears))
            print(index)
            raise
        labels[index,0]=1
        m=0
        index+=1
        np.random.shuffle(random_ordering)
        #After adding the true pair to the l2s array, now create negative_samples times false pairs
        for idx in random_ordering:
            if idx==i or m>(negetive_samples-1):
                continue

            m+=1

            # Here the false parring is created. The "there" flow is kept from the true parring of the for loop (for loop of train idex)
            # "Here" flow is added to the false parring from a random shuffeld idx which is not accidently the true parring.

            l2s[index,0,:,0] = minimum_padding(np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
            l2s[index,1,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0, min_length=flow_size)
            l2s[index,2,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
            l2s[index,3,:,0] = minimum_padding(np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0, min_length=flow_size)

            l2s[index,4,:,0] = minimum_padding(np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
            l2s[index,5,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
            l2s[index,6,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
            l2s[index,7,:,0] = minimum_padding(np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
            
            #l2s[index,0,:,0]=Y_train[i]#np.concatenate((Y_train[i],X_train[idx]))#(Y_train[i]*X_train[idx])/(np.linalg.norm(Y_train[i])*np.linalg.norm(X_train[idx]))
            #l2s[index,1,:,0]=X_train[idx]

            labels[index,0]=0
            index+=1
    
    #lsh.setup((X_test / np.linalg.norm(X_test,axis=1,keepdims=True)) .astype(np.float64))
    index_hard=0
    num_hard_test=0
    l2s_test = np.memmap(l2s_test_path, dtype=np.float32, mode='w+', shape=(len(test_index)*(negetive_samples+1),8,flow_size,1))
    labels_test = np.memmap(labels_test_path, dtype=np.float32, mode='w+', shape=(len(test_index)*(negetive_samples+1)))
    #l2s_test=np.zeros((len(test_index)*(negetive_samples+1),8,flow_size,1))
    #labels_test=np.zeros((len(test_index)*(negetive_samples+1)))
    #l2s_test_hard=np.zeros((num_hard_test*num_hard_test,2,flow_size,1))
    index=0
    random_test=[]+test_index

    for i in tqdm.tqdm(test_index, desc="Testing data"):
        #list(lsh.find_k_nearest_neighbors((Y_test[i]/ np.linalg.norm(Y_test[i])).astype(np.float64),(50)))

        if index % (negetive_samples+1) !=0:
            #print(index, nears)
            print(index)
            raise 
        m=0

        np.random.shuffle(random_test)
        for idx in random_test:
            if idx==i or m>(negetive_samples-1):
                continue

            m+=1

            l2s_test[index,0,:,0] = minimum_padding(np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
            l2s_test[index,1,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0, min_length=flow_size)
            l2s_test[index,2,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
            l2s_test[index,3,:,0] = minimum_padding(np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0, min_length=flow_size)

            l2s_test[index,4,:,0] = minimum_padding(np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
            l2s_test[index,5,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
            l2s_test[index,6,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
            l2s_test[index,7,:,0] = minimum_padding(np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0, min_length=flow_size)

            labels_test[index]=0
            index+=1

        # Everything same for testing as for training data. for details what this does see some lines above
        l2s_test[index,0,:,0] = minimum_padding(np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
        l2s_test[index,1,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0, min_length=flow_size)
        l2s_test[index,2,:,0] = minimum_padding(np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0, min_length=flow_size)
        l2s_test[index,3,:,0] = minimum_padding(np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0, min_length=flow_size)

        l2s_test[index,4,:,0] = minimum_padding(np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
        l2s_test[index,5,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
        l2s_test[index,6,:,0] = minimum_padding(np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0, min_length=flow_size)
        l2s_test[index,7,:,0] = minimum_padding(np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0, min_length=flow_size)
        
        #l2s_test[index,2,:,0]=dataset[i]['there'][0]['->'][:flow_size]
        #l2s_test[index,3,:,0]=dataset[i]['here'][0]['<-'][:flow_size]

        #l2s_test[index,0,:,1]=dataset[i]['here'][1]['->'][:flow_size]
        #l2s_test[index,1,:,1]=dataset[i]['there'][1]['<-'][:flow_size]
        #l2s_test[index,2,:,1]=dataset[i]['there'][1]['->'][:flow_size]
        #l2s_test[index,3,:,1]=dataset[i]['here'][1]['<-'][:flow_size]
        labels_test[index]=1 

        index+=1

    print("TRAINING set size (true and false flow pairs total): ", len(l2s))
    print("TRAINING set size of true flow pairs: ", len(train_index))
    print("TRAINING set size of false flow pairs: ", len(l2s)-len(train_index))
    print("TESTING set size (true and false flow pairs total): ", len(l2s_test))
    print("TESTING set size of true flow pairs: ", len(test_index))
    print("TESTING set size of false flow pairs: ", len(l2s_test)-len(test_index))

    return l2s, labels,l2s_test,labels_test

def truncate_dataset(dataset, training_limit, testing_limit):
    """
    Truncate the dataset to the specified limits for training and testing.

    Args:
        dataset (list): The loaded dataset.
        training_limit (int): The number of true flow pairs for training.
        testing_limit (int): The number of true flow pairs for testing.

    Returns:
        tuple: Two lists containing the truncated dataset for training and testing.
    """
    total_limit = training_limit + testing_limit
    if total_limit > len(dataset):
        raise ValueError("The sum of training and testing limits exceeds the dataset size.")
    
    return dataset[:training_limit], dataset[training_limit:total_limit]

def flatten_arrays(*arrays):
    """
    This function flattens multiple arrays into a list of 1D arrays.
    Useful for flattening the labels
    """
    flattened_arrays = [array.flatten() for array in arrays]
    return tuple(flattened_arrays)

def flatten_generated_flow_pairs(*arrays):
    """
    This operation flattens multiple generated flow pairs into a list of 2D arrays.
    Each input array should be in the format (N, F, M), where N is the number of flow pairs,
    F is the number of features, and M is the number of samples per feature.
    """
    flattened_arrays = [array.reshape(array.shape[0], -1) for array in arrays]
    return flattened_arrays

#old
# def aggregate_features(data):
#     """
#     Aggregate the input data by computing the mean, standard deviation, and variance
#     for each channel across the packet sizes/timing values.
    
#     Args:
#     - data: Input data with shape [18016, 8, 300, 1].
    
#     Returns:
#     - aggregated_data: Output data with shape [18016, 8, 3, 1], where each channel now has
#       mean, standard deviation, and variance as the features.
#     """
#     # Compute mean, std, and variance across the third axis (packet sizes/timing values)
#     mean = np.mean(data, axis=2, keepdims=True)  # Shape: [18016, 8, 1, 1]
#     std = np.std(data, axis=2, keepdims=True)    # Shape: [18016, 8, 1, 1]
#     variance = np.var(data, axis=2, keepdims=True)  # Shape: [18016, 8, 1, 1]
    
#     # Concatenate the computed statistics along the third axis to form a single array
#     # Correctly concatenating to get shape [18016, 8, 3, 1]
#     aggregated_data = np.concatenate([mean, std, variance], axis=2)  # Corrected step
    
#     return aggregated_data

def aggregate_features(data, mode):
    """
    Aggregate the input data by computing the mean, standard deviation, and variance
    for selected channels across the packet sizes/timing values, while padding the aggregated
    channels to match the feature size of the unchanged channels.
    
    Args:
    - data: Input data with shape [18016, 8, 300, 1].
    - mode: A string that can be 'both', 'ipd', or 'packet_sizes' to determine which channels
      to include in the aggregation.
    
    Returns:
    - aggregated_data: Output data with shape [18016, 8, 300, 1], with padded aggregated data
      where necessary.
    """
    original_feature_size = data.shape[2]  # Typically 300

    if mode == 'ipds':
        channels_to_aggregate = data[:, :4, :, :]
        channels_unchanged = data[:, 4:, :, :]
    elif mode == 'packet_sizes':
        channels_to_aggregate = data[:, 4:, :, :]
        channels_unchanged = data[:, :4, :, :]
    else:
        return data  # If mode is 'both', all data remains unchanged (original features per channel)

    # Compute mean, std, and variance for the selected channels
    mean = np.mean(channels_to_aggregate, axis=2, keepdims=True)
    std = np.std(channels_to_aggregate, axis=2, keepdims=True)
    variance = np.var(channels_to_aggregate, axis=2, keepdims=True)
    
    # Concatenate the computed statistics along the third axis to form aggregated channels
    aggregated_channels = np.concatenate([mean, std, variance], axis=2)  # Shape: [18016, 4, 3, 1]

    # Pad aggregated channels to match original feature size
    padding_size = original_feature_size - aggregated_channels.shape[2]
    padded_aggregated_channels = np.pad(aggregated_channels, ((0,0), (0,0), (0, padding_size), (0,0)), 'constant')

    # Combine the padded aggregated channels back with the unchanged channels
    if mode == 'ipds':
        aggregated_data = np.concatenate([padded_aggregated_channels, channels_unchanged], axis=1)
    elif mode == 'packet_sizes':
        aggregated_data = np.concatenate([channels_unchanged, padded_aggregated_channels], axis=1)

    return aggregated_data

def generate_aggregate_flow_pairs_to_memmap(dataset, train_index, test_index, flow_size, memmap_saving_path, negative_samples, agg_mode, drop_feature):
    """
    Wrapper function that generates flow pairs, performs feature aggregation on the generated data,
    and returns the aggregated data as memmap arrays.
    """
    # Ensure the saving path exists
    if not os.path.exists(memmap_saving_path):
        os.makedirs(memmap_saving_path)

    # Step 1: Generate the flow pairs using the provided function
    l2s, labels, l2s_test, labels_test = generate_flow_pairs_to_memmap(dataset, 
                                                                       train_index, 
                                                                       test_index, 
                                                                       flow_size, 
                                                                       memmap_saving_path, 
                                                                       negative_samples)
    
    print("Shape of l2s before dropping feature:", l2s.shape)
    print("Shape of l2s_test before dropping feature:", l2s_test.shape)

    # Determine the new shapes based on the feature to drop
    new_shape = None
    if drop_feature is not None:
        if drop_feature == "ipds":
            new_shape = l2s.shape[1] - 4
        elif drop_feature == "packet_sizes":
            new_shape = 4
        else:
            raise ValueError("Invalid feature to drop. Must be 'ipds' or 'packet_sizes'.")

        # Path for the new memmap files
        new_l2s_path = os.path.join(memmap_saving_path, 'adjusted_l2s.dat')
        new_l2s_test_path = os.path.join(memmap_saving_path, 'adjusted_l2s_test.dat')

        # Create new memmap arrays with the adjusted dimensions
        adjusted_l2s = np.memmap(new_l2s_path, dtype=np.float32, mode='w+', shape=(l2s.shape[0], new_shape, l2s.shape[2], l2s.shape[3]))
        adjusted_l2s_test = np.memmap(new_l2s_test_path, dtype=np.float32, mode='w+', shape=(l2s_test.shape[0], new_shape, l2s_test.shape[2], l2s_test.shape[3]))

        # Copy the data from the original memmap arrays to the new ones
        if drop_feature == "ipds":
            adjusted_l2s[:] = l2s[:, 4:, :, :]
            adjusted_l2s_test[:] = l2s_test[:, 4:, :, :]
        elif drop_feature == "packet_sizes":
            adjusted_l2s[:] = l2s[:, :4, :, :]
            adjusted_l2s_test[:] = l2s_test[:, :4, :, :]
        
        # Replace old memmap variables with the new ones
        l2s, l2s_test = adjusted_l2s, adjusted_l2s_test
    
    print("Shape of l2s after dropping feature:", l2s.shape)
    print("Shape of l2s_test after dropping feature:", l2s_test.shape)
    
    # Define paths for the aggregated features
    aggregated_paths = {
        'train': os.path.join(memmap_saving_path, 'training_flow_pairs'),
        'test': os.path.join(memmap_saving_path, 'test_flow_pairs')
    }
    
    print("\nAggregating features of training data...")
    # Step 2: Perform feature aggregation
    aggregated_train = aggregate_features(l2s,agg_mode)

    print("\nAggregating features of testing data...")
    aggregated_test = aggregate_features(l2s_test,agg_mode)
    
    print("\nSaving aggregated feature data to disk...")
    # Optionally, save the aggregated features to disk as memmap arrays and then load them
    aggregated_memmaps = {}
    for key, data in [('train', aggregated_train), ('test', aggregated_test)]:
        shape = data.shape
        dtype = np.float32  # Ensure this matches the data type of your aggregated features
        aggregated_path = aggregated_paths[key]
        # Save and immediately load the aggregated data to ensure it's memory-mapped
        new_memmap = np.memmap(aggregated_path, dtype=dtype, mode='w+', shape=shape)
        new_memmap[:] = data[:]
        del new_memmap  # Flush to disk
        # Load the memmap to return
        aggregated_memmaps[key] = np.memmap(aggregated_path, dtype=dtype, mode='r', shape=shape)

    print("Processing complete.")

    save_array_to_file(array=aggregated_memmaps['train'], file_name='training_flow_pairs.txt', save_path=memmap_saving_path)
    save_array_to_file(array=labels, file_name='training_labels.txt', save_path=memmap_saving_path)

    save_array_to_file(array=aggregated_memmaps['test'], file_name='test_flow_pairs.txt', save_path=memmap_saving_path)
    save_array_to_file(array=labels_test, file_name='test_labels.txt', save_path=memmap_saving_path)

    # Return memmap arrays for aggregated data along with labels
    return aggregated_memmaps['train'], labels, aggregated_memmaps['test'], labels_test


def minimum_padding(np_array, min_length):
    """
        Pad the input numpy array with zeros to reach the minimum length specified.
    """
    if len(np_array) >= min_length:
        return np_array
    else:
        #print("Before padding shape: ", np_array.shape)
        #print("Before padding content"  , np_array)
        padded_array = np.pad(np_array, (0, min_length - len(np_array)), 'constant')

        #print("After padding shape: ", padded_array.shape)  
        #print("After padding content"  , padded_array)
        return padded_array