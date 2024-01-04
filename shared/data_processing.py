import numpy as np
import tqdm
import os
import pickle

from shared.utils import create_path

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
        l2s[index,0,:,0]=np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0
        l2s[index,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
        l2s[index,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
        l2s[index,3,:,0]=np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0

        # "There/here"[1] are the packet sizes
        l2s[index,4,:,0]=np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0
        l2s[index,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
        l2s[index,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
        l2s[index,7,:,0]=np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0


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

            l2s[index,0,:,0]=np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0
            l2s[index,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s[index,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s[index,3,:,0]=np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0

            l2s[index,4,:,0]=np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0
            l2s[index,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s[index,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s[index,7,:,0]=np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0

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

            l2s_test[index,0,:,0]=np.array(dataset[idx]['here'][0]['<-'][:flow_size])*1000.0
            l2s_test[index,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
            l2s_test[index,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
            l2s_test[index,3,:,0]=np.array(dataset[idx]['here'][0]['->'][:flow_size])*1000.0

            l2s_test[index,4,:,0]=np.array(dataset[idx]['here'][1]['<-'][:flow_size])/1000.0
            l2s_test[index,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
            l2s_test[index,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
            l2s_test[index,7,:,0]=np.array(dataset[idx]['here'][1]['->'][:flow_size])/1000.0
            labels_test[index]=0
            index+=1

        # Everything same for testing as for training data. for details what this does see some lines above
        l2s_test[index,0,:,0]=np.array(dataset[i]['here'][0]['<-'][:flow_size])*1000.0
        l2s_test[index,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
        l2s_test[index,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
        l2s_test[index,3,:,0]=np.array(dataset[i]['here'][0]['->'][:flow_size])*1000.0

        l2s_test[index,4,:,0]=np.array(dataset[i]['here'][1]['<-'][:flow_size])/1000.0
        l2s_test[index,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
        l2s_test[index,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
        l2s_test[index,7,:,0]=np.array(dataset[i]['here'][1]['->'][:flow_size])/1000.0
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