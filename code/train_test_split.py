import pickle
import numpy as np

def split_train_test(dataset):
    len_tr=len(dataset)
    print("Dataset length: ", len_tr)
    #train_ratio=float(len_tr-6000)/float(len_tr)
    train_ratio=0.8
    rr= range(len(dataset))
    np.random.shuffle(rr)

    train_index=rr[:int(len_tr*train_ratio)]
    test_index= rr[int(len_tr*train_ratio):] #range(len(dataset_test)) # #
    pickle.dump(test_index,open('test_index300.pickle','w'))

    return train_index, test_index