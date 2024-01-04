import tensorflow as tf
import shared.data_processing as data_processing
import numpy as np
import tqdm
import datetime
import os

from shared.utils import create_path


# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#saver = tf.train.Saver()
def train_model(num_epochs, dataset, train_index, test_index, flow_size, negetive_samples, batch_size, train_flow_before, train_label, dropout_keep_prob, loss, optimizer, summary_op, init, saver, predict, graph, run_folder_path):
    print("Starting training...")
    log_path = os.path.join(run_folder_path, "logs/tf_log/noise_classifier/allcir_300_", str(datetime.datetime.now()))
    writer = tf.summary.FileWriter(log_path, graph=graph)
    
    with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
        session.run(init)

        #TODO Check before running that everything is the same as the origial
        temp_path = os.join(run_folder_path, "temp")

        for epoch in range(num_epochs):
            l2s, labels, l2s_test, labels_test = data_processing.generate_flow_pairs_to_memmap(dataset=dataset, train_index=train_index, test_index=test_index, flow_size=flow_size, memmap_saving_path=temp_path, negetive_samples=negetive_samples)

            # needs to be done since we have one positive and then 199 negative samples. needs to be shuffled   
            # dont shuffle the l2s_test and labels_test since we want to keep the order of the test_index since need to know which groups belong together (1 group = 1 true flow pair and N_neg false flow pairs)         
            rr= list(range(len(l2s)))
            print("l2s size: ", len(l2s))
            np.random.shuffle(rr)
            l2s = l2s[rr]
            labels = labels[rr]

            average_loss = 0
            new_epoch=True
            num_steps= (len(l2s)//batch_size)-1

            # TODO why is this called step and not mini batch
            for step in range(num_steps):
                start_ind = step*batch_size
                end_ind = ((step + 1) *batch_size)
                if end_ind < start_ind:
                    print('HOOY')
                    continue

                else:
                    batch_flow_before=l2s[start_ind:end_ind,:]
                    batch_label= labels[start_ind:end_ind]

                # Tensorflow speficic code. Keys are the placeholders and the values are the data
                # This seems not the be validation data but training data therefore session run dosent return the validation loss but it returns the training loss
                # train_flow_before = tf.placeholder(tf.float32, shape=[batch_size, 8,flow_size,1],name='flow_before_placeholder') used as placeholder
                feed_dict = {train_flow_before: batch_flow_before,
                                train_label:batch_label,
                            dropout_keep_prob:0.6}
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()

                # see feed_dict comment above. This seems to be the training loss not the validation loss
                _, loss_val,summary = session.run([optimizer, loss, summary_op], feed_dict=feed_dict)

                # average_loss += loss_val
                writer.add_summary(summary, (epoch*num_steps) +step)

                # print step, loss_val
                # if step % FLAGS.print_every_n_steps == 0:
                #     if step > 0:
                #         average_loss /= FLAGS.print_every_n_steps
                #     # The average loss is an estimate of the loss over the last 2000 batches.
                #     print("Average loss at step ", step, ": ", average_loss)
                #     average_loss = 0.

                # Note that this is expensive (~20% slowdown if computed every 500 steps)

                if ((epoch*num_steps) +step) % 100 == 0:
                    # TODO why is this validation loss and not training loss?
                    print("Average loss on validation set at step ",  (epoch*num_steps) +step, ": ", loss_val)
                if (((epoch*num_steps) +step)) % 100 == 0 and epoch >1:
                    tp=0
                    fp=0

                    num_steps_test= (len(l2s_test)//batch_size)-1
                    Y_est=np.zeros((batch_size*(num_steps_test+1)))
                    for step in range(num_steps_test):
                        start_ind = step*batch_size
                        end_ind = ((step + 1) *batch_size)
                        test_batch_flow_before=l2s_test[start_ind:end_ind]
                        feed_dict = {
                                train_flow_before:test_batch_flow_before,
                            dropout_keep_prob:1.0}

                        est=session.run(predict, feed_dict=feed_dict)
                        #est=np.array([xxx.sum() for xxx in test_batch_flow_before])
                        Y_est[start_ind:end_ind]=est.reshape((batch_size))
                    # the '//' rounds down to the nearest integer
                    num_samples_test=len(l2s_test)//(negetive_samples+1)

                    for idx in range(num_samples_test-1):
                        # see notion paper note in deepocorr note section for this formula
                        # Tldr is that we take the index of the highest corr_proboability score (p_i,j) returned from the model for seach "group"
                        # and one group is the one true flow pair and the N_neg false flow pairs
                        best=np.argmax(Y_est[idx*(negetive_samples+1):(idx+1)*(negetive_samples+1)])

                        # Checking if the index of the flow_pair the model said it has the highest corr_probability score is the true flow pair. If they have the same index then it is the true flow pair
                        if labels_test[best+(idx*(negetive_samples+1))]==1:
                            tp+=1
                        else:
                            fp+=1
                    print("True Positive: ", tp)
                    print("False Positive: ", fp)
                    # acc is the formula for precision. this seens not to be the accuracy but the precision. Furthermore this is also not the accuracy from the paper
                    acc= float(tp)/float(tp+fp)
                    if float(tp)/float(tp+fp)>0.8:      
                        print('saving...')
                        model_saving_path = os.path.join(run_folder_path, "saved_models")
                        create_path(model_saving_path)
                        # TODO i think step here should be (epoch*num_steps) +step
                        save_path = saver.save(session, os.path.join(model_saving_path, "tor_199_epoch%d_step%d_acc%.2f.ckpt"%(epoch,step,acc)))
                        print('saved')
            print('Epoch',epoch)
            #save_path = saver.save(session, "/mnt/nfs/scratch1/milad/model_diff_large_1e4_epoch%d.ckpt"%(epoch))

            #t.join()

def test_model(name, dataset, test_index, flow_size, batch_size, saver, predict, graph, train_flow_before, dropout_keep_prob):
    print("Starting testing...")
    with tf.Session(graph=graph) as session:
        #name=raw_input('model name')
        print("inputted name: ", name)
        saver.restore(session, "./saved_models/%s"%name)
        print("Model restored.")
        corrs=np.zeros((len(test_index),len(test_index)))
        batch=[]
        l2s_test_all=np.zeros((batch_size,8,flow_size,1))
        l_ids=[]
        index=0
        xi,xj=0,0
        for i in tqdm.tqdm(test_index):
            xj=0
            for j in test_index:
                
                l2s_test_all[index,0,:,0]=np.array(dataset[j]['here'][0]['<-'][:flow_size])*1000.0
                l2s_test_all[index,1,:,0]=np.array(dataset[i]['there'][0]['->'][:flow_size])*1000.0
                l2s_test_all[index,2,:,0]=np.array(dataset[i]['there'][0]['<-'][:flow_size])*1000.0
                l2s_test_all[index,3,:,0]=np.array(dataset[j]['here'][0]['->'][:flow_size])*1000.0

                l2s_test_all[index,4,:,0]=np.array(dataset[j]['here'][1]['<-'][:flow_size])/1000.0
                l2s_test_all[index,5,:,0]=np.array(dataset[i]['there'][1]['->'][:flow_size])/1000.0
                l2s_test_all[index,6,:,0]=np.array(dataset[i]['there'][1]['<-'][:flow_size])/1000.0
                l2s_test_all[index,7,:,0]=np.array(dataset[j]['here'][1]['->'][:flow_size])/1000.0
                l_ids.append((xi,xj))
                index+=1
                if index==batch_size:
                    index=0
                    cor_vals=session.run(predict,feed_dict={train_flow_before:l2s_test_all,
                            dropout_keep_prob:1.0})
                    for ids in range(len(l_ids)):
                        di,dj=l_ids[ids]
                        corrs[di,dj]=cor_vals[ids]
                    l_ids=[]
                xj+=1
            xi+=1
        np.save(open('correlation_values_test.np','w'),corrs)