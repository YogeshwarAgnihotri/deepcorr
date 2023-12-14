import tensorflow as tf 
import datetime
import os

def model(flow_before,dropout_keep_prob,flow_size):
    last_layer=flow_before
    flat_layers_after=[flow_size*2,1000,50,1]
    for l in range(len(flat_layers_after)-1):
        flat_weight = tf.get_variable("flat_after_weight%d"%l, [flat_layers_after[l],flat_layers_after[l+1]],
        initializer=tf.random_normal_initializer(stddev=0.01,mean=0.0))

        flat_bias = tf.get_variable("flat_after_bias%d"%l, [flat_layers_after[l+1]],
        initializer=tf.zeros_initializer())

        _x=tf.add(
                tf.matmul(last_layer, flat_weight),
                flat_bias)
        if l<len(flat_layers_after)-2:
            _x=tf.nn.dropout(tf.nn.relu(_x,name='relu_noise_flat_%d'%l),keep_prob=dropout_keep_prob)
        last_layer=_x
    return last_layer
        
def model_cnn(flow_before,dropout_keep_prob,batch_size):
    last_layer=flow_before
    
    CNN_LAYERS=[[2,20,1,2000,5],[4,10,2000,800,3]]
    
    for cnn_size in range(len(CNN_LAYERS)):
        cnn_weights = tf.get_variable("cnn_weight%d"%cnn_size, CNN_LAYERS[cnn_size][:-1],
            initializer=tf.random_normal_initializer(stddev=0.01))
        cnn_bias = tf.get_variable("cnn_bias%d"%cnn_size, [CNN_LAYERS[cnn_size][-2]],
            initializer=tf.zeros_initializer())

        _x = tf.nn.conv2d(last_layer, cnn_weights, strides=[1, 2,2, 1], padding='VALID')
        _x = tf.nn.bias_add(_x, cnn_bias)
        conv = tf.nn.relu(_x,name='relu_cnn_%d'%cnn_size)
        pool = tf.nn.max_pool(conv, ksize=[1, 1, CNN_LAYERS[cnn_size][-1], 1], strides=[1, 1, 1, 1],padding='VALID')
        last_layer=pool
    last_layer=tf.reshape(last_layer, [batch_size,-1])
    
    flat_layers_after=[49600,3000,800,100,1]
    for l in range(len(flat_layers_after)-1):
        flat_weight = tf.get_variable("flat_after_weight%d"%l, [flat_layers_after[l],flat_layers_after[l+1]],
        initializer=tf.random_normal_initializer(stddev=0.01,mean=0.0))

        flat_bias = tf.get_variable("flat_after_bias%d"%l, [flat_layers_after[l+1]],
        initializer=tf.zeros_initializer())

        _x=tf.add(
                tf.matmul(last_layer, flat_weight),
                flat_bias)
        if l<len(flat_layers_after)-2:
            _x=tf.nn.dropout(tf.nn.relu(_x,name='relu_noise_flat_%d'%l),keep_prob=dropout_keep_prob)
        last_layer=_x
    return last_layer


def build_graph_training(batch_size, flow_size, learn_rate):
        graph = tf.Graph()
        with graph.as_default():
            train_flow_before = tf.placeholder(tf.float32, shape=[batch_size, 8,flow_size,1],name='flow_before_placeholder')
            train_label = tf.placeholder(tf.float32,name='label_placeholder',shape=[batch_size,1])
            dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_placeholder')
            # train_correlated_var = tf.Variable(train_correlated, trainable=False)
            # Look up embeddings for inputs.


            y2 = model_cnn(train_flow_before, dropout_keep_prob, batch_size)
            predict=tf.nn.sigmoid(y2)
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2,labels=train_label),name='loss_sigmoid')


            # tp = tf.contrib.metrics.streaming_true_positives(predictions=tf.nn.sigmoid(logits), labels=train_correlated)
            # fp = tf.contrib.metrics.streaming_false_positives(predictions=tf.nn.sigmoid(logits), labels=train_correlated)

            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)


            #    gradients = tf.norm(tf.gradients(logits, weights['w_out']))

            #    w_mean, w_var = tf.nn.moments(weights['w_out'], [0])
            s_loss=tf.summary.scalar('loss', loss)
            #    tf.summary.scalar('weight_norm', tf.norm(weights['w_out']))
            #    tf.summary.scalar('weight_mean', tf.reduce_mean(w_mean))
            #    tf.summary.scalar('weight_var', tf.reduce_mean(w_var))

            #    tf.summary.scalar('bias', tf.reduce_mean(biases['b_out']))
            #    tf.summary.scalar('logits', tf.reduce_mean(logits))
            #    tf.summary.scalar('gradients', gradients)
            summary_op = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            return train_flow_before, train_label, dropout_keep_prob, loss, optimizer, summary_op, init, saver, predict, graph


def build_graph_testing(batch_size, flow_size):
    graph = tf.Graph()
    with graph.as_default():
        train_flow_before = tf.placeholder(tf.float32, shape=[batch_size, 8,flow_size,1],name='flow_before_placeholder')
        train_label = tf.placeholder(tf.float32,name='label_placeholder',shape=[batch_size,1])
        dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_placeholder')
        # train_correlated_var = tf.Variable(train_correlated, trainable=False)
        # Look up embeddings for inputs.



        y2 = model_cnn(train_flow_before, dropout_keep_prob, batch_size)
        predict=tf.nn.sigmoid(y2)
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        saver = tf.train.Saver()

        writer = tf.summary.FileWriter('./logs/tf_log/noise_classifier/allcir_300_'+str(datetime.datetime.now()), graph=graph)

        return train_flow_before, train_label, dropout_keep_prob, saver, predict, graph