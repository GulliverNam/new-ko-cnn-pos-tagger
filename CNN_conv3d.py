# Editor : Giwon Nam
# last edit date: 2018-12-01 12:57

"""
전체 데이터 83만 문장을 학습시킬경우(training: 70만 / test: 13만) overfitting문제 발생
training 시 accuracy가 약 94~95%가 나오지만 실제 테스트 진행시 거의 맞추지 못함
=> softmax 결과 41개의 클래스가 거의 비슷한 값이 나옴(==0.2xxx)

따라서 training 데이터를 축소시켜 학습진행(training: 5만 / test: 1만)
accuracy는 거의 비슷한 수준
"""

import tensorflow as tf
import numpy as np
import pickle
import time
from preprocessing import Preprocessing
tf.set_random_seed(10)

indexed_data = pickle.load(open('Data/indexed_data.pkl', 'rb'))
dictionary_char = pickle.load(open('Data/char_dict.pkl', 'rb'))
dictionary_label = pickle.load(open('Data/label_dict.pkl', 'rb'))

word_max_length = 0
for sen in indexed_data:
    for word in sen:
        if word_max_length < len(word[0]): 
            word_max_length = len(word[0])
            
window_size = 7
char_dim = len(dictionary_char)
label_dim = len(dictionary_label)

X = tf.placeholder(tf.float32, [None, window_size*2 + 1, word_max_length, char_dim, 1])
# <tf.Tensor 'Placeholder:0' shape=(?, 15, 48, 189, 1) dtype=float32
Y = tf.placeholder(tf.float32, [None, label_dim])
dropout_rate = tf.placeholder(tf.float32)

for d in ['/device:GPU:0', '/device:GPU:1']:
    with tf.device(d):
        ###### 1. Character CNN ######
        filter_size = [3,5,7,9]
        num_filters = 25
        conved_layers = []
        for f in filter_size:
            word_Y = tf.layers.conv3d(inputs=X, filters=num_filters, kernel_size=[1, f, char_dim], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            word_Y = tf.layers.max_pooling3d(word_Y, pool_size=[1, word_max_length - f + 1, 1], strides=1)
            word_Y = tf.layers.dropout(word_Y,rate = dropout_rate)
            conved_layers.append(word_Y)

        # [<tf.Tensor 'max_pooling3d/MaxPool3D:0' shape=(?, 15, 1, 1, 25) dtype=float32>,
        #  <tf.Tensor 'max_pooling3d_1/MaxPool3D:0' shape=(?, 15, 1, 1, 25) dtype=float32>,
        #  <tf.Tensor 'max_pooling3d_2/MaxPool3D:0' shape=(?, 15, 1, 1, 25) dtype=float32>,
        #  <tf.Tensor 'max_pooling3d_3/MaxPool3D:0' shape=(?, 15, 1, 1, 25) dtype=float32>]

        concat_layers = tf.concat(conved_layers, 4)
        concat_layers = tf.reshape(concat_layers, [-1, window_size*2 + 1, num_filters*4])
        # <tf.Tensor 'Reshape:0' shape=(?, 15, 100) dtype=float32>

        plus_features = tf.placeholder(tf.float32, [None, 15, 5])
        # position embedding[4bits] &  binary embedding[1bit]
        out_word = tf.expand_dims(tf.concat([concat_layers, plus_features], 2), -1)
        #<tf.Tensor 'ExpandDims:0' shape=(?, 15, 105, 1) dtype=float32>

        ###### 2. Words CNN ######
        sen_filter_size = [2,3,4,5]
        sen_num_filters1 = 64
        sen_num_filters2 = 128
        sen_conved_layers = []
        for f in sen_filter_size:
            sen_Y = tf.layers.conv2d(inputs=out_word, filters=sen_num_filters1, kernel_size=[f, 105], activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            sen_Y = tf.layers.dropout(sen_Y,rate = dropout_rate)

            sen_Y = tf.layers.conv2d(inputs=sen_Y, filters=sen_num_filters2, kernel_size=[f, 1], activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            sen_Y = tf.layers.max_pooling2d(sen_Y, pool_size=[15 - 2*(f - 1), 1], strides=2)
            sen_Y = tf.layers.dropout(sen_Y,rate = dropout_rate)
            sen_conved_layers.append(sen_Y)
        cnn_final = tf.reshape(tf.concat(sen_conved_layers,3), [-1, sen_num_filters2*4])
        #<tf.Tensor 'Reshape_1:0' shape=(? 512) dtype=float32>

        ###### 3. Fully connected layer ######
        fully_W = tf.Variable(tf.truncated_normal([512, label_dim], stddev = 0.01))
        fully_b = tf.Variable(tf.zeros([label_dim]))

logit = tf.add(tf.matmul(cnn_final, fully_W), fully_b)
Y_pred = tf.nn.softmax(logit)
#<tf.Tensor 'Softmax:0' shape=(?, 41) dtype=float32>

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logit))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


def cal_total_size(indexed_data):
    total_size = 0
    for sentence in indexed_data:
        total_size += len(sentence)
    return total_size


def make_input(indexed_data, window_size=7, max_length=48, char_dim=189, label_dim=41):
    for sentence in indexed_data:
        for w_index in range(len(sentence)):
            X = np.zeros([window_size*2 +1, max_length, char_dim], dtype=np.float32)
            Y = np.zeros([1, label_dim], dtype=np.float32)
            word_index = 5 - w_index
            for word in sentence:
                word_index = word_index + 1
                if word_index >= 15 or word_index<0 : continue
                if word_index == 6: Y[0][word[1]] = 1
                
                if len(word[0]) % 2 == 0:
                    char_index = max_length//2 - len(word[0]) // 2
                else:
                    char_index = max_length//2 - (len(word[0]) // 2 + 1)
                
                for i, char in enumerate(word[0]):
                    X[word_index][int(i+char_index)][char] = 1
            X = np.reshape(X, [1, window_size*2 +1, max_length, char_dim, 1])
            
            yield X, Y


def make_batch_input(indexed_data, batch_size, mode): # mode: train or test
    if mode == 'train': 
        indexed_data = indexed_data[:50000]
    else: # mode == 'test'
        indexed_data = indexed_data[50000:60000]
    total_size = cal_total_size(indexed_data)
    sentence_input = make_input(indexed_data)
    for i in range(total_size//batch_size):
        X_input = []
        Y_input = []
        for j in range(batch_size):
            X, Y =next(sentence_input)
            X_input.append(X)
            Y_input.append(Y)
        X_input = np.concatenate(X_input, 0)
        Y_input = np.concatenate(Y_input, 0)
        yield X_input, Y_input


def make_features(batch_size):
    features = [[[1,0,0,1,0],
                 [1,0,1,0,0],
                 [1,0,1,1,0],
                 [1,1,0,0,0],
                 [1,1,0,1,0],
                 [1,1,1,0,0],
                 [1,1,1,1,0],
                 [0,0,0,0,1],
                 [0,0,0,1,0],
                 [0,0,1,0,0],
                 [0,0,1,1,0],
                 [0,1,0,0,0],
                 [0,1,0,1,0],
                 [0,1,1,0,0],
                 [0,1,1,1,0]]]*batch_size
    return np.array(features, dtype=np.float32)

saver = tf.train.Saver()
epoch = 1
train_batch_size = 100
test_batch_size = 10000
train_size = cal_total_size(indexed_data[:50000])
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95) # GPU 메모리 95% 사용

print('Start Traing')
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    features_train = make_features(train_batch_size)
    features_test = make_features(test_batch_size)
    
    check_step = 100
    test_step = 1000
    for i in range(epoch):
        start_time = time.time()
        print('epoch',[i+1])
        train_batch = make_batch_input(indexed_data, train_batch_size, 'train')
        a_mean = []
        l_mean = []
        for j in range(train_size//train_batch_size):
            X_train, Y_train = next(train_batch)
            _, l, a = sess.run([optimizer, loss, accuracy],feed_dict={X:X_train, Y:Y_train, dropout_rate:0.5, plus_features:features_train})
            a_mean.append(a)
            l_mean.append(l)
            if j % check_step == 0 and j != 0:
                print([j], 'accuracy: {:0.4f} loss: {:0.4f}'.format(sum(a_mean)/check_step,sum(l_mean)/check_step))
                a_mean = []
                l_mean = []
            if j % test_step == 0 and j != 0:
                test_batch = make_batch_input(indexed_data, test_batch_size, 'test')
                X_test, Y_test = next(test_batch) 
                lo, ac = sess.run([loss, accuracy],feed_dict={X:X_test, Y:Y_test, dropout_rate:0, plus_features:features_test})
                print('****test**** accuracy: {:0.4f} loss: {:0.4f}'.format(ac, lo))
        
        duration = time.time() - start_time
        minute = int(duration / 60)
        second = int(duration) % 60
        print("%dminutes %dseconds" % (minute,second))
        save_path = saver.save(sess, 'Model/CNN_POS_epoch{}.model'.format(i+1))
        print('save_path',save_path)
    
    saver.close()
    
