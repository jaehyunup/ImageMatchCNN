import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from cv2 import cv2
except ImportError:
    pass
import tensorflow as tf
import numpy as np


xypatharray=[]
tempxypatharray=[]

def img_trim(src_image):
    del tempxypatharray[:]
    src_real=src_image.copy()
    temp_list=[]
    x ,y=0,0
    for i in range(0,16):
        for j in range(0,9):
            src_temp=np.array(src_real[j*80:(j+1)*80,i*80:(i+1)*80]).ravel()
            tempxypatharray.append((j*80, (j+1)*80, i*80, (i+1)*80, False))
            temp_list.append(src_temp)
    xypatharray.append(tempxypatharray)
    return np.array(temp_list)

# 학습시 사용되었던 모델과 동일하게 정의
# 학습시 사용되었던 모델과 동일하게 정의
X = tf.placeholder(tf.float32,[None,80,80,1])
Y = tf.placeholder(tf.float32,[None,2])
keep_prob = tf.placeholder(tf.float32) #드롭아웃 확률

with tf.name_scope('Hidden1') as scope:
    # layer 1
    w1 = tf.Variable(tf.random_normal([3,3,1,32],stddev = 0.01),name="W1") #in
    L1 = tf.nn.conv2d(X,w1,strides=[1,1,1,1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME') # 40 x 40 pooling
# layer 2
with tf.name_scope('Hidden2') as scope:
    w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev = 0.01),name="W2") #in
    L2 = tf.nn.conv2d(L1,w2,strides=[1,1,1,1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME') # 20 x 20 pooling

with tf.name_scope('Hidden3') as scope:
    # layer 3
    w3 = tf.Variable(tf.random_normal([3,3,64,128],stddev = 0.01),name="W3") #in
    L3 = tf.nn.conv2d(L2,w3,strides=[1,1,1,1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME') # 10 x 10 pooling

with tf.name_scope('fullyconlayer') as scope:
    # out layer
    w4 = tf.Variable(tf.random_normal([10*10*128, 256],stddev = 0.01),name="W4") #in
    L4 = tf.reshape(L3, [-1,10*10*128])
    L4 = tf.matmul(L4,w4)
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.dropout(L4, keep_prob)

with tf.name_scope('OutLayer') as scope:
    # out layer
    w5 = tf.Variable(tf.random_normal([256,2], stddev=0.01), name="W5")  # in
    model = tf.matmul(L4,w5)

with tf.name_scope('LossFunction') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.summary.scalar('cost', cost)
    # cost 율 체크 , model에 label을 확인해봄으로서,(소프트맥스)

with tf.name_scope("accuracyCheck") as scope:
    prediction = tf.argmax(model, axis=1)
    # 모델 원핫 인코딩
    target = tf.argmax(Y, axis=1)
    # 정확도 계산
    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
saver=tf.train.Saver()
#get model data
sess=tf.Session()
saver.restore(sess, tf.train.latest_checkpoint("model"))
cap = cv2.VideoCapture('img_data\\car3.mp4')
ret, frame = cap.read()  # binary Video 객체
# start
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('nomal.avi', fourcc, 25.0, (1280, 720))

while (ret ==1) :

    grayframe = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    #cv2.imshow('trim', img_trim(grayframe ,0, 80, 0, 80))
    grayframe_trim = img_trim(grayframe)
    #window=cv2.imshow("image",grayframe_trim[5]) # 144개의 trim image 가 있음.3 차원 ndarray.
    # grayframe_trim[0~143] = 지금 프레임에서 trim 되어 갈라진 각 80x80 한 픽셀을 의미함
    test_feature=grayframe_trim / 255.0  # 0~255 => 0~1 치환
    test_feature_real = []
    for et in range(0, 144):
        temp_feture_patch=np.array(test_feature[et]).reshape(80,80,1) #80,80,1 의 패치.
        test_feature_real.append(np.array(test_feature[et]).reshape(80,80,1)) #80,80,1 의 패치.)
        print(len(test_feature_real))
    # 모델의 예측 비 계산
    prelist=sess.run(prediction, feed_dict = {X: test_feature_real,keep_prob:0.7})

    prelist=np.array(prelist)
    width,height=0,0
    count=0

    # 배경영역 색칠(144개의 픽셀다 돌아야함)
    for xi in range (0,16):
        # 픽셀이 배경일때 1 , 전경일때 0 .
        for xj in range(0,9):
            if prelist[count]==1:
                #print(prelist[count])
                frame[xj*80:(xj+1)*80,xi*80:(xi+1)*80]= (0,0,255)
            count = count + 1
    #out.write(frame)
    cv2.imshow('TEST',frame)
    ret, frame = cap.read()  # binary Video 객체
    k = cv2.waitKey(100) & 0xff
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('p'):
        while (1000):
            wk = cv2.waitKey(0)
            if wk == ord('p'):
                break
out.release()
cap.release()
cv2.destroyAllWindows()