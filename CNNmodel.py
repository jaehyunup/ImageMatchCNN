import numpy as np
np.random.seed(3)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # 로그출력 레벨설정
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

def shuffling(features,labels):
    c = list(zip(features, labels))
    shuffle(c)
    x_data, y_data = zip(*c)
    x_data = list(x_data)
    y_data = list(y_data)
    return np.array(x_data),np.array(y_data)


rotateGenerator = ImageDataGenerator(rotation_range=40, fill_mode='nearest')
shiftGenerator = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3, fill_mode='nearest')
shearGenerator = ImageDataGenerator(shear_range=30, fill_mode='nearest')
zoomGenerator = ImageDataGenerator( zoom_range=[-0.1,0.1], fill_mode='nearest')

Dataset = []
filename_in_dir = []
X_data = []
Y_data = []
for root, dirs, files in os.walk('img_data\\img_data_r1'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        filename_in_dir.append(full_fname) #디렉토리 내의 파일 name

# 144개 기존이미지 불러오기 완료
#genenum=int(input("데이터를 몇배로 확장 하시겠습니까?(숫자 입력)"))-1
#genetype=int(input("확장방법 ? 1: rotate , 2: shear , 3: shift , 4: zoom "))
genenum=10-1
genetype=3

for file_image in filename_in_dir:
    img = load_img(file_image,color_mode='grayscale')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    Dataset.append(np.array(x.ravel() / 255.0))
    i1,i2,i3,i4=0,0,0,0
    if genetype== 1 :
        for batch in rotateGenerator.flow(x):
            i1 +=1
            Dataset.append(np.array(batch.ravel()/255.0))
            if i1> genenum-1:
                i1=0
                break
    if genetype == 2:
        for batch in shearGenerator.flow(x):
            i2 += 1
            Dataset.append(np.array(batch.ravel()/255.0))
            if i2 > genenum-1:
                i2 = 0
                break
    if genetype == 3:
        for batch in shiftGenerator.flow(x):
            i3 += 1
            Dataset.append(np.array(batch.ravel()/255.0))
            if i3 > genenum-1:
                i3 = 0
                break
    if genetype == 4:
        for batch in zoomGenerator.flow(x):
            i4 += 1
            Dataset.append(np.array(batch.ravel()/255.0))
            if i4 > genenum-1:
                i4 = 0
                break

# 디렉토리 경로 정의
dirpath=[]
filename_path_list=[] # 파일 상대경로 리스트 (0~143) 프레임당 10개씩
filename_list=[] # 파일 이름 리스트 (0~143) 프레임당 10개씩

for et in range (0,1440):
    Dataset[et]=Dataset[et].reshape(80,80,1)
    cv2.imshow("hey",Dataset[et])
    X_data.append(Dataset[et])
    print(len(X_data))


# read label in Y_data
label_File = open("img_data\\label.txt","r")
Y_data = label_File.read().splitlines()
Y_label = []
#0~144 까지
for i in range(0,len(Y_data)):
    if Y_data[i] == '0':
        for j in range(genenum+1):
            Y_label.append([1.0,0.0])
    elif Y_data[i] == '1':
        for j in range(genenum+1):
            Y_label.append([0.0, 1.0])
#finishd DataSet Setting

print("Y라벨 개수",len(Y_label))

Y_label = np.array(Y_label)
train_features = X_data[0:int(0.8*len(X_data))] # 특징 개수( 이미지 개수) * 0.8
train_labels = Y_label[0:int(0.8*len(Y_label))] # 라벨 개수( 이미지 라벨 개수) * 0.8
test_features = X_data[int(0.8*len(X_data)):] # 테스트 특징 개수
test_labels = Y_label[int(0.8*len(Y_label)):] #테스트 라벨 개수

#saver = tf.train.import_meta_graph('./model/testModel-80.meta')
#saver.restore(sess, "model\\testModel-80")
#graph = tf.get_default_graph()

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

saver = tf.train.Saver()
#session
sess=tf.Session()
init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train',sess.graph)
sess.run(init)
for epoch in range(200):
    #sp_train_features, sp_train_labels= shuffling(train_features,train_labels)
    summary, _,cost_val = sess.run([merged,optimizer, cost], feed_dict={X: train_features, Y: train_labels, keep_prob: 0.7})
    # 트레이닝 과정의 cost_val 변화
    print("%d 번 학습의 Cost : %.6f"%(epoch,cost_val))
    train_writer.add_summary(summary=summary,global_step=epoch)
train_writer.close()
print(' 배경 인식 테스트 정확도: %.2f' % sess.run(accuracy*100, feed_dict = {X: test_features, Y: test_labels, keep_prob:0.7}))
print("==Training finish===")

# 학습 된모델 저장
saver.save(sess, './model\\CNNmodel', global_step=200)
print("==Model Saved OK.===")
'''
# 데이터셋 체크
print("==Check Original DataSet..===")
print(sp_train_labels[0:1000:100])
test_img_data = sp_train_features[0:1000:100]
fig = plt.figure()
for i in range(10):
    subplot = fig.add_subplot(2,5,i+1)
    subplot.imshow(test_img_data[i].reshape((80,80)), cmap=plt.cm.gray_r)
plt.show()

    # 데이터셋 체크
    print("==Check Test DataSet..===")
    print(test_labels[0:1000:100])
    test_img_data = test_features[0:100:10]
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(2,5,i+1)
        subplot.imshow(test_img_data[i].reshape((80,80)), cmap=plt.cm.gray_r)
    plt.show()
    '''