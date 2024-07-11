import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential #계산이 왼쪽에서 오른쪽으로 흐르는 경우 Sequential
from tensorflow.keras.layers import Dense # 완전연결층은 Dense
from tensorflow.keras.optimizers import SGD #스토캐스틱 경사하강법

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data() #데이터셋 저장
x_train=x_train.reshape(60000,784) #1차원으로 변환
x_test=x_test.reshape(10000,784) #1차원으로 변환
x_train=x_train.astype(np.float32)/255.0 #실수 연산이 가능하도록 float형으로 변환,[0,1] 범위로 변환
x_test=x_test.astype(np.float32)/255.0 #실수 연산이 가능하도록 float형으로 변환,[0,1] 범위로 변환
y_train=tf.keras.utils.to_categorical(y_train,10) #원핫 코드로 변환 
y_test=tf.keras.utils.to_categorical(y_test,10) #원핫 코드로 변환 

mlp=Sequential() #mlp객체 생성
mlp.add(Dense(units=512,activation='tanh',input_shape=(784,))) #입력층에 784개 노드, 은닉층에 512개 노드 완전연결층, 활성함수는 하이퍼볼릭 탄젠트
mlp.add(Dense(units=10,activation='softmax')) #출력노드 10개, 활성함수 softmax인 완전 연결층

#손실함수 MSE, 학습률 0.01인 SGD, 정확률을 기준으로 성능 측정
mlp.compile(loss='MSE',optimizer=SGD(learning_rate=0.01),metrics=['accuracy'])
#배치 크기 128, epoch=50, verbose=2는 학습 도중 epoch마다 성능 출력(0이면 출력안함,1이면 진행 막대만 표시)  
mlp.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

res=mlp.evaluate(x_test,y_test,verbose=0) #성능 측정하여 res객체에 저장
print('정확률=',res[1]*100) #res[1]은 정확률임