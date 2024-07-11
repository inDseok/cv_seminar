import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data() #데이터셋 읽어오기
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) #데이터셋 구조 출력
plt.figure(figsize=(12,3))#그림 사이즈 지정
plt.suptitle('MNIST',fontsize=30) # 그림 제목 설정
for i in range(10):
    plt.subplot(1,10,i+1) #한줄에 10개의 영상을 배치하는데 i+1번째를 채우라고 지시
    plt.imshow(x_train[i],cmap='gray') # i번째 샘플 흑백으로 출력
    plt.xticks([]);plt.yticks([]) #x,y축에 눈금 달지 말라고 지시
    plt.title(str(y_train[i]),fontsize=30) #샘플의 부류 정보를 제목으로 달아줌
    
(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data() #데이터셋 읽어오기
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) #데이터셋 구조 출력
class_names=['airplane','car','bird','cat','deer','dog','frog','hosre','ship','truck']
plt.figure(figsize=(12,3)) #그림 사이즈 지정
plt.suptitle('CIFAR-10',fontsize=30) # 그림 제목 설정
for i in range(10):
    plt.subplot(1,10,i+1) #한줄에 10개의 영상을 배치하는데 i+1번째를 채우라고 지시
    plt.imshow(x_train[i])  # i번째 샘플 출력
    plt.xticks([]);plt.yticks([]) #x,y축에 눈금 달지 말라고 지시
    plt.title(class_names[y_train[i,0]],fontsize=30) #샘플의 부류 정보를 제목으로 달아줌