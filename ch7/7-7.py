import numpy as np
import tensorflow as tf
import cv2 as cv 
import matplotlib.pyplot as plt
import winsound

model=tf.keras.models.load_model('dmlp_trained.h5') #저장한 모델 불러옴
#reset 함수
def reset():
    global img
       
    img=np.ones((200,520,3),dtype=np.uint8)*255 #200x520의 3채널 컬러영상 저장할 배열을 255를 곱해 모두 흰색으로 초기화
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255)) #5개의 빨간색 박스 
    cv.putText(img,'e:erase s:show r:recognition q:quit',(10,40),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1) #명령어를 나타내는 글씨
#grab_numerals 함수
def grab_numerals():
    numerals=[] #빈 리스트 생성
    for i in range(5): #이미지에서 숫자를 떼네 28x28크기로 변환하여 리스트에 추가 
        roi=img[51:149,11+i*100:9+(i+1)*100,0]
        roi=255-cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC)
        numerals.append(roi)  
    numerals=np.array(numerals) #리스트를 numpy배열로 변환
    return numerals
#show 함수
def show(): #5개의 숫자 표시
    numerals=grab_numerals()
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()
#recognition 함수    
def recognition():
    numerals=grab_numerals() #grab_numerals()로 부터 5개의 숫자 받아서 numerals객체에 저장
    numerals=numerals.reshape(5,784) #신경망 입력을 위해 1차원으로 변경
    numerals=numerals.astype(np.float32)/255.0 #실수 배열로 바꾸고 [0,1] 범위로 변환
    res=model.predict(numerals) #예측 수행하고 결과 res객체에 저장
    class_id=np.argmax(res,axis=1) #최대값을 가지는 인덱스를 찾아 class_id객체에 저장
    for i in range(5): # 빨간색 박스 밑에 인식 결과 표시
        cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
    winsound.Beep(1000,500) #삑 소리로 주의를 끔   
        
BrushSiz=4
LColor=(0,0,0)
#writing 함수
def writing(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN: #왼쪽 클릭시
        cv.circle(img,(x,y),BrushSiz,LColor,-1) #BrushSiz의 크기의 원을 검은색으로 그림 
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON: #누르고 움직이면 
        cv.circle(img,(x,y),BrushSiz,LColor,-1) #BrushSiz의 크기의 원을 검은색으로 그림 

reset() #reset함수 호출
cv.namedWindow('Writing') #윈도우 생성
cv.setMouseCallback('Writing',writing) #윈도우의 콜백 함수로 writing 함수 등록

while(True):
    cv.imshow('Writing',img) #이미지 표시
    key=cv.waitKey(1) #키보드 입력이 있으면 key에 저장
    if key==ord('e'): 
        reset()
    elif key==ord('s'):
        show()        
    elif key==ord('r'):
        recognition()
    elif key==ord('q'):
        break
    
cv.destroyAllWindows() #모든 윈도우 닫음