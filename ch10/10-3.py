import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f=open('coco_names.txt', 'r') # coco데이터셋
    class_names=[line.strip() for line in f.readlines()] # coco데이터셋의 부류이름을 읽어 저장

    model=cv.dnn.readNet('yolov3.weights','yolov3.cfg') # yolo모델 저장
    layer_names=model.getLayerNames() 
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()] # yolo_82,94,106층을 알아내어 저장
     
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1] # 원본영상 높이와 너비 정보 저장
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) # yolo에 입력할 수 있는 형태로 변환해 저장
    # 화소값을[0,1]로 변환하고 448x448크기로 변환,bgr->rgb로 바꿈
    yolo_model.setInput(test_img) # 신경망에 입력
    output3=yolo_model.forward(out_layers) # 신경망의 전방 계산 수행, out_layers가 출력한 텐서 저장
    
    box,conf,id=[],[],[] # 박스, 신뢰도, 부류 번호
    for output in output3: # 3개의 텐서를 각각 반복 처리
        for vec85 in output:  # 85차원 벡터 반복 처리
            scores=vec85[5:] # 80개의 요소 값
            class_id=np.argmax(scores) # 최고 확률에 해당하는 부류 번호
            confidence=scores[class_id] # 확률
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height) #[0,1] 범위로 표현된 박스 원래 영상 좌표계로 변환
                w,h=int(vec85[2]*width),int(vec85[3]*height) #너비와 높이
                x,y=int(centerx-w/2),int(centery-h/2) # 왼쪽 위의 위치
                box.append([x,y,x+w,y+h]) #박스 추가
                conf.append(float(confidence)) #신뢰도 추가
                id.append(class_id) #부류 정보 추가
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4) #박스를 대상으로 비최대 억제를 적용해 중복성 제거
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind] #비최대 억제에서 살아남은 박스의 위치, 신뢰도, 부류 저장
    return objects

model,out_layers,class_names=construct_yolo_v3()	# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(100,3))		# 100개 색으로 트랙 구분

from sort import Sort

sort=Sort() # sort객체 생성

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 웹 캠과 연결
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret,frame=cap.read() 
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
        
    res=yolo_detect(frame,model,out_layers) # yolo_detect함수로 프레임에서 물체를 검출해 저장
    persons=[res[i] for i in range(len(res)) if res[i][5]==0] # 부류 0은 사람

    if len(persons)==0: # 검출된 사람이 없을 때
        tracks=sort.update() #이전 이력 정보 유지(일관성)
    else: #검출된 사람이 있을 때
        tracks=sort.update(np.array(persons)) # 검출된 사람 정보와 이전 이력 정보를 보고 객체 내부에 있는 추적 정보 갱신
    
    for i in range(len(tracks)): # 추적 물체의 번호에 따라 색을 달리하여 직사각형과 물체 번호 표시
        x1,y1,x2,y2,track_id=tracks[i].astype(int) #물체 번호 저장 
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2) 
        cv.putText(frame,str(track_id),(x1+10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)            
    
    cv.imshow('Person tracking by SORT',frame)
    
    key=cv.waitKey(1) 
    if key==ord('q'): break 
    
cap.release()		# 카메라와 연결을 끊음
cv.destroyAllWindows()