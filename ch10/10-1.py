import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img,flow,step=16): # 광류 맵 flow를 원본 영상 이미지에 그리는 함수
    for y in range(step//2,frame.shape[0],step): #step만큼 건너뛰며 화소에 접근
        for x in range(step//2,frame.shape[1],step): # step만큼 건너뛰며 화소에 접근
            dx,dy=flow[y,x].astype(np.int64) # 모션 벡터를 dy,dx 저장
            if(dx*dx+dy*dy)>1:
                cv.line(img,(x,y),(x+dx,y+dy),(0,0,255),2) # 큰 모션 있는 곳은 빨간색
            else:
                cv.line(img,(x,y),(x+dx,y+dy),(0,255,0),2) # 큰 모션 없는 곳은 초록색
    
cap=cv.VideoCapture(0,cv.CAP_DSHOW)	# 카메라와 연결 시도
if not cap.isOpened(): sys.exit('카메라 연결 실패')
    
prev=None # 시작 순간에는 이전 프레임이 없기 때문에 None으로 설정

while(1):
    ret,frame=cap.read()	# 비디오를 구성하는 프레임 획득
    if not ret: sys('프레임 획득에 실패하여 루프를 나갑니다.')
    
    if prev is None:	# 첫 프레임이면 광류 계산 없이 prev만 설정
        prev=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # 프레임 명암으로 변경해서 prev에 저장
        continue
    
    curr=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # 프레임을 명암으로 변경해서 curr에 저장
    flow=cv.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0) # 연속된 prev와 curr에서 광류맵을 추출하여 flow에 저장
    #이전영상, 현재영상,흐름 벡터가 저장될 행렬 생성, 피라미드 영상 축소 비율 0.5, 피라미드 영상 개수 3
    #평균 윈도우 크기 15, 알고리즘 반복 횟수 3, 다항식 확장을 위한 이웃 픽셀 크기 5, 가우시안 표준편차 1.2,flags 0
    draw_OpticalFlow(frame,flow)
    cv.imshow('Optical flow',frame)

    prev=curr

    key=cv.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv.destroyAllWindows() 