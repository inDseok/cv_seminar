import numpy as np
import cv2 as cv

cap=cv.VideoCapture('slow_traffic_small.mp4') # 비디오 cap에 저장

#최대 특징점의 수, 임계값, 검출된 특징점들 간의 최소 거리, 특징점 검출을 위한 이웃 픽셀의 크기
feature_params=dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
#각 이동 벡터를 추정하기 위한 윈도우의 크기, 이미지 피라미드의 최대 레벨 수, 반복 횟수 10회 or 정확도 0.03이면 알고리즘 종료
lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,10,0.03))

color=np.random.randint(0,255,(100,3)) # 추적 경로를 색으로 구분하기위한 난수

ret,old_frame=cap.read() # 첫 프레임
old_gray=cv.cvtColor(old_frame,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환하여 저장 
p0=cv.goodFeaturesToTrack(old_gray,mask=None,**feature_params) # 함수를 통해 특징점 추출

mask=np.zeros_like(old_frame)	# 물체의 이동 궤적을 그릴 영상 

while(1):
    ret,frame=cap.read()
    if not ret: break

    new_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환하여 저장
    p1,match,err=cv.calcOpticalFlowPyrLK(old_gray,new_gray,p0,None,**lk_params)	# lucas-kanade 광류 계산 후 새로운 특징점을 찾음
    # 이전 p0와 새로 찾은 특징점 매칭하여 성공 여부 파악,새로 찾은 특정점과 매칭 성공 여부와 매칭 오류 반환
    
    if p1 is not None:		# 양호한 쌍 선택
        good_new=p1[match==1] # 매칭에 성공한 특징점
        good_old=p0[match==1] # 매칭에 성공한 특징점
        
    for i in range(len(good_new)): # 이동 궨적 그리기
        a,b=int(good_new[i][0]),int(good_new[i][1]) #매칭에 성공한 특징점에서 추출된 새로운 x, y 좌표
        c,d=int(good_old[i][0]),int(good_old[i][1]) #매칭에 성공한 특징점에서 추출된 이전 x, y 좌표
        mask=cv.line(mask,(a,b),(c,d),color[i].tolist(),2) # 선으로 그리기
        frame=cv.circle(frame,(a,b),5,color[i].tolist(),-1) # 원으로 그리기
        
    img=cv.add(frame,mask) #프레임과 마스크 섞은 영상
    cv.imshow('LTK tracker',img)
    cv.waitKey(30)

    old_gray=new_gray.copy()	# 이번 것이 이전 것이 됨
    p0=good_new.reshape(-1,1,2) # 현재 프레임에서 매칭이 일어난 특징점을 이전 특징점으로 이전
    
cv.destroyAllWindows()