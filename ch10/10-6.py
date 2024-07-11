import cv2 as cv
import mediapipe as mp

dice=cv.imread('dice.png',cv.IMREAD_UNCHANGED)	# 증강 현실에 쓸 장신구, 알파 채널을 포함해 4개 채널 모두 읽어옴 
dice=cv.resize(dice,dsize=(0,0),fx=0.1,fy=0.1) # 영상을 10%축소
w,h=dice.shape[1],dice.shape[0] #너비와 높이 저장

mp_face_detection=mp.solutions.face_detection # face_detection 모듈을 읽어 저장
mp_drawing=mp.solutions.drawing_utils # solutions에서 검출 결과를 그리는데 사용하는 drawing_utils 모듈을 읽어 저장
#0은 카메라로부터 2미터 이내로 가깝게 있을 때 적합,1은 5미터 이내에 적합, 검출 신뢰도가 설정한 값보다 큰 경우 검출 성공
face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) # 얼굴 검출에 쓸 객체 생성 

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res=face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) #프레임 얼굴 검출 수행 후 저장
    
    if res.detections:
        for det in res.detections:
            p=mp_face_detection.get_key_point(det,mp_face_detection.FaceKeyPoint.RIGHT_EYE) #det에서 오른쪽 눈 위치를 꺼내 저장
             # 오른쪽 눈을 중심으로 x,y 좌표
            x1,x2=int(p.x*frame.shape[1]-w//2),int(p.x*frame.shape[1]+w//2)
            y1,y2=int(p.y*frame.shape[0]-h//2),int(p.y*frame.shape[0]+h//2)
            if x1>0 and y1>0 and x2<frame.shape[1] and y2<frame.shape[0]: # 장신구 영상이 원본 영상 안에 있으면
                alpha=dice[:,:,3:]/255 # 0~2는 rgb, 3은 투명도를 나타내는 알파채널
                frame[y1:y2,x1:x2]=frame[y1:y2,x1:x2]*(1-alpha)+dice[:,:,:3]*alpha # 프레임과 장신구 혼합
            
    cv.imshow('MediaPipe Face AR',cv.flip(frame,1))
    if cv.waitKey(5)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()