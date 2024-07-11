import cv2 as cv
import mediapipe as mp

mp_face_detection=mp.solutions.face_detection # face_detection 모듈을 읽어 저장
mp_drawing=mp.solutions.drawing_utils # solutions에서 검출 결과를 그리는데 사용하는 drawing_utils 모듈을 읽어 저장
#0은 카메라로부터 2미터 이내로 가깝게 있을 때 적합,1은 5미터 이내에 적합, 검출 신뢰도가 설정한 값보다 큰 경우 검출 성공
face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) # 얼굴 검출에 쓸 객체 생성 

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 웹 캠 연결 

while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break
    
    res=face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) #프레임 얼굴 검출 수행 후 저장
    
    if res.detections:
        for detection in res.detections:
            mp_drawing.draw_detection(frame,detection) #프레임에 검출된 얼굴 표시
            
    cv.imshow('MediaPipe Face Detection from video',cv.flip(frame,1)) # 얼굴 영상을 보여주는데 flip을 통해 거울모드로 보여짐
    if cv.waitKey(5)==ord('q'):
        break

cap.release()
cv.destroyAllWindows()