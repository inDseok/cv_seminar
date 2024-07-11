import cv2 as cv
import mediapipe as mp

mp_hand=mp.solutions.hands # 손 검출을 담당하는 hands 모듈을 읽어 저장
mp_drawing=mp.solutions.drawing_utils # 검출 결과 그리는데 쓰이는 drwaing_utils 모듈을 읽어 저장 
mp_styles=mp.solutions.drawing_styles # 그리는 유형을 지정하는데 쓰는 drawing_styles 모듈을 읽어 저장
# 손 랜드마크 검출에 쓸 객체
# 최대 손 2개, static_image_mode=False는 입력을 비디오로 간주하고 첫 프레임에 blazehand를 적용하고 이후에는 추적을 사용
hand=mp_hand.Hands(max_num_hands=2,static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 웹 캠 연결

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=hand.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) # 손 검출 수행 후 저장
    
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks: # 검출된 손 각가에 그물망을 그림
            mp_drawing.draw_landmarks(frame,landmarks,mp_hand.HAND_CONNECTIONS,mp_styles.get_default_hand_landmarks_style(),mp_styles.get_default_hand_connections_style())

    cv.imshow('MediaPipe Hands',cv.flip(frame,1))	# 좌우반전
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()