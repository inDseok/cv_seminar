import cv2 as cv
import mediapipe as mp

mp_pose=mp.solutions.pose # 자세 추정을 담당하느 pose 모듈을 읽어 저장 
mp_drawing=mp.solutions.drawing_utils # 검출 결과 그리는데 쓰이는 drwaing_utils 모듈을 읽어 저장 
mp_styles=mp.solutions.drawing_styles # 그리는 유형을 지정하는데 쓰는 drawing_styles 모듈을 읽어 저장
# static_image_mode=False는 입력을 비디오로 간주하고 첫 프레임에 ROI를 적용하고 이후에는 추적을 사용
# enable_segmentation=True 정경과 배경을 분할 하라고 지시 -> 배경을 흐릿하게 만드는 등의 응용에 쓸 수 있음
pose=mp_pose.Pose(static_image_mode=False,enable_segmentation=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 웹캠 연결

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=pose.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) #자세 추정 수행 후 저장
    # 자세를 원본 영상에 겹쳐 보여줌
    mp_drawing.draw_landmarks(frame,res.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    
    cv.imshow('MediaPipe pose',cv.flip(frame,1)) # 좌우반전
    if cv.waitKey(5)==ord('q'):
      mp_drawing.plot_landmarks(res.pose_world_landmarks,mp_pose.POSE_CONNECTIONS) #자세 추정 결과 시각화 하여 보여줌
      break

cap.release()
cv.destroyAllWindows()