import cv2 as cv
import mediapipe as mp

mp_mesh=mp.solutions.face_mesh # face_mesh(얼굴 그물망 검출) 모듈을 읽어 저장
mp_drawing=mp.solutions.drawing_utils # 검출 결과 그리는데 쓰이는 drwaing_utils 모듈을 읽어 저장 
mp_styles=mp.solutions.drawing_styles # 그리는 유형을 지정하는데 쓰는 drawing_styles 모듈을 읽어 저장
# max_num_faces=2는 얼굴 2개까지 처리, refine_landmarks=True는 눈과 입에 있는 랜드마크 더 정교하게 검출
#min_detection_confidence=0.5는 얼굴 검출 신뢰도가 0.5이상일 때 성공을 간주
#min_tracking_confidence=0.5는 추적 신뢰도가 0.5보다 작으면 실패로 간주하고 새로 얼굴 검출 수행
mesh=mp_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW) #웹캠 연결

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=mesh.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB)) # 그물망 검출 수행 후 저장
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks: #그물망, 얼굴 경계 눈 눈썹 , 눈동자 그림
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_IRISES,landmark_drawing_spec=None,connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
        
    cv.imshow('MediaPipe Face Mesh',cv.flip(frame,1))		# 좌우반전
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()