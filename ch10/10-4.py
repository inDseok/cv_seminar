import cv2 as cv
import mediapipe as mp

img=cv.imread('BSDS_376001.jpg') # 이미지 읽어옴

mp_face_detection=mp.solutions.face_detection # face_detection 모듈을 읽어 저장
mp_drawing=mp.solutions.drawing_utils # solutions에서 검출 결과를 그리는데 사용하는 drawing_utils 모듈을 읽어 저장

#0은 카메라로부터 2미터 이내로 가깝게 있을 때 적합,1은 5미터 이내에 적합, 검출 신뢰도가 설정한 값보다 큰 경우 검출 성공
face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5) # 얼굴 검출에 쓸 객체 생성 
res=face_detection.process(cv.cvtColor(img,cv.COLOR_BGR2RGB)) # 실제 검출을 수행하고 결과 저장

if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    for detection in res.detections:
        print(dir(detection))
        cv.waitKey()
        mp_drawing.draw_detection(img,detection) # 검출된 각각의 얼굴을 원본 이미지에 표시
    cv.imshow('Face detection by MediaPipe',img)


cv.destroyAllWindows()
#print(res.detections)