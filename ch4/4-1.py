#소벨 에지
import cv2 as cv

img=cv.imread('soccer.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY) #컬러영상 명암으로 변환

#Sobel함수로 적용, 결과 영상을 32비트 실수맵으로 저장
grad_x=cv.Sobel(gray, cv.CV_32F, 1,0, ksize=3) 
grad_y=cv.Sobel(gray, cv.CV_32F, 0,1, ksize=3) 

#음수가 포함된 맵에 절댓값을 취해 양수로 변환(0보다 작은 값은 0, 255보다 큰 값은 255)
sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

#에지 강도 계산
edge_strength=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)

cv.imshow('Original', gray)
cv.imshow('sobelx', sobel_x)
cv.imshow('sobely', sobel_y)
cv.imshow('edge strength', edge_strength)

cv.waitKey()
cv.destroyAllWindows()