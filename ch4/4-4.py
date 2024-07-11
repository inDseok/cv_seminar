#허프 변환을 이용해 사과 검출하기
import cv2 as cv

img=cv.imread('apples.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#에지 방향정보 추가, 입력영상과 똑같은 사이즈
#원사이의 최소 거리 ,캐니 에지의 T(high), 비최대 억제 임계값
apples=cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 100,param1=100,param2=50,
                       minRadius=30,maxRadius=120)

for i in apples[0]:
    cv.circle(img,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)
    
cv.imshow('Apple detection', img)

cv.waitKey()
cv.destroyAllWindows()