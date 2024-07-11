#이진 영역의 트징을 추출하는 함수 사용하기
import skimage
import cv2 as cv
import numpy as np

orig=skimage.data.horse()
img=255-np.uint8(orig)*255 #말은 255 배경은 0으로 변환
cv.imshow('Horse',img)

#가장 바깥쪽 경계선 찾기
contours,hierarchy=cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img2=cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2,contours,-1,(255,0,255),2)
cv.imshow('Horse with contour',img2)

contour=contours[0]
m=cv.moments(contour) #모멘트 추출
area=cv.contourArea(contour) #면적 계산(m[m00]와 같음)
cx,cy=m['m10']/m['m00'],m['m01']/m['m00'] #중점 계산
perimeter=cv.arcLength(contour,True) #둘레 계산
roundness=(4.0*np.pi*area)/(perimeter*perimeter) #둥근 정도 계산
print('면적=',area,'\n중점={',cx,',',cy,')','\n둘레=',perimeter,'\n둥근 정도=',roundness)

img3=cv.cvtColor(img, cv.COLOR_GRAY2BGR)

contour_approx=cv.approxPolyDP(contour, 8,True) #경계선을 직선으로 근사화
cv.drawContours(img3, [contour_approx], -1,(0,255,0),2) #직선 근사화 결과 표시

hull=cv.convexHull(contour) #볼록 헐
hull=hull.reshape(1,hull.shape[0],hull.shape[2])
cv.drawContours(img3, hull,-1,(0,0,255),2)

cv.imshow('Hores with line segments and convex hull',img3)

cv.waitKey()
cv.destroyAllWindows()