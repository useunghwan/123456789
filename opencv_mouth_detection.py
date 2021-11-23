import cv2
import os

input_directory = 'mouse'  # 이미지 경로
output_directory = 'mouse_detected1'  # 크롭된 이미지 경로

smile_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml") # xml 파일 경로

for img in os.listdir(input_directory):  # 이미지 경로안의 이미지 전부를 for문으로 돌린다.
    frame = cv2.imread(os.path.join(input_directory, img))  # 이미지를 읽어들인다.
    #gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = smile_cascade.detectMultiScale(frame, scaleFactor=2, minNeighbors=5)  # 검출 알고리즘으로 이미지에서 입 위치를 찾아낸다.



    for x,y,w,h in faces :
        #image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # 얼굴이 검출되면 검출된 부분을 직사각형으로 검출된 이미지를 보여라.
        roi_gray = frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # For Mouth

        smile =  smile_cascade.detectMultiScale(roi_gray,1.9,10) # 검출 알고리즘으로 이미지에서 입 위치를 찾아낸다.


        for sx,sy,sw,sh in smile :
            #image = cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
            image = cv2.putText(roi_color,"",(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,0),2)

            image = image[sy:sy+sh,sx:sx+sw]  # boxplot 부분 좌표 넣어서 새로 이미지를 만든다.

            #cv2.imshow("Frame",image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            if not os.path.exists(output_directory):  # 아웃풋 이미지 경로가 존재하지 않으면
                os.makedirs(output_directory)  # 새로 해당 경로를 만들어준다.
                # cv2.imshow('detection', image)
            cv2.imwrite(output_directory + '/' + img, image)  # 아웃풋 이미지 경로에 크롭된 이미지를 새로 생성한다.
