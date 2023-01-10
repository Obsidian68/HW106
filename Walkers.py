import cv2

people_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml') 

cap = cv2.VideoCapture('walking.avi')

while (True):
    
    ret, frame = cap.read()

    capgrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    people = people_cascade.detectMultiScale(capgrey, 1.1, 5)

    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 32:
        break

cap.release()
cv2.destroyAllWindows()
