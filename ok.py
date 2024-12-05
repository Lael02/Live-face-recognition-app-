import threading
import cv2
from deepface import DeepFace

viewer = cv2.VideoCapture(0, cv2.CAP_DSHOW)

viewer.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
viewer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

face_match = False

reference_img = cv2.imread("laella.jpg")


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match =True
        else:
            face_match= False


    except ValueError: 
        face_match = False


while True:
    ret, frame = viewer.read() 

    if ret:
        if counter % 38 == 8:
            try:
                threading.Thread(target=check_face, args=(frame.copy(), )).start()
            except ValueError:
               pass 
        counter += 1


    if face_match:
        cv2.putText(frame, "WELCOME LAEL!", (15, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
     
    else:
        cv2.putText(frame, "NOT A MATCH!", (15, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Video", frame)

    
    key = cv2.waitKey(1)
    if key == ord ("q"):
        break

cv2.destroyAllWindows()




#done with the help of NeuralNine