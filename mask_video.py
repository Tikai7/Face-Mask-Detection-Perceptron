import cv2
import numpy as np
import Ann


cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    X = cv2.resize(gray,(64,64))
    X = X.reshape(1, 64*64)/X.max()
    W = np.loadtxt("params.txt")
    W = W.reshape((X.shape[1], 1))
    b = np.loadtxt("params_b.txt")
    y_pred, proba = Ann.predict(X, W, b)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if y_pred:
        cv2.putText(frame,f"No Mask : {proba[0]}% ",(10,450),font,1,color=(0,0,255),thickness=3)
    else:
        cv2.putText(frame,f"Mask : {1-proba[0]}% ",(10,450),font,1,color=(0,255,0),thickness=3)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()