

ip_addr = '19'
creds = 'root:trustsoft1!'
import cv2
cap = cv2.VideoCapture(f'rtsp://{creds}@{ip_addr}/live1s1.sdp')

#cap = cv2.VideoCapture(f'http://{creds}@{ip_addr}/video')

i = 0

while True:
    print(i)
    ret, frame = cap.read()
    #save img into a file
    cv2.imshow('frame',frame)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

