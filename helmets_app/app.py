import cv2, math
from datetime import datetime
# start webcam


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


window_width = 1920
window_height = 1100
screen_width = 1920
screen_height = 1100
stream_window = 'Webcam'

'''
cv2.namedWindow(
        stream_window,
        flags=(cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO))
cv2.setWindowProperty(stream_window, cv2.WND_PROP_TOPMOST, 1.0)
cv2.setWindowProperty(stream_window, cv2.WND_PROP_FULLSCREEN, 1.0)
cv2.resizeWindow(
        stream_window,
        window_width,
        window_height)
cv2.moveWindow(
        stream_window,
        screen_width - window_width,
        screen_height - window_height - 40)


ip_addr = '192.168.139.102'
creds = 'root:trustsoft1!'
cap = cv2.VideoCapture(f'rtsp://{creds}@{ip_addr}/live1s1.sdp')


'''




from ultralytics import YOLO
model = YOLO("best.pt")
#model = YOLO("models/hemletYoloV8_100epochs.pt")

print(model.names)


classNames = [
    "helmet", "head", "person"
]



import cv2
from typing import Dict, List



def postprocess_labels(img, detections: List[Dict]):
    for detection in detections:
        print(detection)

        coords = [detection['x1'], detection['y1']]
        if detection['label'] == "helmet":
            #count heads
            heads = [d for d in detections if d['label'] == 'head']
            #if confidence is greater than 0.5
            if detection['confidence'] > 0.8:
                #if there are no heads
                txt= f'Helmet ON - {detection["confidence"]}'
            else:

                txt = 'No helmet'
        else:
            txt = 'No helmet'

            x1 = detection['x1']
            y1 = detection['y1']
            y2 = detection['y2']
            x2 = detection['x2']
        font = cv2.FONT_HERSHEY_SIMPLEX
        # red color
        color = (0, 0, 255)
        fontScale = 1
        thickness = 2
        #if text starts with 'Wearing helmet'
        if txt.startswith('Helmet ON'):
            #set color to green
            color = (0, 255, 0)

        cv2.putText(img, txt, coords, font, fontScale, color, thickness)


while True:
    success, img = cap.read()

    #now
    start = datetime.now()

    results = model(img, stream=True)


    # coordinates
    for r in results:
        boxes = r.boxes

        coords = []
        objects = []

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            if cls > len(classNames):
                label = "Unknown"
                print("Class name -->", "Unknown")
            else:
                label = model.names[cls]
                obj = {
                    "label": label,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
                objects.append(obj)
                #print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            coords.append([x1,y1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        #yellow color
        color = (255, 255, 0)
        thickness = 2
        postprocess_labels(img,objects)

    end = datetime.now()
    print('took', end-start)
    print(f'FPS: {1/(end-start).total_seconds()}')

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()