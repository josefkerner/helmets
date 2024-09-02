import cv2, math

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

'''

ip_addr = '192.168.128.225'
creds = 'root:trustsoft1!'
cap = cv2.VideoCapture(f'rtsp://{creds}@{ip_addr}/live1s1.sdp')

'''


from ultralytics import YOLO
model = YOLO("helmets_app/runs/detect/train6/weights/best.pt")
#model = YOLO("models/hemletYoloV8_100epochs.pt")

print(model.names)


classNames = [
    "helmet", "head", "person"
]



import cv2
from typing import Dict, List

def postprocess_labels(detections: List[Dict]):
    for detection in detections:
        print(detection)

        coords = [detection['x1'], detection['y1']]
        if detection['label'] == "helmet":
            #count heads
            heads = [d for d in detections if d['label'] == 'head']
            #if confidence is greater than 0.5
            if detection['confidence'] > 0.5:
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
        #if text starts with 'Wearing helmet'
        if txt.startswith('Wearing helmet'):
            #set color to green
            color = (0, 255, 0)

            cv2.putText(img, txt, coords, font, fontScale, color, thickness)



while True:
    success, img = cap.read()
    results = model(img, stream=True)

    print(results)

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
        color = (255, 0, 0)
        thickness = 2
        postprocess_labels(objects)



    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()