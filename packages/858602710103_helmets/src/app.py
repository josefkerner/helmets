import cv2, math

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


from ultralytics import YOLO
model = YOLO("/panorama/best.pt")
#model = YOLO("models/hemletYoloV8_100epochs.pt")

print(model.names)



'''
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
              
'''

classNames = [
    "helmet", "head", "person"
]



import cv2

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
                obj = "Unknown"
                print("Class name -->", "Unknown")
            else:
                obj = model.names[cls]
                objects.append(obj)
                #print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            coords.append([x1,y1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        if len(coords) > 0:
            if len(coords) > 1:
                distance = math.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
            else:
                distance = 'no'

            if len(objects) == 1 and objects[0] == 'helmet':
                txt = f"Wears helmet"
            else:
                txt = f"No helmet"

            cv2.putText(img, txt, coords[0], font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()