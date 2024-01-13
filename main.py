import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('input_video/8.mp4')

my_file = open("utils/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data.cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    # Draw a filled polygon (quadrilateral) on the frame
    # Person
    point0 = np.array([(737, 220), (744, 186), (786, 193), (787, 221)], dtype=np.int32)
    point1 = np.array([(442, 180), (495, 185), (455, 204), (402, 195)], dtype=np.int32)
    cv2.polylines(frame, [point0,point1], isClosed=True, color=(255, 0, 0), thickness=2)

    # Crossing
    point3= np.array([(508,183), (740, 198), (721, 218), (467, 204)], dtype=np.int32)
    cv2.polylines(frame, [point3], isClosed=True, color=(0, 0, 255), thickness=2)

    #Road
    point4 = np.array([(631,119), (750, 135), (749, 174), (526, 170)], dtype=np.int32)
    point5 = np.array([ (733, 231),(446, 211), (149, 381), (710, 446)], dtype=np.int32)
    cv2.polylines(frame, [point4,point5], isClosed=True, color=(0, 255, 0), thickness=2)

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car'in c.lower() or 'motorcycle' in c.lower() or 'person'in c.lower():
            # Check if the vehicle is within the specified regions
            if cv2.pointPolygonTest(point4, (x1, y1), False) >= 0 or cv2.pointPolygonTest(point5, (x1, y1), False) >= 0:
                # Draw a purple rectangle for vehicles inside the purple region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(frame, c, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)



    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
