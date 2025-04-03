from ultralytics import YOLO
import cv2
import cvzone
import  numpy as np

model = YOLO("yolov8l.pt")  # You can use yolov8s.pt, yolov8m.pt, etc.

car_data = {}

count_in = [[],[],[],[]]  # list of car ids that comes into road number 1,2,3,4
count_out = [[],[],[],[]] # list of car ids that comes out from road number 1,2,3,4

points = [[(350,550),(480,780)],
         [(400,520),(950,470)],
         [(1120,500),(1420,650)],
         [(570,800),(1420,700)]]

# list of lines equation y = mx + p (m and p) manually calculated, these lines were calculated from the points above

lines_equations = [[1.77,-70],
                   [-0.091,556],
                   [0.5,-60],
                   [-0.12,868]]



# Define corresponding points (original -> mapped) for homography (mapping the road) manually labeled
src_points = np.array([[525,780], [385,530], [1025,490], [1410,670]], dtype=np.float32)
dst_points = np.array([[125,220], [125,135], [225,135], [225,220]], dtype=np.float32)
# Compute homography of the central point
H, _ = cv2.findHomography(src_points, dst_points)

def calculateMappingPoint(x, y):
    global  H
    point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_point = cv2.perspectiveTransform(point, H)
    x_m, y_m = transformed_point[0][0]
    return int(x_m), int(y_m)


# Open the video file
video_path = "/home/aous/Desktop/personal projects/CrossRoadAnalysis/cap.mp4"  # Path to input video
cap = cv2.VideoCapture(video_path)

# Define the class name for cars
className = 'car'

# Load the mask image
mask_path = '/home/aous/Desktop/personal projects/CrossRoadAnalysis/mask.png'

mask = cv2.imread(mask_path)

def checkCrossingLine(x,y, id):
    global car_data
    global lines_equations
    global points
    for i, line in enumerate(lines_equations):
        if abs(y - (line[0] * x + line[1])) <= 20:
            if count_out[i].count(id) == 0 and car_data[id]['flag'] == '':
                count_out[i].append(id)
                car_data[id]['flag'] = 'o'
                car_data[id]['line_id'] = i
                cv2.line(img, points[i][0], points[i][1], (0, 0, 255), 2)

            if count_in[i].count(id) == 0 and car_data[id]['flag'] == 'o' and car_data[id]['line_id'] != i:
                count_in[i].append(id)
                cv2.line(img, points[i][0], points[i][1], (0, 255, 0), 2)


# Loop through the video frames
while True:
    # Read a frame from the video
    success, img = cap.read()
    graphicIMG = cv2.imread('graphic.png',cv2.IMREAD_UNCHANGED)
    graphicIMG = cv2.resize(graphicIMG, (350, 350))
    mappingIMG = cv2.imread('mapping.png',cv2.IMREAD_UNCHANGED)
    mappingIMG = cv2.resize(mappingIMG, (350, 350))
    # # graphicIMG = cv2.resize(graphicIMG,(200,200))
    # img = cvzone.overlayPNG(img, graphicIMG, (3150,0))
    if not success:
        break  # Exit the loop if no more frames are available

    # Resize the frame
    img = cv2.resize(img, (1670, 940))
    img = cvzone.overlayPNG(img, graphicIMG, (1670-350, 0))
    img = cvzone.overlayPNG(img, mappingIMG, (0, 0))
    cv2.line(img,points[0][0],points[0][1],(255,0,0),2)
    cv2.line(img,points[1][0],points[1][1],(255,0,0),2)
    cv2.line(img,points[2][0],points[2][1],(255,0,0),2)
    cv2.line(img,points[3][0],points[3][1],(255,0,0),2)
    cvzone.putTextRect(img, f'Road 1', (320, 820), scale=1.5,colorT=(0,0,255),colorB=(255,0,0),colorR=(0,255,0))
    cvzone.putTextRect(img, f'Road 2', (280, 480), scale=1.5,colorT=(0,0,255),colorB=(255,0,0),colorR=(0,255,0))
    cvzone.putTextRect(img, f'Road 3', (1180, 450), scale=1.5,colorT=(0,0,255),colorB=(255,0,0),colorR=(0,255,0))
    cvzone.putTextRect(img, f'Road 4', (1550, 750), scale=1.5,colorT=(0,0,255),colorB=(255,0,0),colorR=(0,255,0))
    imgRegion = cv2.bitwise_and(img, mask)

    # Perform tracking (instead of prediction)
    results = model.track(imgRegion, persist=True, classes=[2, 7])  # Track only cars and trucks (class IDs 2 and 7)

    # Process tracking results
    for r in results:
        boxes = r.boxes  # Get bounding boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Get the tracking ID
            track_id = int(box.id[0]) if box.id is not None else None  # Tracking ID
            if track_id not in car_data:
                car_data[track_id] = {
                    'flag' : '',
                    'line_id' : -1
                    }
            # Draw bounding box and tracking ID
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'Car ID: {track_id}', (x1, y1), scale=1.8)

            # Get center of the bounding box
            cx, cy = x1 + w//2, y1 + h//2

            # Map a new point in original to show it in the top left of the video
            x_mapped, y_mapped = calculateMappingPoint(cx, cy)

            # Draw a circle in the mapping area:
            cv2.circle(img,(x_mapped,y_mapped),7,(255,0,255),-1)
            # cv2.ci

            # check if the car has crossed one of the lines
            checkCrossingLine(cx,cy, track_id)
            # cvzone.putTextRect(img, f'{track_id}', (1360, 140), scale=1.2, colorR=(255,0,0)) # in 1
            cv2.putText(img,f'{len(count_out[0])}',
                        fontScale=1,org=(1360,200),fontFace=2,color=(255,0,0),thickness=2)
            cv2.putText(img,f'{len(count_in[0])}',
                        fontScale=1,org=(1360,150),fontFace=2,color=(255,0,0),thickness=2)
            cv2.putText(img, f'{len(count_out[1])}',
                        fontScale=1, org=(1510, 75),fontFace=2,color=(255, 0, 0), thickness=2)
            cv2.putText(img, f'{len(count_in[1])}',
                        fontScale=1, org=(1510, 25), fontFace=2, color=(255, 0, 0),thickness=2)
            cv2.putText(img, f'{len(count_out[2])}',
                        fontScale=1, org=(1640, 200), fontFace=2, color=(255, 0, 0),thickness=2)
            cv2.putText(img, f'{len(count_in[2])}',
                        fontScale=1, org=(1640, 150),fontFace=2, color=(255, 0, 0),thickness=2)
            cv2.putText(img, f'{len(count_out[3])}',
                        fontScale=1, org=(1510, 340),fontFace=2, color=(255, 0, 0),thickness=2)
            cv2.putText(img, f'{len(count_in[3])}',
                        fontScale=1, org=(1510, 290), fontFace=2, color=(255, 0, 0),thickness=2)


    # Display the frame
    cv2.imshow('Tracking', img)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

