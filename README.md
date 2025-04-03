# Cross-road-video-analysis
A computer vision system that analyzes vehicle traffic at intersections using YOLOv8 object detection and tracking.

https://github.com/user-attachments/assets/6c9ed899-8e78-4627-8415-fca48e8d8b9a

## Features

- **Real-time vehicle tracking** using YOLOv8
- **Directional traffic counting** (incoming/outgoing vehicles per road)
- **Intersection visualization** with overlaid road maps
- **Perspective transformation** for top-down traffic flow analysis
- **Visual analytics dashboard** showing real-time counts

## Sample Video frame
![saved_frame](https://github.com/user-attachments/assets/2160e527-9050-49a5-8cf6-47c9842b296a)

## Sample Output

![saved_frame_3](https://github.com/user-attachments/assets/508c0017-b39c-47fc-b306-09c1ed0d08e1)

## How It Works

1. **Vehicle Detection**: Uses YOLOv8 to detect cars and trucks
2. **Tracking**: Assigns persistent IDs to vehicles
3. **Counting**: Tracks vehicles crossing virtual lines at each road entrance/exit
4. **Mapping**: Applies homography for top-down perspective
5. **Visualization**: Overlays analytics on the video stream

## Technical Details

- **Detection Model**: YOLOv8l (can use smaller variants)
- **Tracking**: ByteTrack algorithm (built into YOLOv8)
- **Counting**: Vector-based line crossing detection
- **Mapping**: Perspective transformation using OpenCV
- **Visualization**: Custom overlays with cvzone

## Requirements

- Python 3.8+
- OpenCV 4.5+
- Ultralytics (for YOLOv8)
- cvzone
- NumPy


