# Vehicle Detection and Tracking System

A comprehensive vehicle detection and tracking system that processes video streams to identify and track multiple vehicle types in real-time using state-of-the-art computer vision techniques.

## 📋 Overview

This project implements an automated vehicle detection and tracking pipeline that can identify and track various vehicle types (cars, buses, trucks, motorcycles, bicycles, and trains) in video footage. The system processes each frame of the input video, detects vehicles, assigns unique tracking IDs, and generates an annotated output video with bounding boxes, center points, and labels.

## 🔧 Tools Used

### **YOLO11** - Vehicle Detection
- **Purpose**: Object detection and classification
- **Model**: YOLO11 Large (yolo11l.pt)
- **Capabilities**: 
  - Real-time object detection with high accuracy
  - Multi-class detection (80 COCO classes)
  - Optimized for vehicle classes: bicycle, car, motorcycle, bus, train, truck
  - GPU-accelerated inference for faster processing

### **BoT-SORT & ByteTrack** - Object Tracking
- **Purpose**: Multi-object tracking across video frames
- **BoT-SORT (Bot-SORT)**: 
  - Advanced tracking algorithm that combines motion and appearance features
  - Handles occlusions and re-identification
  - Maintains consistent tracking IDs across frames
- **ByteTrack**: 
  - High-performance tracking algorithm
  - Tracks both high and low confidence detections
  - Reduces ID switches and improves tracking continuity
- **Implementation**: Integrated via Ultralytics YOLO's built-in tracking capabilities

### **Python & OpenCV** - Implementation and Visualization
- **Python**: Core programming language for the entire pipeline
- **OpenCV (cv2)**: 
  - Video I/O operations (reading input, writing output)
  - Frame processing and manipulation
  - Visualization: drawing bounding boxes, center points, labels
  - Color encoding and text rendering

## 🎯 Methodology

### 1. **Vehicle Detection Phase**

The detection process uses YOLO11, which employs a single-stage detection architecture:

- **Input Processing**: Each video frame is fed into the YOLO11 model
- **Feature Extraction**: The model extracts multi-scale features using a CNN backbone
- **Detection Head**: Predicts bounding boxes, class probabilities, and confidence scores
- **Non-Maximum Suppression (NMS)**: Removes duplicate detections and filters low-confidence predictions
- **Class Filtering**: Only vehicle classes are retained (classes 1, 2, 3, 5, 6, 7):
  - Class 1: Bicycle
  - Class 2: Car
  - Class 3: Motorcycle
  - Class 5: Bus
  - Class 6: Train
  - Class 7: Truck

### 2. **Object Tracking Phase**

The tracking system maintains vehicle identities across frames:

- **Initialization**: First detection of a vehicle assigns a unique tracking ID
- **Motion Prediction**: Kalman filter predicts the next position based on velocity
- **Data Association**: 
  - IoU (Intersection over Union) matching between detections and existing tracks
  - Appearance features for re-identification after occlusions
  - Hungarian algorithm for optimal assignment
- **Track Management**:
  - Active tracks: Vehicles currently visible
  - Lost tracks: Temporarily occluded vehicles (maintained for re-identification)
  - Removed tracks: Vehicles that have left the scene
- **Persistence**: `persist=True` ensures tracking IDs are maintained across frames

### 3. **Visualization Phase**

Each detected and tracked vehicle is annotated with:

- **Bounding Box**: Green rectangle around the vehicle
- **Center Point**: Red dot at the geometric center of the bounding box
- **Label**: White text showing vehicle class name and tracking ID (e.g., "car | ID 5")

### 4. **Video Processing Pipeline**

```
Input Video → Frame Extraction → Detection → Tracking → Visualization → Output Video
```

1. **Frame Reading**: OpenCV reads frames sequentially from input video
2. **Detection**: YOLO11 processes each frame to detect vehicles
3. **Tracking**: Tracking algorithm associates detections with existing tracks
4. **Annotation**: Visual elements are drawn on the frame
5. **Output Writing**: Processed frame is written to output video file

## 📁 Project Structure

```
Vehicle Detection/
├── Code/
│   └── vehicle-detection.ipynb    # Main implementation notebook
├── Input/
│   └── Test Video for Vehicle Counting Model.mp4  # Input video file
├── Model/
│   └── yolo11l.pt                  # YOLO11 Large model weights
├── Output/
│   └── output.mp4                  # Processed output video
└── README.md                       # This file
```

## 🚀 How It Works

### Step-by-Step Process

1. **Model Loading**: The YOLO11 Large model is loaded from the pre-trained weights
2. **Video Initialization**: Input video is opened and video properties (width, height, FPS) are extracted
3. **Output Setup**: Video writer is configured to save processed frames
4. **Frame-by-Frame Processing**:
   - Read frame from input video
   - Run YOLO tracking on the frame
   - Extract bounding boxes, tracking IDs, and class indices
   - Draw visualizations for each detected vehicle
   - Write annotated frame to output video
5. **Cleanup**: Release video resources and finalize output

### Key Features

- ✅ **Multi-Vehicle Detection**: Simultaneously detects multiple vehicle types
- ✅ **Persistent Tracking**: Maintains vehicle IDs across frames
- ✅ **Real-time Processing**: Optimized for efficient video processing
- ✅ **GPU Acceleration**: Supports CUDA for faster inference
- ✅ **Visual Annotations**: Clear bounding boxes, center points, and labels
- ✅ **Configurable**: Easy to adjust visualization parameters

## 🛠️ Technical Details

### Detection Classes
- **Bicycle** (Class ID: 1)
- **Car** (Class ID: 2)
- **Motorcycle** (Class ID: 3)
- **Bus** (Class ID: 5)
- **Train** (Class ID: 6)
- **Truck** (Class ID: 7)

### Configuration Parameters
- **Font Scale**: 0.45 (for label text)
- **Box Thickness**: 2 pixels
- **Output Frame Rate**: 30 FPS
- **Bounding Box Color**: Green (0, 200, 0)
- **Center Point Color**: Red (0, 0, 255)
- **Label Color**: White (255, 255, 255)

## 📊 Performance Considerations

- **GPU Support**: The system automatically detects and uses CUDA if available
- **Processing Speed**: Depends on video resolution, frame rate, and hardware
- **Model Size**: YOLO11 Large provides a good balance between accuracy and speed
- **Memory Usage**: Efficient frame-by-frame processing minimizes memory footprint

## 🔍 Use Cases

- Traffic monitoring and analysis
- Vehicle counting systems
- Parking lot management
- Highway surveillance
- Urban planning and traffic flow analysis
- Security and surveillance applications

## 📝 Requirements

- Python 3.7+
- Ultralytics YOLO
- OpenCV (cv2)
- PyTorch (for GPU support)
- NumPy
- SciPy

## 🎓 Key Concepts Explained

### Why YOLO11?
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that processes entire images in a single pass, making it extremely fast and suitable for real-time applications. YOLO11 is the latest version with improved accuracy and efficiency.

### Why Multi-Object Tracking?
Simple detection only identifies objects in each frame independently. Tracking adds temporal consistency by maintaining identities across frames, enabling:
- Vehicle counting
- Path analysis
- Behavior understanding
- Long-term monitoring

### Why BoT-SORT & ByteTrack?
These algorithms are specifically designed to handle challenges in multi-object tracking:
- **Occlusions**: When vehicles are temporarily hidden
- **ID Switches**: Preventing identity confusion
- **Re-identification**: Recovering tracks after occlusions
- **Real-time Performance**: Fast enough for video processing

---
