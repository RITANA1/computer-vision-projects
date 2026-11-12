# Image Segmentation with YOLO

A real-time **instance segmentation** project using YOLO and OpenCV. It detects individual objects through a webcam, generates separate masks, and displays the results live.

## Features

- Real-time webcam segmentation
- YOLO instance segmentation
- Individual colored masks for detected objects
- Object labels and confidence scores
- FPS monitoring
- Screenshot capture
- Video recording

## Technologies

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- PyTorch

## Installation

Clone the repository:

```bash
git clone https://github.com/RITANA1/computer-vision-projects.git
cd computer-vision-projects
```

Install dependencies:

```bash
pip install -r projects/image_segmentation/requirements.txt
```

## Run the project

From the repository root:

```bash
python projects/image_segmentation/src/main.py
```

The pretrained YOLO segmentation model will be downloaded automatically if it is not already available.

## Controls

| Key | Action |
|---|---|
| Q | Quit |
| S | Save screenshot |

## How it works

1. Capture frames from the webcam.
2. Pass each frame to the YOLO segmentation model.
3. Detect objects and generate individual masks.
4. Overlay colored masks and labels on the frame.
5. Display the processed video with FPS.

## Learning Outcomes

- Understanding instance segmentation
- Working with binary masks
- Applying pretrained deep learning models
- Real-time computer vision with OpenCV
- Visualizing model predictions

## Author

**Rita**

GitHub: [@RITANA1](https://github.com/RITANA1)