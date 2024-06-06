
# Car Detection using Haar Cascade Classifier

## Introduction

This project demonstrates how to use OpenCV's Haar Cascade classifier to detect cars in images and video streams. The Haar Cascade classifier is a machine learning-based approach for object detection proposed by Paul Viola and Michael Jones in their 2001 paper "Rapid Object Detection using a Boosted Cascade of Simple Features".

## Requirements

To run this project, you need to have the following installed:

- Python 3.x
- OpenCV (cv2)
- Numpy

## Installation

1. Clone the repository or download the source code.
   ```bash
   git clone https://github.com/aliyuldashev/BSD_system.git
   cd car-detection-haar-cascade
   ```

2. Install the required Python packages.
   ```bash
   pip install opencv-python numpy
   ```

## Usage

### Detecting Cars in Images

To detect cars in a static image, use the following script:

```python
import cv2

# Load the pre-trained car classifier (Haar cascade XML file)
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Read the image
image = cv2.imread('car_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect cars
cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected Cars', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Detecting Cars in Real-Time Video

To detect cars in a video stream (e.g., from a webcam), use the following script:

```python
import cv2

# Load the pre-trained car classifier (Haar cascade XML file)
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Open the video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Real-Time Car Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
```

### Parameters Explanation

- `scaleFactor`: Specifies how much the image size is reduced at each image scale. A value of 1.1 means the image is reduced by 10% at each scale.
- `minNeighbors`: Specifies how many neighbors each candidate rectangle should have to retain it. Higher values mean fewer detections but with higher quality.
- `minSize`: Specifies the minimum object size. Objects smaller than this size are ignored.

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Viola-Jones Object Detection Framework](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
- [Pre-trained Haar Cascades for Object Detection](https://github.com/opencv/opencv/tree/master/data/haarcascades)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- Paul Viola and Michael Jones for the Viola-Jones object detection framework.
- OpenCV team for providing extensive computer vision tools and libraries.

---

Feel free to customize this README file according to your project's specifics, such as repository URL, contribution guidelines, or any additional instructions.