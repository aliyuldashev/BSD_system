
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
   pip install opencv-python numpy cv2 picamera  threading spidev RPi  time
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