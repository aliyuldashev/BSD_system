import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from threading import Thread
import spidev
import RPi.GPIO as GPIO
import time

# Initialize SPI for HB100
spi = spidev.SpiDev( )
spi.open(0, 0)
spi.max_speed_hz = 1350000

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)
# Set up the GPIO pin for the HB100 output
GPIO_PIN = 24
GPIO.setup(GPIO_PIN, GPIO.IN)

# Setup for OpenCV vehicle detection
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')


def read_adc(channel):
    adc = spi.xfer2([ 1, (8 + channel) << 4, 0 ])
    data = ((adc[ 1 ] & 3) << 8) + adc[ 2 ]
    return data


def detect_vehicle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return len(cars) > 0, image


def capture_from_camera(camera_num):
    camera = PiCamera(camera_num=camera_num)
    camera.resolution = (1920, 1080)
    rawCapture = PiRGBArray(camera, size=(1920, 1080))

    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            vehicle_detected, processed_image = detect_vehicle(image)

            if vehicle_detected:
                cv2.imshow(f'Camera {camera_num}', processed_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            rawCapture.truncate(0)

            # Break the loop if no vehicle is detected
            if not vehicle_detected:
                break

    finally:
        camera.close( )
        cv2.destroyAllWindows( )


def detect_motion_and_capture(threshold=2.5):
    while True:
        sensor_value = read_adc(0)  # Reading from channel 0
        voltage = sensor_value * (3.3 / 1023.0)
        print("Sensor Value: {}, Voltage: {:.2f}V".format(sensor_value, voltage))

        if voltage > threshold:
            print("Motion detected in blind spot!")
            thread1 = Thread(target=capture_from_camera, args=(0,))
            thread1.start( )
            thread1.join( )

            thread2 = Thread(target=capture_from_camera, args=(1,))
            thread2.start( )
            thread2.join( )
        else:
            print("No motion detected.")

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        detect_motion_and_capture( )
    except KeyboardInterrupt:
        print("Program stopped by User")
