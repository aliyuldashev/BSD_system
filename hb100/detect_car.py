import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)
# Set up the GPIO pin for the HB100 output
GPIO_PIN = 24
GPIO.setup(GPIO_PIN, GPIO.IN)

try:
    while True:
        # Read the value from the HB100 sensor
        sensor_value = GPIO.input(GPIO_PIN)
        print("Sensor Value: ", sensor_value)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Measurement stopped by user")
    GPIO.cleanup()
