import cv2
import base64
import requests
import json
import logging
import time
import sys
from threading import Thread
from contextlib import contextmanager
from djitellopy import Tello

# logging configuration
logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-10s) %(message)s')

@contextmanager
def tello_connection():
    tello = Tello()
    tello.connect()
    tello.streamon()
    try:
        yield tello
    finally:
        tello.streamoff()
        tello.end()

def move_drone(tello):
    tello.takeoff()
    time.sleep(2)
    tello.move_forward(20)
    time.sleep(2)
    for _ in range(4):
        tello.rotate_clockwise(90)
        wait_for_ok(tello)
        time.sleep(2)
    tello.move_back(100)
    time.sleep(2)
    wait_for_ok(tello)
    tello.land()

# Wait for the 'ok' response from a move command
def wait_for_ok(tello):
    while True:
        response = tello.get_frame_read().get_bounding_box()
        if response:
            # Move command completed, 'ok' received
            break
        time.sleep(0.1)  # Sleep for a short interval to avoid busy loop

def convert_to_base64(frame):
    retval, buffer = cv2.imencode('.jpg', frame)
    encoded_data = base64.b64encode(buffer)
    return encoded_data.decode('utf-8')

def post_base64_image_to_api(image):
    url = "http://127.0.0.1:5000/infer"
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(image)
    response = requests.post(url, headers=headers, data=data)
    print(response)

class CameraThread(Thread):
    def __init__(self, thread_id, name, delay, counter, tello):
        super().__init__()
        self.thread_id = thread_id
        self.name = name
        self.delay = delay
        self.counter = counter
        self.tello = tello

    def run(self):
        while self.counter:
            try:
                frame = self.tello.get_frame_read().frame
                encoded_data = convert_to_base64(frame)
                post_base64_image_to_api({'binary': "data:image/jpeg;base64," + encoded_data})
                time.sleep(self.delay)
                self.counter -= 1
            except Exception as e:
                print(e)

def main():
    with tello_connection() as tello:
        camera_thread = CameraThread(1, "camera_thread", 0.2, 400, tello)
        camera_thread.daemon = True
        movement_thread = Thread(target=move_drone, args=(tello,))

        camera_thread.start()
        movement_thread.start()

        camera_thread.join()
        movement_thread.join()

        logging.info("Drone landed OK")
        sys.exit()

if __name__ == '__main__':
    main()
