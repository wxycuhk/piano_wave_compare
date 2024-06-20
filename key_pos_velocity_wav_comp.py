# Python code to generate the position and velocity information of piano keys
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import struct
import logging 
import math
import keyboard
import serial

class piano:

    def __init__(self, verbose = True):
        # self.key_pos = np.array([0, 0, 0, 0, 0])
        self.m_timer = int(time.time() * 1000)
        self.key_list = [0, 0, 0, 0, 0]
        self.key_pos = np.array([0, 0, 0, 0, 0])
        self.key_vel = np.array([0, 0, 0, 0, 0])
        self.key_pos_history = []
        self.key_vel_history = []

        self.piano_active = True
        self.piano_connect = False
        self.piano_serial = None
        self.piano_thread = None

        logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
        logging.info("Piano object created")
        self.prev_time = time.time()
        self.piano_recording = False
        self.piano_read = False
        self.piano_recording_path = None
        self.piano_recording_file = None

    def open_serial(self, port, baudrate):
        if self.piano_serial is not None:
            self.piano_serial.close()
        self.piano_serial = serial.Serial(port, baudrate)
        if self.piano_serial.is_open:
            logging.info("Serial port opened")
            self.piano_connect = True
            if self.piano_thread is None:
                self.piano_thread = threading.Thread(target=self.piano_thread_func)
                self.piano_thread.start()
        else:
            logging.error("Serial port open failed")
            self.piano_connect = False

        def piano_recording_write(self, path):
            '''Start write real-time recording data of piano key position and velocity'''
            self.piano_recording = True
            self.piano_recording_path = path
            self.piano_recording_file = open(path, "w")
            logging.info("Recording started")

        def piano_recording_read(self, path):
            '''Read real-time recording data of piano key position'''
            self.piano_read = True
            self.piano_recording_path = path
            self.piano_recording_file = open(path, "r")
            lines = self.piano_recording_file.readlines()
            # The piano key values in a line are separated by a comma
            for line in lines:
                key_pos = line.split(",")
                self.key_pos = np.array([float(x) for x in key_pos.split()])
            logging.info("Recording read")

        def piano_recording_stop(self):
            '''Stop real-time recording of piano key position and velocity'''
            self.piano_recording = False
            self.piano_read = False
            self.piano_recording_file.close()
            logging.info("Recording stopped")

        def piano_velocity_calculate(self, path_pos, path_vel):
            '''Calculate the velocity of piano keys from the recorded position data'''
            self.piano_read = True
            pos_file = open(path_pos, "r")
            vel_file = open(path_vel, "w")
            lines = pos_file.readlines()
            prev_time = 0
            prev_key_pos = np.array([0, 0, 0, 0, 0])
            for line in lines:
                key_pos = line.split(",")
                key_pos = np.array([float(x) for x in key_pos.split()])
                key_vel = (key_pos - prev_key_pos) / (time.time() - prev_time)
                prev_time = time.time()
                prev_key_pos = key_pos
                vel_file.write(str(key_vel) + "\n")
            pos_file.close()
            vel_file.close()
            logging.info("Velocity calculated")
            

        def piano_thread_func(self):
            while self.piano_active:
                if self.piano_connect is True:
                    self.prev_time = time.time()
                    # Read the serial data from the piano, serial input is line with 5 integers separated by commas
                    # Press 'p' to start record, 'r' to read record, 's' to stop record
                    # Write time, key position to the file

                    line = self.piano_serial.readline()
                    line = line.decode("utf-8")
                    key_pos = line.split(",")
                    if len(key_pos) == 5:
                        try:
                            self.key_pos = np.array([float(x) for x in key_pos])
                            if self.piano_recording:
                                line = str(time.time()) + "," + line
                                self.piano_recording_file.write(line)
                            if self.piano_read:
                                self.key_pos = np.array([float(x) for x in key_pos])
                        except ValueError:
                            logging.error("Error Reading keys: " + key_pos)
                    else:
                        logging.error("Error key length: " + len(key_pos))
                else:
                    logging.error("Serial port not connected")

if __name__ == "__main__":
    p = piano(verbose = True)
    p.open_serial("COM3", 115200)
    time.sleep(1)
    time_start = time.time()
    prev_time = time.time()
    duration = 5000

    # Press 'p' to start recording, 's' to stop recording, 'r' to read recording
    while(time.time() - time_start < duration):
        if time.time() - prev_time > 0.1:
            print("[Recording] " if p.piano_recording else "", end="")
            print("[Reading] " if p.piano_read else "", end="")
            print(p.key_pos)
            prev_time = time.time()
        if keyboard.is_pressed("p"):
            if not p.piano_recording:
                p.piano_recording_write("piano_recording.txt")
                print("Recording started")
        if keyboard.is_pressed("s"):
            if p.piano_recording:
                p.piano_recording_stop()
                print("Recording stopped")
        if keyboard.is_pressed("r"):
            if not p.piano_read:
                p.piano_recording_read("piano_recording.txt")
                print("Reading started")

        time.sleep(0.005)
    plt.plot(p.key_pos_history)
    plt.plot(p.key_vel_history)
    plt.show()
    print("End of program")



        

            

