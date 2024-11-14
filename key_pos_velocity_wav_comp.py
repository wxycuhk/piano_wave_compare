# Python code to generate the position and velocity information of piano keys
# Modify the serial port code to gain data from AT32
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pylab import mpl
import time
import threading
import struct
import logging 
import math
import keyboard
import serial
import os
import sounddevice as sd
import pandas as pd
import wave
import pyaudio
from scipy.fftpack import fft, ifft, irfft
import codecs

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
        self.piano_velocity_path = None
        self.piano_recording_file = None
        self.piano_velocity_file = None
        self.fps = 24

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

    def piano_velocity_write(self, path):
        '''Start write real-time recording data of piano key position and velocity'''
        self.piano_velocity = True
        self.piano_velocity_path = path
        self.piano_velocity_file = open(path, "w")
        logging.info("Velocity recording started")

    def piano_recording_read(self, path):
        '''Read real-time recording data of piano key position'''
        self.piano_read = True
        #self.piano_recording_path = path
        pf = open(path, "r")
        lines = pf.readlines()
        # The piano key values in a line are separated by a comma
        for line in lines:
            key_pos = line.split(",")
            self.key_pos = np.array([float(x) for x in key_pos[:-1]])
        logging.info("Recording read")

    def piano_recording_stop(self):
        '''Stop real-time recording of piano key position and velocity'''
        self.piano_recording = False
        self.piano_read = False
        self.piano_recording_file.close()
        self.piano_velocity_file.close()
        logging.info("Recording stopped")

# Calculate the velocity of the piano keys with data in key_pos file at different time points and save into a file
    def piano_velocity(self, key_pos, key_vel, time):
        '''Calculate the velocity of the piano keys with data in key_pos file at different time points and save into a file'''
        key_pos = np.array(key_pos)
        key_vel = np.array(key_vel)
        time = np.array(time)
        key_vel = np.diff(key_pos) / np.diff(time)
        return key_vel
    

    def piano_thread_func(self):
        while self.piano_active:
            if self.piano_connect is True:
                #self.prev_time = time.time()
                # Read the serial data from the piano, serial input is line with 5 integers separated by commas
                # Press 'p' to start record, 'r' to read record, 's' to stop record
                # Write time, key position to the file

                line = self.piano_serial.readline()
                #print(line)
                line = line.decode("utf-8")
                key_pos = line.split(",")
                if len(key_pos) == 6: # The last element besides the 5 integer is '\r\n'
                    try:
                        self.key_pos = np.array([float(x) for x in key_pos[:-1]])
                        if self.piano_recording:
                            line = str(time.time()) + "," + line +'\n'
                            print("Writing..." + line)
                            #print(self.key_pos)
                            self.piano_recording_file.write(line)
                            #print path of piano recording file
                            print(os.path.abspath(self.piano_recording_path))
                            #calculate the velocity of the piano keys by key_pos -key_pos_history/time_diff
                            #start calculating the velocity of the piano keys when the key_pos_history has more than 1 line
                            if len(self.key_pos_history) > 0:   
                                time_diff = 1
                                # Replace time_diff by line index diff
                                self.prev_time = time.time()
                                for i in range(5):
                                    self.key_vel[i] = (self.key_pos[i] - (float)(self.key_pos_history[-1][i]))/time_diff
                                    #print key_pos, key_pos_history, time_diff, key_vel
                                    #print(self.key_pos[i], self.key_pos_history[-1][i], time_diff, self.key_vel[i])
                                    if abs(self.key_vel[i]) > 50:
                                        self.key_vel[i] = 0
  
                                self.key_pos_history.append(key_pos)
                                line = str(time.time()) + ","
                                for i in range(5):
                                    line += str(self.key_vel[i]) + ","
                                line += '\n'
                                self.piano_velocity_file.write(line)
                                print(line)
                                # write the velocity of the piano keys to the file
                                print(os.path.abspath(self.piano_velocity_path))


                            else:
                                self.key_pos_history.append(key_pos)

                        if self.piano_read:
                            self.key_pos = np.array([float(x) for x in key_pos[:-1]])
                    except ValueError:
                        logging.error("Error Reading keys: " + str(key_pos))
                else:
                    logging.error("Error key length: " + str(len(key_pos)))
                    continue
            else:
                logging.error("Serial port not connected")

class Audiowave:

    def __init__(self):

        self.wavedata=[]
        self.wavewidth=2
        self.wavechannel=1
        self.framerate = 48000
        self.fps=24
        self.output_path = r'C:\Users\WANG Xingyu\Desktop\Parsons\piano_audio'
        self.filename = 'test.wav'
        self.filename1 = 'test1.wav'
        self.Timedata=[]
        self.nframes=0
        self.N=0
        self.data=[]
        self.dataall = []
        np.array(self.dataall)
        self.start_recording = False
        # plt.figure(figsize=(16, 8))
        self.fig, self.ax = plt.subplots(2,1,figsize=(8, 6))
        self.p=pyaudio.PyAudio()  # 实例化对象
        self.q=pyaudio.PyAudio()  # 实例化对象
        if self.wavewidth == 1:
            self.format= pyaudio.paInt8
        elif self.wavewidth == 2:
            self.format= pyaudio.paInt16
        elif self.wavewidth == 3:
            self.format= pyaudio.paInt24
        elif self.wavewidth == 4:
            self.format= pyaudio.paFloat32

        self.waveCHUNK = int(self.framerate / self.fps) # Frequency range of Single-sided spectrum
        # a=time.time()
        self.c = 0
        self.a = time.time()
        self.b = 0
        self.x=[]
        self.y=[]
        self.fft_int=[]
        np.array(self.x)
        np.array(self.y)
        self.sound_thread = None
        self.wf = None
        self.wf1 = None
        self.sound_active = False

    def open_stream(self):#打开录音流
        self.sound_active = True
        self.stream =self.p.open(format=self.format,
                        channels=self.wavechannel,
                        rate=self.framerate,
                        input=True,
                        frames_per_buffer=self.waveCHUNK)  #录音
        self.stream1 =self.q.open(format=self.format,
                        channels=self.wavechannel,
                        rate=self.framerate,
                        output=True,
                        frames_per_buffer=self.waveCHUNK) #播放
        if self.stream.is_active():
            logging.info("Recording channel is open")
            if self.sound_thread is None:
                self.sound_thread = threading.Thread(target=self.micdata)
                self.sound_thread.start()
        else:
            print("Recording channel activation failed")
        
    def wave_init(self):#初始化录音文件
        self.start_recording = True
        self.wf = wave.open(self.filename, 'w')
        self.wf.setnchannels(self.wavechannel)  # 声道设置
        self.wf.setsampwidth(self.p.get_sample_size(self.format))  # 采样位数设置
        self.wf.setframerate(self.framerate)
        self.wf1 = wave.open(self.filename1, 'w')
        self.wf1.setnchannels(self.wavechannel)  # 声道设置
        self.wf1.setsampwidth(self.p.get_sample_size(self.format))  # 采样位数设置
        self.wf1.setframerate(self.framerate)

    def wave_stop(self):#关闭录音文件
        self.start_recording = False
        self.wf.close()
        self.wf1.close()
        self.sound_active = False
        #self.stream.stop_stream()
        #self.stream.close()
        #self.stream1.stop_stream()
        #self.stream1.close()
        #self.p.terminate()
        #self.q.terminate()
    def to_fft(self,N,data):#转换为频域数据，并进行取半、归一化等处理
        # N=self.nframes #取样点数
        df = self.framerate / (N - 1)  #每个点分割的频率 如果你采样频率是4096，你FFT以后频谱是从-2048到+2048hz（4096除以2），然后你的1024个点均匀分布，相当于4个HZ你有一个点，那个复数的模就对应了频谱上的高度
        freq = [df * n for n in range(0, N)]

        wave_data2 = data[0:N]
        self.fft_int=np.fft.fft(wave_data2)
        # print(N, len(data),len(wave_data2))
        c = self.fft_int * 2 / N  #*2能量集中化  /N归一化
        d = int(len(c) / 2)  #对称取半
        freq=freq[:d - 1]
        fredata=abs(c[:d - 1])

        return freq,fredata

    def wavehex_to_DEC_n(self,wavedata,wavewidth,wavechannel):#录音存储数据十六进制数据转换十进制
        # print("#####################")
        Timedata=[]


        # print(type(self.Timedata))
        n = int(len(wavedata) / wavewidth)
        i = 0
        j = 0
        for i in range(0, n):
            b = 0
            for j in range(0, wavewidth):
                temp = wavedata[i * wavewidth:(i + 1) * wavewidth][j] * int(math.pow(2, 8 * j))
                b += temp
            if b > int(math.pow(2, 8 * wavewidth - 1)):
                b = b - int(math.pow(2, 8 * wavewidth))
            Timedata.append(b)
        Timedata = np.array(Timedata)
        # print(len(self.Timedata))
        Timedata.shape = -1, wavechannel
        Timedata = Timedata.T

        x = np.linspace(0, len(Timedata[0])-1, len(Timedata[0])) / self.framerate

        return x,Timedata

    def DEC_to_wavehex(self,DEC_data, wavewidth=2, wavechannel=1):
        wavewidth=self.wavewidth
        wavedata = ""
        for data in DEC_data:
            data = int(data)

            if data < 0:
                data += int(math.pow(2, 8 * wavewidth))

                # print(data)
            a = hex(data)[2:]
            a = a[::-1]
            while len(a) < 2 * wavewidth:
                a += "0"
            for i in range(0, wavewidth):
                # print(a[i * 2:2 * i + 2])
                b = r"\x" + a[i * 2:2 * i + 2]
                wavedata += b
            # wavedata.append(b)
        wavedata=bytes(wavedata, encoding="utf8")

        return codecs.escape_decode(wavedata, "hex-escape")[0]

    def micdata(self):#录音数据，存储，播放
        while self.sound_active:
            t00=time.time()
            data=self.stream.read(self.waveCHUNK) #录音


            self.data = data
            filter_data=self.filter(data,0,600)   ###这个要转成16进制进行播放

            t11=time.time()
            if self.wf is not None:
                self.stream1.write(data) #播放
                self.wf.writeframes(data)#写入存储
                self.wf1.writeframes(filter_data)#写入存储
            #return data,filter_data




        #return data

    def Dynamic_micwave_init(self):#动态显示图像-初始化图像

        self.ax[0].set_xlim(0, 1/self.fps)
        self.ax[1].set_xlim(0,2000)

        ln, = self.ax[0].plot([], [], animated=False)
        return ln,  # 返回曲线

    def Dynamic_micwave_update(self,n):#动态显示图像-更新图像


        x, y = self.wavehex_to_DEC_n(self.data, self.wavewidth, self.wavechannel)


        fre_x,fre_y=self.to_fft(self.waveCHUNK,y[0])

        ln, = self.ax[0].plot(x, y[0], "g-")
        # print("###############")
        # print(len(fre_y))
        ln1, = self.ax[1].plot(fre_x, fre_y, "g-")

        return ln,ln1,

    def Dynamic_micwave_run(self): #动态显示图像

        ani = FuncAnimation(self.fig, self.Dynamic_micwave_update,
                            interval=1000/self.fps, init_func=self.Dynamic_micwave_init, blit=True)
        plt.show()

    def filter(self,data,m,n):
        x,DEC_data=self.wavehex_to_DEC_n(data,self.wavewidth,self.wavechannel)
        self.to_fft(self.waveCHUNK, DEC_data[0])

        a=int(m*(self.waveCHUNK-1)/self.framerate)
        b=int(n*(self.waveCHUNK-1)/self.framerate)

        self.fft_int[a:b]=0
        self.fft_int[-b:-a]=0

        # print(self.fft_int[0:10])
        # print(self.fft_int)
        ifft_y = np.fft.ifft(self.fft_int).real
        # print(ifft_y[0:10])
        ifft_y=np.trunc(ifft_y)
        wave_data = self.DEC_to_wavehex(ifft_y)
        return wave_data

# Define a new function to plot the position and velocity of the piano keys according to the time with the data in the files
def plot_piano_data(path_pos, path_vel):
    '''Plot the position and velocity of the piano keys according to the time with the data in the csv files'''
    
    # Read the data from the csv files
    df_pos = pd.read_csv(path_pos)
    df_vel = pd.read_csv(path_vel)

    # Extract the time, position and velocity data
    time = df_pos.iloc[:, 0]
    key_pos = df_pos.iloc[:, 1:]
    key_vel = df_vel.iloc[:, 1:]

    # Plot the position of the piano keys
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    for i in range(5):
        axs[0].plot(time, key_pos.iloc[:, i], label='Key ' + str(i + 1))
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Position')
    axs[0].set_title('Position of Piano Keys Over Time')
    axs[0].legend()
    axs[0].grid(True)

    # Plot the velocity of the piano keys
    time = df_vel.iloc[:, 0]
    for i in range(5):
        axs[1].plot(time, key_vel.iloc[:, i], label='Key ' + str(i + 1))
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Velocity of Piano Keys Over Time')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot the wave file recorded IN AX[2]
    # Read the wave file
    wf = wave.open(r'C:\Users\WANG Xingyu\Desktop\Parsons\piano_audio\test.wav', 'rb')
    # Read the wave data
    nframes = wf.getnframes()
    data = wf.readframes(nframes)
    # Convert the wave data to integers
    wavedata = np.frombuffer(data, dtype=np.int16)
    # Plot the wave data
    # Adjust the layout
    plt.tight_layout()
    plt.show()


    axs[2].plot(wavedata)

a=Audiowave()



if __name__ == "__main__":
    p = piano(verbose = True)
    #wave = piano_wave()
    p.open_serial("COM5", 115200)
    a.open_stream()
   # wave.sound_channel_open()
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
        if keyboard.is_pressed("s"): # Press 's' to start sensor and sound recording simultaneously
            if not p.piano_recording:
                p.piano_recording_write("piano_recording.csv")
                p.piano_velocity_write("piano_velocity.csv")
                print("Sensor Recording started")
            if not a.start_recording:
                a.wave_init()
                print("Sound Recording started")
                a.Dynamic_micwave_run()
        if keyboard.is_pressed("q"):
            if p.piano_recording:
                p.piano_recording_stop()
                print("Recording stopped")
            if not p.piano_read:
                print("Recording initialize failed")
            if a.start_recording:
                a.wave_stop()
                print("Sound Recording stopped")
        if keyboard.is_pressed("r"):
            if not p.piano_read:
                p.piano_recording_read("piano_recording.csv")
                print("Reading starSed")
        if keyboard.is_pressed("e"):
            logging.info("Exiting recording")
            p.piano_active = False
            break

        time.sleep(0.005)
    p.piano_active = False
    plot_piano_data("piano_recording.csv", "piano_velocity.csv")
    print("End of program")



        

            