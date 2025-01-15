# Python code to generate the position and velocity information of piano keys
# Modify the serial port code to gain data from AT32

#import sounddevice as sd
import numpy as np
import logging 
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pylab import mpl
import time
import threading
import struct
import math
import keyboard
import serial
import os
import csv
import pandas as pd
import wave
import pyaudio
from scipy.fftpack import fft, ifft, irfft
import codecs
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

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
                                    self.key_vel[i] = (float)((self.key_pos[i] - (float)(self.key_pos_history[-1][i]))/time_diff) * 10
                                    #print key_pos, key_pos_history, time_diff, key_vel
                                    #print(self.key_pos[i], self.key_pos_history[-1][i], time_diff, self.key_vel[i])
                                    #if abs(self.key_vel[i]) > 500:
                                     #   self.key_vel[i] = 0
  
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

# Establish a csv file to store the conversed wave data
class Audiowave:

    def __init__(self):

        self.wavedata=[]
        self.wavewidth=2
        self.wavechannel=1
        self.framerate = 48000
        self.fps=48
        self.output_path = r'C:\Users\wxycuhk\Desktop\piano_haptic'
        self.filename = 'test.wav'
        self.filename1 = 'test1.wav'
        self.Timedata=[]
        self.nframes=0
        self.N=0
        self.data=[]
        self.dataall = []
        np.array(self.dataall)
        self.start_recording = False
        self.stop_recording = False
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
            if self.start_recording:
                t00 = time.time()
                data=self.stream.read(self.waveCHUNK) #录音


                self.data = data
                filter_data=self.filter(data,0,600)   ###这个要转成16进制进行播放

                t11 = time.time()
                if self.wf is not None:
                    try:
                        #self.stream1.write(data) #播放
                        self.wf.writeframes(data)#写入存储
                        self.wf1.writeframes(filter_data)#写入存储
                    except:
                        logging.error("Recording file failed")
                else:
                    logging.info("Recording file done")
                    #break
                #return data,filter_data
            else:
                logging.info("Not recording")
                #break




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
def plot_piano_data(path_pos, path_vel, path_acc):
    '''Plot the position and velocity of the piano keys according to the time with the data in the csv files'''
    
    # Read the data from the csv files
    df_pos = pd.read_csv(path_pos)
    df_vel = pd.read_csv(path_vel)
    df_acc = pd.read_csv(path_acc)
    # Extract the time, position and velocity data
    time = df_pos.iloc[:, 0]
    key_pos = df_pos.iloc[:, 1:]
    key_vel = df_vel.iloc[:, 1:]
    key_acc = df_acc.iloc[:, 1:]

    # Plot the position of the piano keys
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
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
        axs[1].plot(time, key_vel)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('Velocity of Piano Keys Over Time')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot the acceleration of the piano keys, time is the same as the velocity
    time = df_acc.iloc[:, 0]
    print(len(time))
    axs[2].plot(time, key_acc, label='Key ' + str(1))
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Acceleration')
    axs[2].set_title('Acceleration of Piano Keys Over Time')
    axs[2].legend()
    axs[2].grid(True)


    # Plot the wave file recorded IN AX[2]
    # Read the wave file
    with wave.open(r'C:\Users\wxycuhk\Desktop\piano_haptic\test.wav', 'rb') as wf:
        # Read the wave data
        nframes = wf.getnframes() # Number of frames
        print('Frames:', nframes)
        n_channels = wf.getnchannels() # Number of channels
        print('Channels:', n_channels)
        frame_rate = wf.getframerate() # Frame rate
        print('Frame rate:', frame_rate)
        duration = nframes / frame_rate # Duration of the wave file
        print('Duration:', duration)
        data = wf.readframes(nframes) # Read the wave data
        print('Data:', len(data))

        # Convert the wave data to integers
        wavedata = np.frombuffer(data, dtype=np.int16)
        print('Wave data:', wavedata.shape)

        if n_channels > 1:
            wavedata = wavedata[::n_channels]

        # Generate time axis
        time = np.linspace(0, duration, num=len(wavedata))
    # Plot the wave data
    axs[3].plot(time, wavedata, color='blue')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_title('Wave Data of the Recorded Sound')

    # Adjust the layout
    plt.tight_layout()
    plt.show()

# Define a new function to calculate the RMS value of the wave per frame and store the data in a csv file
def Calculate_RMS_Per_Frame(signal, frame_size, frame_rate, output_csv):
    '''Calculate the RMS value of the wave per frame and store the data in a csv file'''
    rms_values = []
    time_stamps = []
    # Calculate the RMS value of the wave per frame
    for i in range(0, len(signal), frame_size):
        frame = signal[i:i + frame_size]
        if len(frame) == frame_size:
            rms = np.sqrt(np.mean(np.square(frame)))
            rms_values.append(rms)
            time_stamps.append(i / frame_rate)
    
    # Store the RMS values in a csv file
    with open(output_csv, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        # writer.writerow(['Time', 'RMS'])
        rms_values = np.array(rms_values)
        rms_values = 20 * np.log10(rms_values / 0.0002)
        for i in range(len(rms_values)):
            writer.writerow([time_stamps[i], rms_values[i]])

    # Plot the RMS values
    # Transfer rms_value to spl and plot
    plt.plot(time_stamps, rms_values - 50)
    plt.xlabel('Time (s)')
    plt.ylabel('SPL(dB)')
    plt.title('SPL Value of the Wave Per Frame')
    plt.show()

    return time_stamps, rms_values


def butter_filter(signal, sample_rate, cutoff, btype, order=5):
    '''Apply a Butterworth filter to the signal'''
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, signal)
    return y

def plot_frequency_spectrum(signal, sample_rate):
    '''Plot the frequency spectrum of the signal'''
    n = len(signal)
    k = np.arange(n)
    T = n / sample_rate
    frq = k / T
    frq = frq[range(n // 2)]
    Y = np.fft.fft(signal) / n
    Y = Y[range(n // 2)]
    plt.plot(frq, abs(Y))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum')
    plt.show()

a=Audiowave()

# 定义一个函数，将csv文件的每一列视为离散序列，获得离散序列的极值点与对应的line index
def get_peaks_from_csv(csv_file, column_index, h, d):
    '''Get the peaks of a discrete sequence from a csv file'''
    # Read the data from the csv file
    df = pd.read_csv(csv_file)

    # Check if the column index is valid
    if column_index < 0 or column_index >= df.shape[1]:
        raise ValueError(f"Colukmn index {column_index} out of range, valid range is 0 to {df.shape[1] - 1}")
    
    # Extract the data from the column
    data = df.iloc[:, column_index]

    # Find the peaks of the data
    peak_indices, _ = find_peaks(data, height = h, distance = d)

    # Find the valleys of the data
    valley_indices, _ = find_peaks(-data, height = h, distance = d)

    # Return the peak and valley indices
    return {
        "maxima": [data.iloc[i] for i in peak_indices],
        "minima": [data.iloc[i] for i in valley_indices]
    }

def acceleration_calculation(file_path_intput, file_path_output, column_index):
    df = pd.read_csv(file_path_intput)
    velocity = df.iloc[:, column_index].to_list()

    acceleration = [
        (velocity[i] - velocity[i - 1]) for i in range(1, len(velocity))
    ]
    time = df.iloc[:, 0].to_list()[1:]
    # Write 2 columns time and acceleration to a new csv file
    final_data = [{"time": time,"acceleration": acceleration} for time, acceleration in zip(time, acceleration)]
    results = pd.DataFrame(final_data)
    results.to_csv(file_path_output, header= None, index=False)
    print(f"Aceleration data saved to {file_path_output}")


if __name__ == "__main__":
    p = piano(verbose = True)
    #wave = piano_wave()
    p.open_serial("COM13", 115200)
    a.open_stream()
   # wave.sound_channel_open()
    time.sleep(1)
    time_start = time.time()
    prev_time = time.time()
    duration = 5000
    recorded = False

    # Press 'p' to start recording, 's' to stop recording, 'r' to read recording
    while(not recorded):
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
                # a.Dynamic_micwave_run() # close the dynamic wave plot if you want to continue
        if keyboard.is_pressed("q"):
            if p.piano_recording:
                p.piano_recording_stop()
                print("Recording stopped")
            if not p.piano_read:
                print("Recording initialize failed")
            if a.start_recording:
                a.wave_stop()
                print("Sound Recording stopped")
                recorded = True
            break
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
    acceleration_calculation("piano_velocity.csv", "piano_acceleration.csv", 1)
    plot_piano_data("piano_recording.csv", "piano_velocity.csv", "piano_acceleration.csv")
    with wave.open(r'C:\Users\wxycuhk\Desktop\piano_haptic\test.wav', 'rb') as wf:
        # Read the wave data
        nframes = wf.getnframes() # Number of frames
        n_channels = wf.getnchannels() # Number of channels
        frame_rate = wf.getframerate()  # Frame rate
        duration = nframes / frame_rate # Duration of the wave file
        print(f"Total frames: {nframes}, Channels: {n_channels}, Frame rate: {frame_rate}, Duration: {duration}")
        audio_data = wf.readframes(nframes) # Read the wave data

        # Convert the wave data to integers
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        if n_channels > 1:
            audio_data = audio_data[::n_channels]
            
    plot_frequency_spectrum(audio_data, frame_rate)
    Calculate_RMS_Per_Frame(audio_data, frame_rate, 1000, "RMS_data.csv")
    butter_filtered_data = butter_filter(audio_data, frame_rate, 5000, 'low')
    Calculate_RMS_Per_Frame(butter_filtered_data, frame_rate, 1000, "RMS_filtered_data.csv")
    # Establish new csv & wave files to store the audio & RMS data after filtering

    # Get the peaks and valleys of the velocity data & Rms data
    try:
        vel_peaks = get_peaks_from_csv("piano_velocity.csv", 1, 8000, 10)
        print("Peaks of the velocity data:")
        for point in vel_peaks["maxima"]:
            print(f"velocity peaks: {vel_peaks["maxima"]}")
        print("Valleys of the velocity data:")
        for point in vel_peaks["minima"]:
            press_vel = vel_peaks["minima"]
            print(f"velocity valleys: {vel_peaks["minima"]}")
    except Exception as e:
        print("Error getting peaks from velocity data:", e)

    try:
        rms_peaks = get_peaks_from_csv("RMS_filtered_data.csv", 1, 130, 1)
        print("Peaks of the RMS data:")
        press_sound = rms_peaks["maxima"]
        print(f"SPL peaks: {rms_peaks["maxima"]}")
        
        print("Valleys of the RMS data:")
        print(f"SPL valleys: {vel_peaks["minima"]}")
    except Exception as e:
        print("Error getting peaks from RMS data:", e)
    
    try:
        acc_peaks = get_peaks_from_csv("piano_acceleration.csv", 1, 4000, 10)
        print("Peaks of the acceleration data:")
        for point in acc_peaks["maxima"]:
            print(f"Acceleration peaks: {acc_peaks["maxima"]}")
        print("Valleys of the acceleration data:")
        for point in acc_peaks["minima"]:
            press_acc = acc_peaks["minima"]
            print(f"Acceleration valleys: {acc_peaks["minima"]}")
    except Exception as e:
        print("Error getting peaks from acceleration data:", e)

    # Compare the peaks of the velocity and RMS data
    if len(press_vel) == len(press_sound):
        combined_data = [{"velocity_value": press_vel, "RMS_value": press_sound} for press_vel, press_sound in zip(press_vel, press_sound)]
        df = pd.DataFrame(combined_data)
        if os.path.exists("RMS_vel_match_gp2.csv"):
            df.to_csv("RMS_vel_match_gp2.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("RMS_vel_match_gp2.csv", mode='w', header=False, index=False)
        print("Data saved to RMS_vel_match_gp2.csv")
        # 只取acc中序列号为偶数的值，这是琴键下压的加速度，奇数为负值，为琴键抬起的加速度
        press_acc = [press_acc[i] for i in range(0, len(press_acc), 2)]
        combined_kine = [{"acceleration_value": press_acc, "RMS_value": press_sound} for press_acc, press_sound in zip(press_acc, press_sound)]
        df = pd.DataFrame(combined_kine)
        if os.path.exists("RMS_acc_match_gp2.csv"):
            df.to_csv("RMS_acc_match_gp2.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("RMS_acc_match_gp2.csv", mode='w', header=False, index=False)
        print("Data saved to RMS_acc_match_gp2.csv")
    else:
        print("Number of valleys and peaks do not match")
    #plot_piano_data("piano_recording.csv", "piano_velocity.csv")
    print("End of program")



        

            