import pyaudio

# 获取指定设备的详细信息
def get_device_info_by_index(index):
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(index)
    p.terminate()
    return device_info

# 获取默认输入设备的详细信息
def get_default_input_device_info():
    p = pyaudio.PyAudio()
    default_input_info = p.get_default_input_device_info()
    p.terminate()
    return default_input_info

# 获取默认输出设备的详细信息
def get_default_output_device_info():
    p = pyaudio.PyAudio()
    default_output_info = p.get_default_output_device_info()
    p.terminate()
    return default_output_info

# 获取计算机上可用音频设备的数量
def get_device_count():
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    p.terminate()
    return device_count

# 示例用法
index = 0
print("可用音频设备数量：", get_device_count())
print("设备{}的信息：{}".format(index, get_device_info_by_index(index)))
print("默认录音设备的信息：", get_default_input_device_info())
print("默认播放设备的信息：", get_default_output_device_info())
