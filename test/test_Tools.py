from Tools import Tools
import constants
import time 

tools = Tools()
tools.read_file('./songs/interstellar.mp3')
current_second = 0
while (current_second < float(tools.duration)):
    fft_magnitude, fft_frequencies = tools.fourier_transform_range(int(current_second * tools.getSampleRate()), int((current_second + constants.timestep) * tools.getSampleRate()))
    time.sleep(constants.timestep)
    current_second += tools.getSampleRate() * constants.timestep

    for x in fft_frequencies:
        print(x)