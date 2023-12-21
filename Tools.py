import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
from scipy.signal import butter, lfilter
import sounddevice as sd
import threading

class Tools:
        
    def __init__(self):
        pass
    def read_file(self, file_name):
        if file_name.endswith(".wav"):
            # self.sample_rate, self.data = wavfile.read(file_name)
            # self.data = self.data[:,0]
            audio = AudioSegment.from_file(file_name,format="wav")
            if audio.channels > 1:
                audio = audio.set_channels(1)
            self.data = np.array(audio.get_array_of_samples())
            self.sample_rate = int(mediainfo(file_name)['sample_rate'])
            self.duration = mediainfo(file_name)['duration']

        elif file_name.endswith(".mp3"):
            audio = AudioSegment.from_file(file_name,format="mp3")
            if audio.channels > 1:
                audio = audio.set_channels(1)
            self.data = np.array(audio.get_array_of_samples())
            self.sample_rate = int(mediainfo(file_name)['sample_rate'])
            self.duration = mediainfo(file_name)['duration']
            print("duration: ", self.duration)
        else:
            print("File format not supported")
            return
        
        self.playback_thread = None
        self.position = 0
        self.paused = True
        self.stop_flag = False
        print(self.data.shape)

    def fourier_transform(self):
        fft_result = np.fft.fft(self.data)
        self.fft_magnitude = np.abs(fft_result)
        self.fft_frequencies = np.fft.fftfreq(len(fft_result), d=1/int(self.sample_rate))
    
    def fourier_transform_range(self, range_low, range_high):
        fft_result = np.fft.fft(self.data[range_low:range_high])
        fft_magnitude = np.abs(fft_result)
        fft_frequencies = np.fft.fftfreq(len(fft_result), d=1/int(self.sample_rate))
        return fft_magnitude, fft_frequencies
    
    def inverse_fourier_transform(self):
        self.inverse_fourier = np.fft.ifft(self.fft_magnitude)
        
    def inverse_fourier_transform_range(self, range_low, range_high):
        inverse_fourier = np.fft.ifft(self.fft_magnitude[range_low:range_high])
        return inverse_fourier

    def apply_stopband_filter(self, low, high):
        def butter_bandstop_filter(lowcut, highcut, fs, order=5):
            print("fs ", fs,"type: ", type(fs))
            nyq = 0.5 * int(fs)
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='bandstop')
            return b, a

        b, a = butter_bandstop_filter(low, high, self.sample_rate)
        self.data = lfilter(b, a, self.data)
    
    def apply_highpass_filter(self, cutoff):
        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        b, a = butter_highpass(cutoff, self.sample_rate)
        self.data = lfilter(b, a, self.data)
    
    def apply_lowpass_filter(self, cutoff):
        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        b, a = butter_lowpass(cutoff, self.sample_rate)
        self.data = lfilter(b, a, self.data)
    
    def getData(self):
        return self.data
    def getSampleRate(self):
        return self.sample_rate
    
    def plot(self,range_low,range_high):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(self.duration) * len(self.fft_frequencies))
        range_high = int( range_high / float(self.duration) * len(self.fft_frequencies))
        plt.plot(self.data[range_low:range_high])
        plt.title("Waveform of the Audio")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.show()
    
    def plot_fft(self,range_low,range_high):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(self.duration) * len(self.fft_frequencies))
        range_high = int( range_high / float(self.duration) * len(self.fft_frequencies))
        plt.plot(self.fft_frequencies[range_low:range_high], self.fft_magnitude[range_low:range_high])
        plt.title("Fourier Transform of the Audio")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()

    def saveFile(self, file_name):
        wavfile.write(file_name, self.sample_rate, self.data)
    
    def play(self):
        if self.paused:
            self.paused = False
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._play_audio)
                self.playback_thread.start()

    def pause(self):
        if not self.paused:
            self.paused = True
            sd.stop()

    def _play_audio(self):
        with sd.OutputStream(samplerate=self.sample_rate, channels=1) as stream:
            while not self.paused and not self.stop_flag:
                start = self.position
                end = start + int(self.sample_rate * 0.5)  # Play half a second at a time
                if end > len(self.data):
                    end = len(self.data)

                if self.data.ndim > 1:
                    chunk = self.data[start:end, :]
                else:
                    chunk = self.data[start:end]

                # Convert the chunk to float32
                chunk = chunk.astype(np.float32)

                stream.write(chunk)
                self.position = end
                if self.position >= len(self.data):
                    break  

    def stop(self):
        self.stop_flag = True
        self.pause()
        self.position = 0 


if __name__ == "__main__":
    print("Tools.py")
    tool = Tools()
    tool.read_file("audio.mp3")
    #tool.plot()
    #tool.plot_fft()
    #tool.apply_stopband_filter(2500, 3000)
    #tool.fourier_transform()
    #tool.plot_fft(50,60)

    tool.play()
    input("Press Enter to stop...")
    tool.pause()
    input("Press Enter to play again...")
    tool.play()
    input("Press Enter to pause...")
    tool.pause()
    input("Press Enter to exit...")
    tool.stop()
