import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
from scipy.signal import butter, lfilter
import threading
import wave
import pyaudio

class Tools:
        
    def __init__(self):
        pass
    
    # Load the audio file as wav file
    def read_file(self, file_name):
        with wave.open(file_name, 'rb') as wf:
            audio_signal = wf.readframes(wf.getnframes())
            audio_signal = np.frombuffer(audio_signal, dtype=np.int16)
           
            self.data = audio_signal
            self.sample_rate = wf.getframerate()
            self.sample_width = wf.getsampwidth()
            self.channels = wf.getnchannels()
            self.nframes = wf.getnframes()
            self.duration = self.nframes / float(self.sample_rate)

        self.normalized_data = np.int16((self.data / self.data.max()) * 32767)
        self.playback_thread = None
        self.position = 0
        self.paused = True
        self.stop_flag = False
        print(self.data.shape)
        print(self.data.ndim)

    # Apply RFFT to the audio signal, RFFT is used for computational efficiency
    def fourier_transform(self):
        self.fft_magnitude = rfft(self.normalized_data)
        self.fft_frequencies = rfftfreq(len(self.normalized_data), d=1/self.sample_rate)
    
    # Apply RFFT to a specific range of an audio signal
    def fourier_transform_range(self, range_low, range_high):
        fft_magnitude = rfft(self.normalized_data[range_low:range_high])
        fft_frequencies = rfftfreq(len(fft_magnitude), d=1/int(self.sample_rate))
        return fft_magnitude, fft_frequencies
    
    # Apply inverse RFFT
    def inverse_fourier_transform(self):
        inverse_fourier = irfft(self.fft_magnitude)
        max_val = 32767
        min_val = -32768
        inverse_fourier = np.clip(inverse_fourier, min_val, max_val)
        self.inverse_fourier = np.int16(inverse_fourier * (32767 / inverse_fourier.max())) # Normalize the resulting audio signal

    # Apply inverse RFFT to a specific range of a RFFT magnitude signal   
    def inverse_fourier_transform_range(self, range_low, range_high):
        inverse_fourier = irfft(self.fft_magnitude[range_low:range_high])
        max_val = 32767
        min_val = -32768
        inverse_fourier = np.clip(inverse_fourier, min_val, max_val)
        return np.int16(inverse_fourier * (32767 / inverse_fourier.max())) # Normalize the resulting audio signal

    # Apply bandpass filter
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
    
    # Apply highpass filter
    def apply_highpass_filter(self, cutoff):
        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        b, a = butter_highpass(cutoff, self.sample_rate)
        self.data = lfilter(b, a, self.data)
    
    # Apply lowpass filter
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
        return int(self.sample_rate)
    
    # Plot the audio signal
    def plot(self,range_low,range_high):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(self.duration) * len(self.fft_frequencies))
        range_high = int( range_high / float(self.duration) * len(self.fft_frequencies))
        plt.plot(self.data[range_low:range_high])
        plt.title("Waveform of the Audio")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.show()
    
    # Plot the FFT of the signal
    def plot_fft(self,range_low,range_high):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(self.duration) * len(self.fft_frequencies))
        range_high = int( range_high / float(self.duration) * len(self.fft_frequencies))
        plt.plot(self.fft_frequencies[range_low:range_high], self.fft_magnitude[range_low:range_high])
        plt.title("Fourier Transform of the Audio")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()
    
    # Set and start the thread that will play the audio
    def play(self, buffer_size):
        if self.paused:
            self.paused = False
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._play_audio, args=(buffer_size,))
                self.playback_thread.start()

    # def pause(self):
    #     if not self.paused:
    #         self.paused = True
    #         sd.stop()

    # Save audio file
    def save(self, audio_signal, name="mySong.wav"):
        with wave.open(fr"songs\\reconstructed\\wave_{name}", 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_signal.tobytes())

    # Play the audio file
    def _play_audio(self, buffer_size):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(self.sample_width),
                            channels=self.channels,
                            rate=self.sample_rate,
                            output=True)
        
        while not self.paused and not self.stop_flag:
            end = min(self.position + buffer_size, len(self.normalized_data))
            stream.write(self.normalized_data[self.position:end].tobytes())
            self.position = end

            if end == len(self.normalized_data):
                break
            
        stream.stop_stream()
        stream.close()

        p.terminate()

    # Stop Condition
    def stop(self):
        self.stop_flag = True
        self.position = 0 


if __name__ == "__main__":
    print("Tools.py")
    tool = Tools()
    tool.read_file(r"D:\\Emir\\School\\Semester 6\\EE 473\\project\\audio\\Je te laisserai des mots.wav")
    #tool.read_file("./songs/audio.mp3")


    tool.save(tool.data)
    tool.play(512)

    
    # tool.play()
    # input("Press Enter to stop...")
    # tool.pause()
    # input("Press Enter to play again...")
    # tool.play()
    # input("Press Enter to pause...")
    # tool.pause()
    input("Press Enter to exit...")
    tool.stop()
