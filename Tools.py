import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq, irfft
import numpy as np

import threading
import wave
import pyaudio
import pathlib
import time
import constants

class Tools:
        
    def __init__(self):
        self.playback_thread = None
        self.position = 0
        self.paused = True
        self.stop_flag = False
        self.band_list = constants.band_list
        self.gain_list = [0] * len(self.band_list)
        self.output_freq = None
        self.output_mag = None
        self.input_freq = None
        self.input_mag = None
    
    # Load the audio file as wav file
    def read_file(self, file_name):
        with wave.open(pathlib.Path(__file__).parent.resolve().joinpath('songs').joinpath(file_name).as_posix() , 'rb') as wf:
            audio_signal = wf.readframes(wf.getnframes())
            audio_signal = np.frombuffer(audio_signal, dtype=np.int16)
           
            self.data = audio_signal
            self.sample_rate = wf.getframerate()
            self.sample_width = wf.getsampwidth()
            self.channels = wf.getnchannels()
            self.nframes = wf.getnframes()
            self.duration = self.nframes / float(self.sample_rate)

        self.normalized_data = np.int16((self.data / self.data.max()) * 32767)

        
    # Apply RFFT to the audio signal, RFFT is used for computational efficiency
    def fourier_transform(self, normalized_data):
        fft_magnitude = rfft(normalized_data)
        fft_frequencies = rfftfreq(len(normalized_data), d=1/self.sample_rate)

        return fft_magnitude, fft_frequencies
    
    # Apply inverse RFFT
    def inverse_fourier_transform(self, fft):
        inverse_fourier = irfft(fft)
        max_val = 32767
        min_val = -32768
        inverse_fourier = np.clip(inverse_fourier, min_val, max_val)
        return np.int16(inverse_fourier * (32767 / inverse_fourier.max())) # Normalize the resulting audio signal

    # Apply equalizer to the signal 
    def equalizer(self, fft_mag, ftt_freq, bands_list, gain_list, save=False):
        assert len(bands_list) == len(gain_list), "Lengths of band_list and gain_list should be the same."
        scale_list = 10**(np.array(gain_list)/20)
        equalized_fft = np.copy(fft_mag)

        for (low, high), scale_factor in zip(bands_list, scale_list):
            idx_band = np.logical_and(ftt_freq > low, ftt_freq < high)
            equalized_fft[idx_band] = fft_mag[idx_band]* scale_factor

        return self.inverse_fourier_transform(equalized_fft), equalized_fft
    
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
        self.paused = False
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._play_audio, args=(buffer_size,))
            self.playback_thread.start()

    # Save audio file
    def save(self, audio_signal, name="mySong.wav"):
        with wave.open(pathlib.Path(__file__).parent.resolve().joinpath('songs').joinpath(name).as_posix(), 'wb') as wf:
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
            fft_mag, fft_freq = self.fourier_transform(self.normalized_data[self.position:end])
            self.input_mag = fft_mag
            self.input_freq = fft_freq
            equalized_data, self.output_mag = self.equalizer(fft_mag, fft_freq, self.band_list, self.gain_list)

            stream.write(equalized_data.tobytes())
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
    tool.read_file("Je te laisserai des mots.wav")
    # tool.read_file("./songs/audio.mp3")


    tool.save(tool.data)

    tool.play(128000)
    time.sleep(5)
    tool.gain_list = [12, 12, 0, 0, 0, 0]
    print("Bass boosting")
    tool.paused = True
    time.sleep(5)
    tool.play(12800)
    tool.gain_list = [0, 0, 12, 12, 0, 0]
    print("Mid boosting")

    time.sleep(5)
    tool.gain_list = [0, 0, 0, 0, 12, 12]
    print("High boosting")


    input("Press Enter to exit...")
    tool.stop()
