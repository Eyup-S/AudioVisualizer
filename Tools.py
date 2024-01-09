import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq, irfft

import io
import numpy as np
from PIL import Image

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
        self.nr_freq_bands = None
        self.nr_thresholds = None
    
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
    
    def spectral_gate(self,fft_signal,fft_frequencies):
        """
        :param freq_bands: List of frequency bands (each band is a tuple of start and end frequencies).
        :param thresholds: List of threshold percentages for each frequency band.
 
        """
        if (self.nr_freq_bands is None) or (self.nr_thresholds is None):
            return fft_signal
        
        for t in self.nr_thresholds:
            if t >= 99:
                t = 100

        # Number of samples in the FFT signal
        num_samples = len(fft_signal)

        # Frequency resolution
        freq_resolution = self.sample_rate / num_samples
        
        # Function to find the index range for a given frequency band
        for band, threshold in zip(self.nr_freq_bands, self.nr_thresholds):
            start_freq = band[0]
            end_freq = band[1]
            band_indices = np.where((fft_frequencies >= start_freq) & (fft_frequencies < end_freq))
            band_magnitudes = np.abs(fft_signal[band_indices])
            max_magnitude = np.max(band_magnitudes)
           
            threshold_value = max_magnitude * (threshold / 100)

            a = fft_signal[band_indices]
            a[a < threshold_value] = 0
            fft_signal[band_indices] = a

        return fft_signal
    
    # Plot the audio signal
    @staticmethod
    def plot(fft_mag,fft_frequencies,range_low,range_high, duration):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(duration) * len(fft_frequencies))
        range_high = int( range_high / float(duration) * len(fft_frequencies))
        plt.plot(fft_mag[range_low:range_high])
        plt.title("Waveform of the Audio")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Open the image and convert to a NumPy array
        img = Image.open(buf)
        img_array = np.asarray(img)

        # Close the buffer
        buf.close()

        return img_array

    @staticmethod
    def plot_fft(fft_magnitude,fft_frequencies,range_low,range_high, duration):
        """
        @param range_high: max frequency to plot
        """
        
        plt.figure(figsize=(12, 4))
        # range_low = int( range_low / float(duration) * len(fft_frequencies))
        # range_high = int( range_high / float(duration) * len(fft_frequencies))


        # plt.plot(filtered_frequencies, abs(filtered_magnitudes))
        plt.plot(fft_frequencies[fft_frequencies < range_high], abs(fft_magnitude[fft_frequencies < range_high]))
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Open the image and convert to a NumPy array
        img = Image.open(buf)
        img_array = np.asarray(img)

        # Close the buffer
        buf.close()

        return img_array
    
    # Set and start the thread that will play the audio
    def play(self, buffer_size):
        self.paused = False
        self.stop_flag = False
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
            end = min(self.position + int(buffer_size * 2), len(self.normalized_data))
            self.input_mag , self.input_freq = self.fourier_transform(self.normalized_data[self.position:end])
            equalized_data, self.output_mag = self.equalizer(self.input_mag, self.input_freq, self.band_list, self.gain_list)
            equalized_data = self.inverse_fourier_transform(self.spectral_gate(self.output_mag , self.input_freq))
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
    tool.gain_list = [0, 0, 12, 12, 0, 0,    0, 0, 0, 0, 0, 0]
    print("Mid boosting")

    time.sleep(5)
    tool.gain_list = [0, 0, 0, 0, 12, 12]
    print("High boosting")

    input("Press Enter to exit...")
    tool.stop()
