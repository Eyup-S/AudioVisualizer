import Tools
import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq, irfft
import numpy as np

import threading
import wave
import pyaudio
import pathlib
import time
import constants

class NoiseReduction:
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

    def fourier_transform(self, normalized_data):
        fft_magnitude = rfft(normalized_data)
        fft_frequencies = rfftfreq(len(normalized_data), d=1/self.sample_rate)

        return fft_magnitude, fft_frequencies

    def inverse_fourier_transform(self, fft):
        inverse_fourier = irfft(fft)
        max_val = 32767
        min_val = -32768
        inverse_fourier = np.clip(inverse_fourier, min_val, max_val)
        return np.int16(inverse_fourier * (32767 / inverse_fourier.max())) # Normalize the resulting audio signal

    def spectral_gate(self,fft_signal,fft_frequencies, freq_bands, thresholds):
        """
        :param freq_bands: List of frequency bands (each band is a tuple of start and end frequencies).
        :param thresholds: List of threshold percentages for each frequency band.

        """
        # Number of samples in the FFT signal
        num_samples = len(fft_signal)

        # Frequency resolution
        freq_resolution = self.sample_rate / num_samples
        
        # Function to find the index range for a given frequency band
        for band, threshold in zip(freq_bands, thresholds):
            start_freq = band[0]
            end_freq = band[1]
            band_indices = np.where((fft_frequencies >= start_freq) & (fft_frequencies < end_freq))
            print("band_indices: ", band_indices)
            band_magnitudes = np.abs(fft_signal[band_indices])
            mean_magnitude = np.max(band_magnitudes)
            print("mean: ", mean_magnitude)
            print("max: ", np.max(band_magnitudes))
            threshold_value = mean_magnitude * (threshold / 100)

            a = fft_signal[band_indices]
            a[a < threshold_value] = 0
            fft_signal[band_indices] = a

        return fft_signal
    
    def plot(self,fft_mag,fft_frequencies,range_low,range_high):
        plt.figure(figsize=(12, 4))
        range_low = int( range_low / float(self.duration) * len(fft_frequencies))
        range_high = int( range_high / float(self.duration) * len(fft_frequencies))
        plt.plot(fft_mag[range_low:range_high])
        plt.title("Waveform of the Audio")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_fft(self,fft_magnitude,fft_frequencies,range_low,range_high):
        plt.figure(figsize=(12, 4))
        
        plt.plot(fft_frequencies[fft_frequencies < range_high], abs(fft_magnitude[fft_frequencies < range_high]))
        plt.title("Fourier Transform of the Audio")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()
    
    def play(self, buffer_size):
        self.paused = False
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._play_audio, args=(buffer_size,))
            self.playback_thread.start()
    
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

    def equalizer(self, fft_mag, ftt_freq, bands_list, gain_list, save=False):
        assert len(bands_list) == len(gain_list), "Lengths of band_list and gain_list should be the same."
        scale_list = 10**(np.array(gain_list)/20)
        equalized_fft = np.copy(fft_mag)

        for (low, high), scale_factor in zip(bands_list, scale_list):
            idx_band = np.logical_and(ftt_freq > low, ftt_freq < high)
            equalized_fft[idx_band] = fft_mag[idx_band]* scale_factor
        
        return self.inverse_fourier_transform(equalized_fft), equalized_fft

if __name__ == "__main__":
    nr = NoiseReduction()

    nr.read_file('Je te laisserai des mots.wav')
    nr.input_mag, nr.input_freq = nr.fourier_transform(nr.normalized_data)
    nr.plot_fft(nr.input_mag,nr.input_freq,0,500)
    # nr.plot(nr.normalized_data,nr.input_freq,0,50)
    # nr.plot_fft(nr.input_mag,nr.input_freq,0,50)
    freq_bands = [(200, 750), (600, 1000), (1000, 4000)] 
    thresholds = [5, 30, 50]

    gated_fft = nr.spectral_gate(nr.input_mag,nr.input_freq,freq_bands,thresholds)
    print("Gated...")
    nr.plot_fft(gated_fft,nr.input_freq,0,500)
    gated_signal = nr.inverse_fourier_transform(gated_fft)
    # nr.plot(gated_signal,nr.input_freq,0,50)
    nr.normalized_data = np.int16((gated_signal / gated_signal.max()) * 32767)
    nr.gain_list = [0, 0, 12, 12, 0, 0, 0, 0, 0, 0, 0, 0]
    nr.play(128000)

    input("Press Enter to exit...")
    nr.stop()