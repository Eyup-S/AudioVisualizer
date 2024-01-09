import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, sosfiltfilt
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft


def band_limited_noise(min_freq, max_freq, samples, sr):
    # Generate white noise
    noise = np.random.normal(0, 1, samples)
    # Create a butterworth bandpass filter
    sos = butter(10, [min_freq, max_freq], btype='bandpass', fs=sr, output='sos')
    # Apply the filter to the noise
    return sosfiltfilt(sos, noise)

def add_noise_to_audio(file_path, output_path, min_freq, max_freq, noise_amplitude, sampling_rate):
    # Read the audio file
    sr, data = wavfile.read(file_path)

    if len(data.shape) == 2:
        data = data.mean(axis=1)

    samples = len(data)
    print("samples", samples)

    # Generate band-limited noise
    noise = band_limited_noise(min_freq, max_freq, samples, sr) * noise_amplitude

    # If stereo, duplicate the noise
    if len(data.shape) == 2:
        noise = np.repeat(noise[:, np.newaxis], 2, axis=1)

    # Add noise to the signal
    noisy_signal = data + noise

    # Ensure the signal is within valid range
    noisy_signal = np.clip(noisy_signal, -32768, 32767)

    
    fft_magnitude_noise = rfft(noise if len(data.shape) == 1 else noise[:, 0])
    fft_frequencies_noise = rfftfreq(len(noise), d=1/sampling_rate)

    #take fourier transform of the noisy_signal
    fft_magnitude = rfft(noisy_signal if len(data.shape) == 1 else noisy_signal[:, 0])
    fft_frequencies = rfftfreq(len(noisy_signal), d=1/sampling_rate)

    #plot the noise
    # plt.title("Waveform of the Noise")
    # plt.xlabel("Sample Number")
    # plt.ylabel("Amplitude")
    # plt.plot(noise)
    # plt.show()

    plt.title("Fourier Transform of the Noise")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.plot(fft_frequencies_noise, np.abs(fft_magnitude_noise))
    plt.show()
    
    
    #plot the fourier transform of the noise
    plt.title("Fourier Transform of the Noisy signal")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.plot(fft_frequencies, np.abs(fft_magnitude))
    plt.savefig('plot.png')
    plt.show()

    print(len(fft_frequencies), len(fft_magnitude))
    print(noisy_signal.shape)

    #plot the noisy_signal
    plt.title("Waveform of the Noisy Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.plot(noisy_signal)
    plt.show()
    

def add_noise_from_mat_file(mat_path, song_path, output_path):
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample
    from pydub import AudioSegment

    # Load the .mat audio file
    mat_audio = loadmat(mat_path)['data']

    # Resample to 44.1 kHz
    original_sample_rate = 19980  # Adjust to your original sample rate
    resampled_audio = resample(mat_audio, int(len(mat_audio) * 48000 / original_sample_rate))

    # Load the song in WAV format
    song = AudioSegment.from_wav(song_path)

    # Convert the resampled audio to a format compatible with pydub (16-bit PCM)
    resampled_audio = resampled_audio.astype(np.int16)

    # Create an AudioSegment from the resampled audio
    resampled_audio_segment = AudioSegment(
        resampled_audio.tobytes(),
        frame_rate=48000,
        sample_width=resampled_audio.dtype.itemsize,
        channels=1  # Adjust if the resampled audio is stereo
    )

    # Add the resampled audio to the song
    output_song = song.overlay(resampled_audio_segment)

    # Export the final song with added audio
    output_song.export('noisy_song', format='wav')

def mat_to_wav(mat_file_path, wav_file_path, sample_rate):
    import scipy.io
    import scipy.io.wavfile as wav

    def mat_to_wav2(mat_file_path, wav_file_path, sample_rate):
        """
        Convert an audio signal from a .mat file to a WAV file.

        :param mat_file_path: Path to the .mat file containing the audio signal.
        :param wav_file_path: Path where the WAV file will be saved.
        :param sample_rate: The sample rate to use for the WAV file.
        """
        # Load the .mat file
        mat_data = scipy.io.loadmat(mat_file_path)

        # Assuming the audio data is stored in the first variable in the .mat file
        audio_key = list(mat_data.keys())[3]  # The 4th key (index 3) is typically the first variable
        audio_data = mat_data[audio_key].flatten()

        # Save the audio data to a WAV file
        wav.write(wav_file_path, sample_rate, audio_data)
    mat_to_wav2(mat_file_path, wav_file_path, sample_rate)

def mix_and_save_wav(song_file_path, noise_file_path, output_file_path):
    import scipy.io
    import scipy.io.wavfile as wav
    """
    Mix a song with a noise and save the result as a WAV file.

    :param song_file_path: Path to the song file (WAV format).
    :param noise_file_path: Path to the noise file (WAV format).
    :param output_file_path: Path where the mixed WAV file will be saved.
    """
    # Load the song and noise files
    song_sample_rate, song_data = wav.read(song_file_path)
    noise_sample_rate, noise_data = wav.read(noise_file_path)

    # Ensure the sample rates are the same
    if song_sample_rate != noise_sample_rate:
        raise ValueError("Sample rates of the song and noise must match")

    # Handle stereo/mono formats
    if len(song_data.shape) == 2:  # Stereo song
        if len(noise_data.shape) == 1:  # Mono noise
            # Duplicate the noise data to make it stereo
            noise_data = np.tile(noise_data[:, np.newaxis], (1, 2))
    
    # If the song is mono and the noise is stereo, convert noise to mono
    elif len(song_data.shape) == 1 and len(noise_data.shape) == 2:
        noise_data = np.mean(noise_data, axis=1)

    # Match the length of the song and noise
    min_length = min(len(song_data), len(noise_data))
    song_data = song_data[:min_length]
    noise_data = noise_data[:min_length]

    # Mix the song and the noise
    mixed_signal = song_data + noise_data

    # Normalize the mixed signal to avoid clipping
    max_val = np.max(np.abs(mixed_signal))
    if max_val > 1:
        mixed_signal = mixed_signal / max_val

    # Save the mixed signal as a WAV file
    wav.write(output_file_path, song_sample_rate, mixed_signal.astype(np.int16))

def read_and_plot(path):
    import scipy.io.wavfile as wav
    import scipy.fft
    #read wav file and take fourier transform
    sample_rate, data = wav.read(path)

    # If stereo, just use one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Plot the waveform
    plt.figure(figsize=(12, 6))

    # Time axis in seconds
    time_axis = np.arange(len(data)) / sample_rate

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, data)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot the Fourier Transform
    fft_data = scipy.fft.fft(data)
    fft_freq = scipy.fft.fftfreq(len(fft_data), 1 / sample_rate)
    print(fft_freq)
    print(fft_data)

    plt.subplot(2, 1, 2)
    plt.plot(fft_freq, np.abs(fft_data))
    plt.title("Fourier Transform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


read_and_plot("songs/cendere_noise.wav")


