import argparse
import os
import datetime
import time
import logging

import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def minutes_since_midnight(dt):
  return dt.hour * 60 + dt.minute

def record_and_save(filename, seconds=5):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    logging.info('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    logging.info('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def compute_spectrum_for_window(data, sample_rate, window_size):
    """
    Compute the spectrum for a given window of audio data
    
    Parameters:
    data: np.array of audio samples
    sample_rate: sampling rate in Hz
    window_size: size of the FFT window
    
    Returns:
    frequencies, magnitude spectrum
    """
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(data))
    windowed_data = data * window
    
    # Compute FFT
    fft_data = fft(windowed_data)
    n = len(fft_data)
    frequencies = np.linspace(0, sample_rate/2, n//2)
    
    # Compute magnitude spectrum
    magnitude = 2.0/n * np.abs(fft_data[0:n//2])
    
    return frequencies, magnitude

def plot_audio_spectrum(file_path, save_fig_path, save_npz_path=None, window_duration=1.0, avg_duration=5.0):
    """
    Load a WAV file and plot its waveform and averaged frequency spectrum
    
    Parameters:
    file_path (str): Path to the WAV file
    window_duration (float): Duration of each FFT window in seconds
    avg_duration (float): Duration over which to average spectra in seconds
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Read the wav file
    sample_rate, data = wavfile.read(file_path)
    
    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Calculate window parameters
    window_size = int(window_duration * sample_rate)
    n_windows = int(avg_duration / window_duration)
    
    # Ensure we have enough data
    if len(data) < window_size * n_windows:
        raise ValueError(f"Audio file too short. Need at least {avg_duration} seconds.")
    
    # Initialize arrays for averaging
    all_spectra = []
    
    # Compute spectrum for each window
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        frequencies, magnitude = compute_spectrum_for_window(window_data, sample_rate, window_size)
        all_spectra.append(magnitude)
    
    # Average the spectra
    avg_spectrum = np.mean(all_spectra, axis=0)
    
    # Create subplots
    plt.figure(figsize=(12, 12))
    
    # # Plot waveform
    # plt.subplot(2, 1, 1)
    # time = np.linspace(0, len(data)/sample_rate, len(data))
    # plt.plot(time, data)
    # plt.title('Audio Waveform')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    
    # Plot averaged spectrum
    #plt.subplot(2, 1, 2)
    plt.plot(frequencies, avg_spectrum, 'b-', linewidth=2)
    plt.title(f'Averaged Spectrum (over {avg_duration} seconds)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10, 4000)
    plt.plot([120, 120], [0.01, 10], '--', alpha=0.3)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=300)

    if save_npz_path:
        np.savez(save_npz_path, frequencies=frequencies, avg_spectrum=avg_spectrum)


parser = argparse.ArgumentParser(description="record 5s for every minute")
parser.add_argument("-o", "--output", type=str, help="output directory", required=True)
args = parser.parse_args()

wav_output_dir = os.path.join(args.output, "wav")
spectrum_output_dir = os.path.join(args.output, "spectrum")

os.makedirs(wav_output_dir, exist_ok=True)
#os.makedirs(spectrum_output_dir, exist_ok=True)

# prev_minute = None

# while True:
#     curr_dt = datetime.datetime.now()
#     curr_minute = minutes_since_midnight(curr_dt)

#     if curr_minute != prev_minute:
#         prev_minute = curr_minute

#         formatted_date = curr_dt.strftime("%Y-%m-%d/%H-%M-%S")
#         wav_output_file = os.path.join(wav_output_dir, formatted_date + ".wav")
#         os.makedirs(os.path.dirname(wav_output_file), exist_ok=True)

#         record_and_save(wav_output_file)
#     else:
#         time.sleep(1)

curr_dt = datetime.datetime.now()
formatted_date = curr_dt.strftime("%Y-%m-%d/%H-%M-%S")
wav_output_file = os.path.join(wav_output_dir, formatted_date + ".wav")
os.makedirs(os.path.dirname(wav_output_file), exist_ok=True)

record_and_save(wav_output_file)

spectrum_output_file = os.path.join(spectrum_output_dir, formatted_date + ".png")
npz_output_file = os.path.join(spectrum_output_dir, formatted_date + ".npz")
os.makedirs(os.path.dirname(spectrum_output_file), exist_ok=True)
plot_audio_spectrum(wav_output_file, spectrum_output_file, save_npz_path=npz_output_file,
                    avg_duration=4)