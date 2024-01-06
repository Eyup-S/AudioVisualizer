# # Apply bandpass filter
# def apply_stopband_filter(self, low, high):
#     def butter_bandstop_filter(lowcut, highcut, fs, order=5):
#         print("fs ", fs,"type: ", type(fs))
#         nyq = 0.5 * int(fs)
#         low = lowcut / nyq
#         high = highcut / nyq
#         b, a = butter(order, [low, high], btype='bandstop')
#         return b, a

#     b, a = butter_bandstop_filter(low, high, self.sample_rate)
#     self.data = lfilter(b, a, self.data)

# # Apply highpass filter
# def apply_highpass_filter(self, cutoff):
#     def butter_highpass(cutoff, fs, order=5):
#         nyq = 0.5 * fs
#         normal_cutoff = cutoff / nyq
#         b, a = butter(order, normal_cutoff, btype='high', analog=False)
#         return b, a

#     b, a = butter_highpass(cutoff, self.sample_rate)
#     self.data = lfilter(b, a, self.data)

# # Apply lowpass filter
# def apply_lowpass_filter(self, cutoff):
#     def butter_lowpass(cutoff, fs, order=5):
#         nyq = 0.5 * fs
#         normal_cutoff = cutoff / nyq
#         b, a = butter(order, normal_cutoff, btype='low', analog=False)
#         return b, a

#     b, a = butter_lowpass(cutoff, self.sample_rate)
#     self.data = lfilter(b, a, self.data)