import numpy as np
import parabolic
from matplotlib.mlab import find
from helper_function import parabolic
import matplotlib.pyplot as plt
import python_speech_features as psf

class Features:

    def __init__(self, samplerate):
        self.samplerate = samplerate

    def estimate_autocorrelation(self, stream):
        n = len(stream)
        variance = stream.var()
        stream = stream - stream.mean()
        autocorrelation = np.correlate(stream, stream, 'full')[-n:]/(variance*np.arange(n, 0, -1))
        return autocorrelation

    def estimate_pitch_frequency(self, filename, window):
        audio = np.load(filename + '.npy')
        stream = audio[:, 0]
        #pitch = []
        # var = stream.var()
        # stream = (stream - stream.mean()) / var
        auto = self.estimate_autocorrelation(stream)
        d = np.diff(auto)
        start = find(d > 0)[0]
        peak = np.argmax(auto[start: start + int(window / 2)]) + start
        px, py = parabolic(auto, peak)
        pitch = (self.samplerate / px)

        return pitch

    def intesity_mean(self, filename):
        audio = np.load(filename + '.npy')
        return audio.mean()

    def intesity_variance(self, filename):
        audio = np.load(filename + '.npy')
        return audio.var()

    def plot_signal_and_autocorrelation(self, filename):
        audio = np.load(filename + '.npy')
        autocorr = self.estimate_autocorrelation(audio[:, 0])
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(audio)
        axarr[0].set_title('Audio signal')
        axarr[1].plot(autocorr)
        axarr[1].set_title('Autocorrelation for audio signal')

        plt.show()

    def get_mell_frequency_cepstral_ceofficients(self, filename):
        audio = np.load(filename + '.npy')[:, 0]
        var = audio.var()
        audio = (audio - audio.mean())/var
        mfcc = psf.mfcc(audio, self.samplerate, winstep=1, numcep=1, nfft=1024)
        return mfcc

    def get_spectral_centroid(self, filename):
        audio = np.load(filename + '.npy')[:, 0]
        var = audio.var()
        audio = (audio - audio.mean()) / var
        sc = psf.base.ssc(audio, self.samplerate, nfft=1024)
        return sc


if __name__ == "__main__":
    features = Features(40000)
    # print(features.get_mell_frequency_cepstral_ceofficients('dataset/positive/audio_90'))
    # print(features.get_mell_frequency_cepstral_ceofficients('dataset/negative/audio_90'))
    print(features.estimate_pitch_frequency('dataset/negative/audio_0', 199000))
    print(features.estimate_pitch_frequency('dataset/positive/audio_0', 199000))
    #print(features.estimate_pitch_frequency('dataset/test_sample', 80000))

