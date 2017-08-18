import numpy as np
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

    def estimate_pitch_frequency(self, filename):
        audio = np.load(filename + '.npy')
        first_stream = audio[:, 0]
        second_stream = audio[:, 1]
        autos = []
        autos.append(self.estimate_autocorrelation(first_stream))
        autos.append(self.estimate_autocorrelation(second_stream))
        pitch = []
        offset = 20
        for auto in autos:
            maxi = 0
            index = 0
            for i in range(offset, 160):
                if auto[i] > maxi:
                    maxi = auto[i]
                    index = i
            pitch.append(float(self.samplerate/index))
        return pitch

    def intesity_mean(self, filename):
        audio = np.load(filename + '.npy')
        return audio.mean()

    def intesity_median(self, filename):
        audio = np.load(filename + '.npy')
        return audio.median()

    def intesity_variance(self, filename):
        audio = np.load(filename + '.npy')
        return audio.var()

    def plot_signal_and_autocorrelation(self, filename):
        audio = np.load(filename + '.npy')
        autocorr = self.estimate_autocorrelation(filename)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(audio)
        axarr[0].set_title('Audio signal')
        axarr[1].plot(autocorr)
        axarr[1].set_title('Autocorrelation for audio signal')

        plt.show()


if __name__ == "__main__":
    features = Features(40000)
    print features.intesity_variance('dataset/positive/audio_0')



