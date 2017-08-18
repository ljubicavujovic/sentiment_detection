import numpy as np
import sounddevice as sd


class Record:

    def __init__(self, duration, samplerate, channels):
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels

        sd.default.channels = channels
        sd.default.samplerate = self.samplerate
        print(sd.query_devices())

    def record_audio(self, filename):
        record = sd.rec(int(self.duration*self.samplerate), device=1)
        sd.wait()
        np.save(filename, record)

    def play_audio(self, filename):
        record = np.load(filename + '.npy')
        sd.play(record, blocking=True, device=3)

    def record_positive_audio(self, sample_number):
        for i in range(0, sample_number):
            filename = 'dataset/positive/audio_' + str(i)
            print('Say something positive:')
            self.record_audio(filename)
            print('Stop.')
            self.play_audio(filename)

    def record_negative_audio(self, sample_number):
        for i in range(0, sample_number):
            filename = 'dataset/negative/audio_' + str(i)
            print('Say something negative:')
            self.record_audio(filename)
            print('Stop.')
            self.play_audio(filename)


if __name__ == "__main__":
    record = Record(5, 40000, 2)
    record.record_positive_audio(10)


