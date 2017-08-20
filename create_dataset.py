import numpy as np
import sounddevice as sd


class Record:

    def __init__(self, duration, samplerate, channels):
        self.duration = duration
        self.samplerate = samplerate
        self.channels = channels

        sd.default.channels = channels
        sd.default.samplerate = self.samplerate

    def record_audio(self, filename):
        record = sd.rec(int(self.duration*self.samplerate), device=1)
        sd.wait()
        np.save(filename, record)

    def play_audio(self, filename):
        record = np.load(filename + '.npy')
        sd.play(record, blocking=True, device=3)

    def record_positive_audio(self, begin, sample_number):
        for i in range(begin, begin + sample_number):
            filename = 'dataset/positive/audio_' + str(i)
            print('Say something positive:')
            self.record_audio(filename)
            print('Stop.')
            self.play_audio(filename)

    def record_negative_audio(self, begin, sample_number):
        for i in range(begin, begin + sample_number):
            filename = 'dataset/negative/audio_' + str(i)
            print('Say : ')
            self.record_audio(filename)
            print('Stop.')
            self.play_audio(filename)

    def list_sentences(self, filenames):
        open('dataset/positive/positive_sentences.txt', 'w').close()
        open('dataset/negative/negative_sentences.txt', 'w').close()

        for filename in filenames:
            with open('dataset/' + filename, 'r') as file:
                for line in file:
                    if len(line) < 70:
                        if int(line[-2]) == 1:
                            with open('dataset/positive/positive_sentences.txt', 'a') as out:
                                out.write(line[:-2] + '\n')
                        else:
                            with open('dataset/negative/negative_sentences.txt', 'a') as out:
                                out.write(line[:-2] + '\n')


if __name__ == "__main__":
    record = Record(5, 40000, 2)
    #record.list_sentences(['imdb_labelled.txt', 'yelp_labelled.txt', 'amazon_cells_labelled.txt'])
    #record.record_negative_audio(59, 50)
    record.play_audio('dataset/negative/audio_98')


