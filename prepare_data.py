import create_features
import numpy as np
import pandas as pd


df = pd.read_csv('dataset/dataset.csv', index_col=0)
features = create_features.Features(40000)


def append_pitch(df, size):
    pitch = np.zeros(2*size)
    for i in range(0, size):
        pitch[i] = features.estimate_pitch_frequency('dataset/positive/audio_' + str(i), 199000)
        pitch[size + i] = features.estimate_pitch_frequency('dataset/negative/audio_' + str(i), 199000)
    print(pitch)
    df['Pitch'] = pitch
    df.to_csv('dataset/dataset.csv')
    print(df.head())


def append_mell_cc(df, size):
    mfcc_0 = np.zeros(size * 2)
    mfcc_1 = np.zeros(size * 2)
    mfcc_2 = np.zeros(size * 2)
    mfcc_3 = np.zeros(size * 2)
    mfcc_4 = np.zeros(size * 2)
    for i in range(size):
        pos = features.get_mell_frequency_cepstral_ceofficients('dataset/positive/audio_' + str(i))
        neg = features.get_mell_frequency_cepstral_ceofficients('dataset/negative/audio_' + str(i))
        mfcc_0[i] = pos[0]
        mfcc_0[size + i] = neg[0]
        mfcc_1[i] = pos[1]
        mfcc_1[size + i] = neg[1]
        mfcc_2[i] = pos[2]
        mfcc_2[size + i] = neg[2]
        mfcc_3[i] = pos[3]
        mfcc_3[size + i] = neg[3]
        mfcc_4[i] = pos[4]
        mfcc_4[size + i] = neg[4]

    df['Mfcc_0'] = mfcc_0
    df['Mfcc_1'] = mfcc_1
    df['Mfcc_3'] = mfcc_2
    df['Mfcc_3'] = mfcc_3
    df['Mfcc_4'] = mfcc_4

    df.to_csv('dataset/dataset.csv')


def append_intesity_features(df, size):
    mean = np.zeros(2 * size)
    var = np.zeros(2 * size)
    for i in range(size):
        mean[i] = features.intesity_mean('dataset/positive/audio_' + str(i))
        mean[size + i] = features.intesity_mean('dataset/negative/audio_' + str(i))
        var[i] = features.intesity_variance('dataset/positive/audio_' + str(i))
        var[size + i] = features.intesity_variance('dataset/negative/audio_' + str(i))
    df['Mean'] = mean
    df['Variance'] = var
    df.to_csv('dataset/dataset.csv')

    print(df.head())


if __name__ == "__main__":
    append_pitch(df, 200)


