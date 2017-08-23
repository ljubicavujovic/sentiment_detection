from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset/dataset.csv', index_col=0)


def add_polarity_features(df):

    sid = SentimentIntensityAnalyzer()
    negative = np.zeros(400)
    neutral = np.zeros(400)
    positive = np.zeros(400)
    compound = np.zeros(400)
    with open('dataset/text.txt', 'r') as f:
        for i, line in enumerate(f):
            ss = sid.polarity_scores(line)
            negative[i] = ss['neg']
            neutral[i] = ss['neu']
            positive[i] = ss['pos']
            compound[i] = ss['compound']

    df['Negative'] = negative
    df['Neutral'] = neutral
    df['Positive'] = positive
    df['Compound'] = compound
    df.to_csv('dataset/dataset.csv')


def feature_correlation(df, features=None):

    if features is not None:
        df = df[features]
    correlation_matrix = df.corr()

    data = correlation_matrix.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = correlation_matrix.columns
    row_labels = correlation_matrix.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
    plt.savefig('graphs/correlation_heatmap.png')
    plt.close()
    return correlation_matrix


feature_correlation(df, list(df))

