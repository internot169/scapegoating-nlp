from nltk import pos_tag, word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas

def sentiment(text):
    vader = SentimentIntensityAnalyzer()

    sentences = sent_tokenize(text)
    sentences_count = len(sentences)
    sentiment_sum = 0

    for sentence in sentences:
        sentiment_sum += vader.polarity_scores(sentence)['compound']

    mean = sentiment_sum / sentences_count

    return mean

def get_naes(df):
    df = df.reset_index()
    naes_dict = {}

    for index, row in df.iterrows():
        text = word_tokenize(row['Body'])
        tags = pos_tag(text)
        for tag in tags:
            if (tag[1] == 'NN' or tag[1] == 'NNS' or tag[1] == 'NNP' or tag[1] == 'NNPS'):
                array = naes_dict[tag[1]]
                array[1] += 1
                if (row['Sentiment'] <= -0.05):
                    array[2] += 1

df = pandas.read_csv('./data/reuters.csv', encoding='utf-8', engine='python', names=['Date', 'Year', 'Month', 'Day', 'Author', 'Title', 'Body', 'Link', 'Section', 'Publication'], error_bad_lines=False, nrows=1000)
df = df.assign(Body = lambda x: str(x['Body']))
df = df.assign(Sentiment = lambda x: sentiment(x['Body']))

get_naes(df)