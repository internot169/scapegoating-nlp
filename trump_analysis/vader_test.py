from nltk import sent_tokenize
import nssm
import pandas
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

# number of days to group for visualization
GROUPING_SIZE = 5

def sentiment(text):
    vader = SentimentIntensityAnalyzer()

    sentences = sent_tokenize(text)
    sentences_count = len(sentences)
    sentiment_sum = 0

    for sentence in sentences:
        sentiment_sum += vader.polarity_scores(sentence)['compound']

    mean = sentiment_sum / sentences_count

    return mean

def group_by_time_series(df):
    for index, row in df.iterrows():
        df['Time_Num'] = float(row['Year']) / 365 + float(row['Month']) / 12 + float(row['Day'])
    df = df.sort_values(by=['Time_Num'])

    return df

def visualize_over_time(df):
    df = group_by_time_series(df)

    print(df.head())

    sns.lineplot(x="Time_Num", y="B_Sentiment", data=df)

def analyze_with_nssm(src):
    df = pandas.read_csv(src, encoding='utf-8', engine='python', names=['Date', 'Year', 'Month', 'Day', 'Author', 'Title', 'Body', 'Link', 'Section', 'Publication'], on_bad_lines='skip', nrows=1000)

    df['Sentiment'] = df['Body'].apply(sentiment)
    df = nssm.apply_nssm(df)

    visualize_over_time(df)

analyze_with_nssm('./data/reuters.csv')