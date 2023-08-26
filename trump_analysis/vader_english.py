from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize

# ALSO TRY WEIGHTED MEAN
def sentiment(text):
    vader = SentimentIntensityAnalyzer()

    sentences = sent_tokenize(text)
    sentences_count = len(sentences)
    sentiment_sum = 0

    for sentence in sentences:
        # Explain the compound normalization thing in the paper
        sentiment_sum += vader.polarity_scores(sentence)['compound']

    mean = sentiment_sum / sentences_count

    return mean