from nltk import sent_tokenize
import nssm
import pandas
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

# number of days to group for visualization
GROUPING_SIZE = 30.4167

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
        df.loc[index, 'Time_Num'] = (float(row['Year']) * 365) + (float(row['Month']) * 30.4167) + float(row['Day'])
    
    df = df.sort_values(by=['Time_Num'])
    min = df['Time_Num'].iloc[0]
    df = df.assign(Time_Num = lambda x: x['Time_Num'] - min)

    max_lim = GROUPING_SIZE

    cleaned_dict = {

    }

    for index, row in df.iterrows():
        if row['Time_Num'] >= max_lim:
            max_lim += GROUPING_SIZE

        if str(max_lim) not in cleaned_dict:
            cleaned_dict[str(max_lim)] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        if row['A_Sentiment_Final'] == -1:
            cleaned_dict[str(max_lim)][0] += 1
        elif row['A_Sentiment_Final'] == 0:
            cleaned_dict[str(max_lim)][1] += 1
        elif row['A_Sentiment_Final'] == 1:
            cleaned_dict[str(max_lim)][2] += 1
        
        if row['B_Sentiment_Final'] == -1:
            cleaned_dict[str(max_lim)][3] += 1
        elif row['B_Sentiment_Final'] == 0:
            cleaned_dict[str(max_lim)][4] += 1
        elif row['B_Sentiment_Final'] == 1:
            cleaned_dict[str(max_lim)][5] += 1
        
        if row['C_Sentiment_Final'] == -1:
            cleaned_dict[str(max_lim)][6] += 1
        elif row['C_Sentiment_Final'] == 0:
            cleaned_dict[str(max_lim)][7] += 1
        elif row['C_Sentiment_Final'] == 1:
            cleaned_dict[str(max_lim)][8] += 1
        
        if row['Sentiment'] <= -0.05:
            cleaned_dict[str(max_lim)][6] += 1
        elif row['Sentiment'] > -0.05 and row['Sentiment'] < 0.05:
            cleaned_dict[str(max_lim)][7] += 1
        elif row['Sentiment'] >= 0.05:
            cleaned_dict[str(max_lim)][8] += 1
            
    return pandas.DataFrame.from_dict(cleaned_dict).transpose()

def visualize_over_time(df):
    new_df = group_by_time_series(df)
    new_df.set_axis(["A_Sentiment_Negative", "A_Sentiment_Neutral", "A_Sentiment_Positive", "B_Sentiment_Negative", "B_Sentiment_Neutral", "B_Sentiment_Positive", "C_Sentiment_Negative", "C_Sentiment_Neutral", "C_Sentiment_Positive"], axis=1,inplace=True)

    print(df.head())
    print(new_df.head())
    #data_preproc = pandas.DataFrame({'Year': df['Time_Num'], 'A': df['A_Sentiment'], 'B': df['B_Sentiment'], 'C': df['C_Sentiment']})
    #sns.lineplot(data=data_preproc[['A', 'B', 'C']])
    #plt.show()

def analyze_with_nssm(src):
    df = pandas.read_csv(src, encoding='utf-8', engine='python', names=['Date', 'Year', 'Month', 'Day', 'Author', 'Title', 'Body', 'Link', 'Section', 'Publication'], on_bad_lines='skip', nrows=1000)

    df['Sentiment'] = df['Body'].apply(sentiment)
    df = nssm.apply_nssm(df)

    visualize_over_time(df)

analyze_with_nssm('./data/reuters.csv')