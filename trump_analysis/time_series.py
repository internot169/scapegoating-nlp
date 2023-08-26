import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import vader_english
import nssm_english

# number of days to group for visualization
GROUPING_SIZE = 10

def group_by_time_series(df):
    for index, row in df.iterrows():
        df.loc[index, 'Time_Num'] = (float(row['Year']) * 365) + (float(row['Month']) * 30.4167) + float(row['Day'])
    
    df = df.sort_values(by=['Time_Num'])
    min = df['Time_Num'].iloc[0]
    df = df.assign(Time_Num = lambda x: x['Time_Num'] - min)

    max_lim = GROUPING_SIZE

    a_dict = {

    }
    b_dict = {

    }
    c_dict = {

    }
    s_dict = {

    }
    nrows = len(df)
    for index, row in df.iterrows():
        if row['Time_Num'] >= max_lim:
            max_lim += GROUPING_SIZE

        key = str(max_lim)

        if (key not in a_dict):
            a_dict[key] = [0,0,0]
        if row['A_Sentiment_Final'] == -1:
            a_dict[key][0] += 1
        elif row['A_Sentiment_Final'] == 0:
            a_dict[key][1] += 1
        elif row['A_Sentiment_Final'] == 1:
            a_dict[key][2] += 1
        
        if (key not in b_dict):
            b_dict[key] = [0,0,0]
        if row['B_Sentiment_Final'] == -1:
            b_dict[key][0] += 1
        elif row['B_Sentiment_Final'] == 0:
            b_dict[key][1] += 1
        elif row['B_Sentiment_Final'] == 1:
            b_dict[key][2] += 1

        if (key not in c_dict):
            c_dict[key] = [0,0,0]
        if row['C_Sentiment_Final'] == -1:
            c_dict[key][0] += 1
        elif row['C_Sentiment_Final'] == 0:
            c_dict[key][1] += 1
        elif row['C_Sentiment_Final'] == 1:
            c_dict[key][2] += 1

        if (key not in s_dict):
            s_dict[key] = [0,0,0]
        if row['Sentiment'] <= -0.05:
            s_dict[key][0] += 1
        elif row['Sentiment'] > -0.05 and row['Sentiment'] < 0.05:
            s_dict[key][1] += 1
        elif row['Sentiment'] >= 0.05:
            s_dict[key][2] += 1
            
    return (a_dict, b_dict, c_dict, s_dict)

def visualize_over_time(df, name):
    index = 0
    for dict in group_by_time_series(df):
        new_df = pandas.DataFrame.from_dict(dict).transpose()
        new_df.columns = ["Negative", "Neutral", "Positive"]
        new_df = new_df.rename_axis('Time_Num').reset_index()

        for i, row in new_df.iterrows():
            total = row['Negative'] + row['Neutral'] + row['Positive']
            new_df.loc[i, 'Negative'] = (new_df.loc[i, 'Negative'] / total) * 100
            new_df.loc[i, 'Neutral'] = (new_df.loc[i, 'Neutral'] / total) * 100
            new_df.loc[i, 'Positive'] = (new_df.loc[i, 'Positive'] / total) * 100

        sns.lineplot(data=new_df[['Negative', 'Neutral', 'Positive']])

        sentiment_type = ""
        if (index == 0):
            sentiment_type = "NSSM_A"
        elif (index == 1):
            sentiment_type = "NSSM_B"
        elif (index == 2):
            sentiment_type = "NSSM_C"
        elif (index == 3):
            sentiment_type = "Vader"
        
        plt.xlabel = "Time"
        plt.ylabel = "Sentiment (%)"
        plt.title(name + ' Sentiment Analysis (' + sentiment_type + ')')
        plt.show()
        plt.savefig(sentiment_type + '_' + name + '.png')

        index += 1
  
    

def analyze_with_nssm(src, name):
    df = pandas.read_csv(src, encoding='utf-8', engine='python', names=['Date', 'Year', 'Month', 'Day', 'Author', 'Title', 'Body', 'Link', 'Section', 'Publication'], on_bad_lines='skip')

    df['Sentiment'] = df['Body'].apply(vader.sentiment)
    df = nssm.apply_nssm(df)

    visualize_over_time(df, name)

#analyze_with_nssm('./data/reuters.csv', 'Reuters')
#analyze_with_nssm('./data/cnn.csv', 'CNN')
#analyze_with_nssm('./data/fox.csv', 'Fox News')
#analyze_with_nssm('./data/nytimes.csv', 'NY Times')