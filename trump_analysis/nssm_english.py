from nltk import pos_tag, word_tokenize

P_THRESHOLD = 0.5
BETA = 0.9
ALPHA = 1 - BETA

# finding NAEs

def get_potential_naes(df):
    df = df.reset_index()
    potential_naes_dict = {}

    for index, row in df.iterrows():
        text = word_tokenize(row['Body'])
        tags = pos_tag(text)
        for tag in tags:
            if (tag[1] == 'NN' or tag[1] == 'NNS' or tag[1] == 'NNP' or tag[1] == 'NNPS'):
                word = tag[0]
                
                if word not in potential_naes_dict:
                    potential_naes_dict[word] = [0, 0]
                
                values = potential_naes_dict[word]
                values[0] += 1
                if (row['Sentiment'] <= -0.05):
                    values[1] += 1
                
    return potential_naes_dict

def get_naes(df):
    dict = get_potential_naes(df)
    naes_dict = {}

    for entity in dict:
        p = (dict[entity][1] / dict[entity][0])
        if (p > P_THRESHOLD):
            naes_dict[entity] = p

    return naes_dict

# corpus specific methods

def sum_naes_in_corpus(dict, corpus):
    sum = 0
    for entity in dict:
        if (corpus.find(entity) != -1):
            sum += dict[entity]
    
    return sum

def most_negative_entity_in_corpus(dict, corpus):
    max = 0
    max_key = ""
    for entity in dict:
        if(corpus.find(entity) != -1 and dict[entity] > max):
            max = dict[entity]
            max_key = entity

    return (max, max_key)

def nssm_a(corpus, sentiment, naes):
    smoothed =  sentiment * (ALPHA + (BETA * (1 - most_negative_entity_in_corpus(naes, corpus)[0])))

    sentiment_final = 0
    if (smoothed <= -0.05):
        sentiment_final = -1
    elif (smoothed > -0.05 and smoothed < 0.05):
        sentiment_final = 0
    else:
        sentiment_final = 1
    
    return (smoothed, sentiment_final)

def nssm_b(corpus, sentiment, naes):
    smoothed =  sentiment * (ALPHA + (1 / (BETA / (1 + sum_naes_in_corpus(naes, corpus)))))

    sentiment_final = 0
    if (smoothed <= -0.05):
        sentiment_final = -1
    elif (smoothed > -0.05 and smoothed < 0.05):
        sentiment_final = 0
    else:
        sentiment_final = 1
    
    return (smoothed, sentiment_final)

def nssm_c(corpus, sentiment, naes):
    smoothed = sentiment * (ALPHA + (BETA / 1 + most_negative_entity_in_corpus(naes, corpus)[0]))

    sentiment_final = 0
    if (smoothed <= -0.05):
        sentiment_final = -1
    elif (smoothed > -0.05 and smoothed < 0.05):
        sentiment_final = 0
    else:
        sentiment_final = 1
    
    return (smoothed, sentiment_final)


def apply_nssm(df):
    naes = get_naes(df)

    for index, row in df.iterrows():
        a = nssm_a(row['Body'], row['Sentiment'], naes)
        b = nssm_b(row['Body'], row['Sentiment'], naes)
        c = nssm_c(row['Body'], row['Sentiment'], naes)
        df.loc[index, 'A_Sentiment'] = a[0]
        df.loc[index, 'B_Sentiment'] = b[0]
        df.loc[index, 'C_Sentiment'] = c[0]
        df.loc[index, 'A_Sentiment_Final'] = a[1]
        df.loc[index, 'B_Sentiment_Final'] = b[1]
        df.loc[index, 'C_Sentiment_Final'] = c[1]

    return df