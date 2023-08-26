import vader_english
import nssm_english
import time_series

time_series.analyze_with_nssm('./data/reuters.csv', 'Reuters', vader_english.sentiment, nssm_english)
#time_series.analyze_with_nssm('./data/cnn.csv', 'CNN', vader_english.sentiment, nssm_english)
#time_series.analyze_with_nssm('./data/fox.csv', 'Fox News', vader_english.sentiment, nssm_english)
#time_series.analyze_with_nssm('./data/nytimes.csv', 'The New York Times', vader_english.sentiment, nssm_english)