from collections import Counter
import csv, json
import math
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from pandas.plotting import scatter_matrix
import regex as re
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob 


csv_file_0 = '/Users/arianamaisonet/Downloads/corpus/trump_pre_musk.csv'
csv_file_1 = '/Users/arianamaisonet/Downloads/corpus/trump_back_on_twitter.csv'

#creating a pandas dataframe from the csv file containing tweets from 4-29-22 - 11-4-24
df0 = pd.read_csv(csv_file_0, encoding='utf-8')
df1 = pd.read_csv(csv_file_1, encoding='utf-8')

#here I remove retweets, tweets that only contain urls, and ones with no text: these tend to be photos with text like Fox News polls
def clean_tweets(df: pd.DataFrame) -> pd.DataFrame:    
    df = df.drop('date', axis=1)                   #removing the date column because one of the datasets has an invalid data
    
    retweeted = df[df['isRetweet'].isin(['True', 't'])]    
    df['isRetweet'] = df['isRetweet'].astype('string')
    df = df.drop(retweeted.index)
    df = df[~df['text'].str.startswith('RT', na=False)]

    df = df.dropna(subset=['text'])                     #removing tweets that are empty or purely urls
    df['text'] = df['text'].astype(str)
    df = df[~df['text'].str.startswith('http', na=False)]
    return df
df0 = clean_tweets(df0)
df1 = clean_tweets(df1)

#using Collections.counter to count tokenized tweets
def CountTokens(pandas_df, output_ls):
    tweets = pandas_df['text'].astype(str).tolist()
    text = ' '.join(tweets)
    text = text.lower()
    text_no_punct = re.sub(r'\p{P}+', '', text)
    tokens = text_no_punct.split()

    for token, count in Counter(tokens).items():
        if not token.startswith('http'):
            output_ls.append([token, count])
tokens_ls_0 = []
tokens_ls_1 = [] 
CountTokens(df0, tokens_ls_0)
CountTokens(df1, tokens_ls_1)

# def total_tokens(ls):
#     total = 0
#     for row in ls:
#         total += row[1]
#     return total

# N_0 = total_tokens(tokens_ls_0)
# N_1 = total_tokens(tokens_ls_1)

# def log_odds(row, N:int):
#     p = row[1] / N
#     p_complement = 1 - p
#     O = p/p_complement
#     log_odd = math.log10(O)
#     return log_odd

# def subtraction(x,y):
#     delta = x - y
#     delta = round(delta, 4)
#     return delta

# def calculate_log_odds(counted_tokens_0, counted_tokens_1, N_0, N_1):
#     new_list = []
#     for row_0 in counted_tokens_0:
#         for row_1 in counted_tokens_1:
#             if row_0[0] == row_1[0]:
#                 lst = []
#                 lo_token_0 = log_odds(row_0, N_0)
#                 lo_token_1 = log_odds(row_1, N_1)
#                 lgr = subtraction(lo_token_0, lo_token_1)
#                 lst.append(row_0[0])
#                 lst.append(lgr)
#                 new_list.append(lst)
#             else:
#                 continue
#     return new_list

# lgr_list = calculate_log_odds(tokens_ls_0, tokens_ls_1, N_0, N_1)
# sorted_lgr_list = sorted(lgr_list, key=lambda x: x[1])
# def t_table(ls):
#     df = pd. DataFrame(ls, columns=['token', 'log_odds'])
#     return df
# lor_table = t_table(sorted_lgr_list)
# df_top = t_table(sorted_lgr_list[:50])
# df_top.to_csv('top_50_log_odds.csv', index=False)
# df_bottom = t_table(sorted_lgr_list[-50:])
# df_bottom.to_csv('bottom_50_log_odds.csv', index=False)

#adding a column that displays length of tweet text
def add_length_column(df: pd.DataFrame) -> pd.DataFrame:
    words_in_i = []
    for i in range(len(df)):
        holder = str(df['text'].iloc[i])
        list_holder = len(holder.split())
        words_in_i.append(list_holder)
    df['text_length'] = words_in_i
    return df
df0 = add_length_column(df0)
df1 = add_length_column(df1)


#adding a column that displays the number of tokens in each tweet
def add_token_count_column(df: pd.DataFrame, tokens_ls: list) -> pd.DataFrame:
    df_text_list = df['text'].astype(str).tolist()
    token_count = []
    for i in df_text_list:
        tokenized_i = list(re.sub(r'\p{P}+', '', i.lower()))
        count = 0
        for word in tokenized_i:
            if word in tokens_ls[0]:
                count += 1
        token_count.append(count)
    df['token_count'] = token_count
    df['token_count'] = df['token_count'].astype(int)
    return df
df0 = add_token_count_column(df0, tokens_ls_0)
df1 = add_token_count_column(df1, tokens_ls_1)

#preparing to merge the two dataframes
def prepare_for_merge(df0: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    df0['pre_Musk'] = 1
    df1['pre_Musk'] = 0
    return pd.concat([df0, df1], ignore_index=True)
df = prepare_for_merge(df0, df1)
df = df.drop(['isRetweet'], axis=1)

#adding a column that displays the sentiment and subjectivity of each tweet using SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')
df['sentiment'] = df['text'].apply(lambda x: nlp(x)._.blob.polarity if isinstance(x, str) else 0)
df['subjectivity'] = df['text'].apply(lambda x: nlp(x)._.blob.subjectivity if isinstance(x, str) else 0)
    
pos_tags = set()
for doc in nlp.pipe(df['text']):
    pos_tags.update([token.pos_ for token in doc])
for tag in pos_tags:
    df[tag] = 0
 
for i,doc in enumerate(nlp.pipe(df['text'])):
    for token in doc:
        df.at[i, token.pos_] += 1

df.to_csv('trump_tweets.csv', index=False)



#making a classifier from df0 and df1
# df.hist(bins = 30, figsize=(20,15))
# at = ['text_length', 'token_count', 'sentiment', 'subjectivity']
# scatter_matrix(df[at], figsize=(12,8))

