import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import sys
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import itertools
import numpy as np
from nltk.util import ngrams

election_years = [
    1979,
    1983,
    1987,
    1992,
    1997,
    2001,
    2005,
    2010,
    2015,
    2017,
    2019
]

filelist = [
    'LabourManifesto%d' % entry for entry in election_years
] + [
    'ToryManifesto%d' % entry for entry in election_years
]

def tokenize(lines):
    return [word_tokenize(line) for line in lines]

regex = re.compile('[{0}]'.format(re.escape(string.punctuation)))
yet_another_regex = re.compile("^\d+\s|\s\d+\s|\s\d+$")

def strip_punctuation(lines):
    return [regex.sub(u'', word).lower() for line in lines for word in line if regex.sub(u'', word)]

def strip_stopwords(words):
    return " ".join(word for word in words if not word in stopwords.words('english'))

def get_formatted_text(filename):
    with open(filename + '.txt', 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.readlines()
#    return strip_stopwords(strip_punctuation(tokenize(raw)))
    return " ".join(strip_punctuation(tokenize(raw)))

election_dict = dict((name, yet_another_regex.sub(u' ', get_formatted_text(name))) for name in filelist)

tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=2, stop_words='english')
outy = tfidf.fit_transform(election_dict.values())
pairwise_similarity = outy * outy.T

similarity = pd.DataFrame(pairwise_similarity.toarray(), columns=filelist, index=filelist).unstack().reset_index()

similarity.columns = ['base', 'comparison', 'correlation']
similarity = similarity[similarity.correlation < 0.999]
similarity['joined'] = similarity.base + '-' + similarity.comparison
similarity['unique_joined'] = similarity.joined.apply(lambda x: '-'.join(entry for entry in sorted(x.split('-'))))
final_similarity = similarity[['unique_joined', 'correlation']].drop_duplicates()

terms_matrix = outy.toarray().T

num_files = len(filelist)

col_list = []

for x1, x2 in itertools.combinations(range(num_files), 2):
    t = np.ones((num_files, 1)) * -1
    t[x1] = 1
    t[x2] = 1
    col_list.append(t)


the_shit = np.matmul(terms_matrix, np.hstack(col_list))
feature_names = tfidf.get_feature_names()

common_word_dict = {}
for index, files in enumerate(itertools.combinations(filelist, 2)):
    best_10_words = [feature_names[entry] for entry in np.argpartition(the_shit[:, index], -10)[-10:]]
    common_word_dict["{0}-{1}".format(files[0], files[1])] = best_10_words
