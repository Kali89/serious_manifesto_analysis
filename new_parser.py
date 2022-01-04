import re
import calendar
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
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import textstat
from collections import Counter
import seaborn as sns
from scipy.stats import kendalltau

sns.set(font_scale=1.5, rc={'text.usetex': True})
code_dict = {
    51320: 'Labour',
    51620: 'Conservative'
}


def transform_filename(filename, code_dict):
    split_name = filename.split('_')
    party = code_dict.get(int(split_name[0]), 'Unknown')
    year = split_name[1][:4]
    month = int(split_name[1][4:].split('.')[0])
    return '{0} {1} {2}'.format(party, calendar.month_abbr[month], year)


filelist = list(os.walk('.'))
manifestos = sorted(
    [entry for entry in list(filelist)[0][2] if '.csv' in entry],
    key = lambda x: (int(x.split('_')[1][:6]), int(x.split('_')[0]))
)
manifesto_names = [transform_filename(entry, code_dict) for entry in manifestos]

PUNCTUATION_REGEX = re.compile("[^\w\s]")
NUMBER_REGEX = re.compile("(^|\W)\d+")


def tokenize(lines):
    return [word_tokenize(line) for line in lines]


def strip_punctuation_and_blank_lines(lines):

    return [
        PUNCTUATION_REGEX.sub(u'', word).lower() for line in lines for word in line if PUNCTUATION_REGEX.sub(u'', word)
    ]


def strip_na(words):
    return [word for word in words if word not in ('na', 'h')]


def strip_numbers(words):
    return [
        NUMBER_REGEX.sub(u'', word).lower() for word in words if NUMBER_REGEX.sub(u'', word)
    ]


def get_formatted_text(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.readlines()
    tokenized = [word_tokenize(line) for line in raw]
    without_punctuation = strip_punctuation_and_blank_lines(tokenized)
    without_na = strip_na(without_punctuation)
    without_numbers = strip_numbers(without_na)
    print(without_numbers[:5])
    return without_numbers


def get_raw_text(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.readlines()
    tokenized = [word_tokenize(line) for line in raw]
    joined = [word for line in tokenized for word in line]
    return ' '.join(joined)


raw_text = dict((name, get_raw_text(name)) for name in manifestos)

election_dict = dict((name, get_formatted_text(name)) for name in manifestos)

# Length of manifesto
values = [len(val) for val in election_dict.values()]

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.3
x = np.arange(len(manifesto_names)/2)

plt.bar(x, values[::2], width=width, label='Labour', color='red')
plt.bar(x + width, values[1::2], width=width, label='Conservative', color='blue')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(
    x + width/2,
    [' '.join(e.split(' ')[1:]) for e in manifesto_names][::2],
    fontsize=12,
    rotation=30
)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.xlabel('Date of election', fontsize=14)
plt.ylabel('Number of words in the manifesto', fontsize=14)
plt.title('Word count of UK political party manifestos', fontsize=16)
plt.tight_layout()
plt.savefig('manifesto_word_length.png')
plt.show()


def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words('english')]


def lexical_diversity(text, stop=False):
    if stop:
        text = [word for word in text if word not in stopwords.words('english')]
    return len(text)/len(set(text))


ld_values = [lexical_diversity(text) for text in election_dict.values()]

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.3
x = np.arange(len(manifesto_names)/2)

plt.bar(x, ld_values[::2], width=width, label='Labour', color='red')
plt.bar(x + width, ld_values[1::2], width=width, label='Conservative', color='blue')
#ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(
    x + width/2,
    [' '.join(e.split(' ')[1:]) for e in manifesto_names][::2],
    fontsize=12,
    rotation=30
)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.xlabel('Date of election', fontsize=14)
plt.ylabel('Lexical Diversity', fontsize=14)
plt.title('Lexical diversity of UK political party manifestos', fontsize=16)
plt.tight_layout()
plt.savefig('manifesto_lexical_diversity.png')
plt.show()

without_stopwords = dict((key, remove_stopwords(val)) for key, val in election_dict.items())

def get_common_words(text):
    counter = Counter(text)
    return counter.most_common(n=10)

from nltk.corpus import words
english_words = set(words.words())

def non_english_words(text):
    filtered = [word for word in text if word not in english_words]
    counter = Counter(filtered)
    return counter.most_common(n=10)

tory_words = [word for key, val in without_stopwords.items() for word in val if code_dict[int(key.split('_')[0])] == 'Conservative']
labour_words = [word for key, val in without_stopwords.items() for word in val if code_dict[int(key.split('_')[0])] == 'Labour']


most_common_words = dict((key, get_common_words(val)) for key, val in without_stopwords.items())


ld_no_stopwords_values = [lexical_diversity(text, True) for text in election_dict.values()]

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.3
x = np.arange(len(manifesto_names)/2)

plt.bar(x, ld_no_stopwords_values[::2], width=width, label='Labour', color='red')
plt.bar(x + width, ld_no_stopwords_values[1::2], width=width, label='Conservative', color='blue')
#ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
plt.xticks(
    x + width/2,
    [' '.join(e.split(' ')[1:]) for e in manifesto_names][::2],
    fontsize=12,
    rotation=30
)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.xlabel('Date of election', fontsize=14)
plt.ylabel('Lexical Diversity', fontsize=14)
plt.title('Lexical diversity of UK political party manifestos', fontsize=16)
plt.tight_layout()
plt.savefig('manifesto_lexical_diversity_no_stopwords.png')
plt.show()




#tfidf = TfidfVectorizer(ngram_range=(1,4), min_df=2, stop_words='english')
tfidf = TfidfVectorizer(ngram_range=(2,4), stop_words='english', min_df=1, norm='l2')
outy = tfidf.fit_transform([' '.join(text) for text in election_dict.values()])

feature_array = np.array(tfidf.get_feature_names())
word_ranking = feature_array[np.argsort(outy.toarray()[-1])[::-1]]

cons_only = [item for key, item in election_dict.items() if code_dict[int(key.split('_')[0])] == 'Conservative']
cons_outy = tfidf.fit_transform([' '.join(text) for text in cons_only])
feature_array = np.array(tfidf.get_feature_names())
cons_word_ranking = feature_array[np.argsort(cons_outy.toarray()[-1])[::-1]]

labour_only = [item for key, item in election_dict.items() if code_dict[int(key.split('_')[0])] == 'Labour' or key == '51620_201912.csv']
labour_outy = tfidf.fit_transform([' '.join(text) for text in labour_only])
feature_array = np.array(tfidf.get_feature_names())
labour_word_ranking = feature_array[np.argsort(labour_outy.toarray()[-1])[::-1]]

recent_only = [item for key, item in election_dict.items() if int(key.split('_')[1][:4]) >= 2015 ]
recent_outy = tfidf.fit_transform([' '.join(text) for text in recent_only])
feature_array = np.array(tfidf.get_feature_names())
recent_word_ranking = feature_array[np.argsort(recent_outy.toarray()[-1])[::-1]]

def max_diff(list_a, list_b):
    difference_dict = {}
    better_a = dict((word, (len(list_a) - i)/len(list_a)) for i, word in enumerate(list_a))
    better_b = dict((word, (len(list_b) - i)/len(list_b)) for i, word in enumerate(list_b))
    for key in set().union(*[better_a, better_b]):
        a_score = better_a.get(key, 0)
        b_score = better_b.get(key, 0)
        difference_dict[key] = a_score - b_score
    return difference_dict



pairwise_similarity = outy * outy.T
#
similarity = pd.DataFrame(pairwise_similarity.toarray(), columns=manifesto_names, index=manifesto_names).unstack().reset_index()
#
similarity.columns = ['base', 'comparison', 'correlation']
similarity = similarity[similarity.base != similarity.comparison]
similarity['joined'] = similarity.base + '-' + similarity.comparison

terms_to_keep = [
    '{0}-{1}'.format(a1, a2) for a1, a2 in itertools.combinations(
        manifesto_names, 2
    )
]

final_similarity = similarity[similarity.joined.isin(terms_to_keep)][['joined', 'correlation']]
terms_to_keep = [
    '{0}-{1}'.format(a1, a2) for a1, a2 in itertools.combinations(
        manifesto_names, 2
    )
]

## Kendall Tau measurement

output_rankings = {}
output_pairs = {}
for pair in itertools.combinations_with_replacement(range(1,5), 2):
    print(pair)
    tfidf = TfidfVectorizer(ngram_range=pair, stop_words='english', min_df=1)
    outy = tfidf.fit_transform([' '.join(text) for text in election_dict.values()])
    pairwise_similarity = outy * outy.T
    similarity = pd.DataFrame(pairwise_similarity.toarray(), columns=manifesto_names, index=manifesto_names).unstack().reset_index()
    #
    similarity.columns = ['base', 'comparison', 'correlation']
    similarity = similarity[similarity.base != similarity.comparison]
    similarity['joined'] = similarity.base + '-' + similarity.comparison
    final_similarity = similarity[similarity.joined.isin(terms_to_keep)][['joined', 'correlation']]
    final_similarity.loc[:, 'rankle'] = final_similarity.correlation.rank(method='average')
    output_rankings[pair] = final_similarity
    output_pairs[pair] = pairwise_similarity

## Plotting
to_use = output_pairs[(2,4)].toarray()

fig, ax = plt.subplots(figsize=(14,12))
mask = np.triu(np.ones_like(to_use, dtype=np.bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heatmap = sns.heatmap(to_use, ax=ax, mask=mask, cmap=cmap, square=True)
plt.title('Manifesto similarities')
ax.set_xticklabels(
    manifesto_names,
    rotation=90,
    fontsize=12
#    horizontalalignment='right'
)
ax.set_yticklabels(
    manifesto_names,
    rotation=0,
    fontsize=12
#    horizontalalignment='right'
)

#plt.tight_layout(w_pad=0.5)
plt.subplots_adjust(left=0.02, bottom=0.25, right=0.98, top=0.95, wspace=0, hspace=0)
plt.savefig('all_manifesto_heatmaps_2_4.png')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
width = 0.3
x = np.arange(len(manifesto_names)/2)

same_year = output_rankings[(2,4)][
    output_rankings[(2,4)].joined.apply(lambda x:
                                        x.split('-')[0].split(' ')[1:] == x.split('-')[1].split(' ')[1:]
                                       )
]

plt.bar(x, same_year.correlation, width=width)
plt.xticks(
    x,
    [' '.join(e.split(' ')[1:]) for e in manifesto_names][::2],
    fontsize=12,
    rotation=30
)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.xlabel('Date of election', fontsize=14)
plt.ylabel('Cosine similarity', fontsize=14)
plt.title('Cosine similarity of the Labour and Conservative election manifestos', fontsize=16)
plt.tight_layout()
plt.savefig('manifesto_similarity.png')
plt.show()



to_use_labour = output_pairs[(2,4)].toarray()[::2, ::2]

fig, ax = plt.subplots(figsize=(12,8))
mask = np.triu(np.ones_like(to_use_labour, dtype=np.bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heatmap = sns.heatmap(to_use_labour, mask=mask, cmap=cmap, ax=ax, square=True)
#plt.title('Heatmap')
ax.set_xticklabels(
    manifesto_names[::2],
    rotation=90,
#    horizontalalignment='right'
)
ax.set_yticklabels(
    manifesto_names[::2],
    rotation=0,
#    horizontalalignment='right'
)

plt.title('Labour manifesto similarities')

plt.tight_layout()
plt.savefig('labour_manifesto_heatmaps_2_4.png')
plt.show()

to_use_tory = output_pairs[(2,4)].toarray()[1::2, 1::2]

fig, ax = plt.subplots(figsize=(12,8))
mask = np.triu(np.ones_like(to_use_tory, dtype=np.bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heatmap = sns.heatmap(to_use_tory, mask=mask, cmap=cmap, ax=ax, square=True)
#plt.title('Heatmap')
ax.set_xticklabels(
    manifesto_names[1::2],
    rotation=90,
#    horizontalalignment='right'
)
ax.set_yticklabels(
    manifesto_names[1::2],
    rotation=0,
#    horizontalalignment='right'
)

plt.title('Conservative manifesto similarities')

plt.tight_layout()
plt.savefig('tory_manifesto_heatmaps_2_4.png')
plt.show()

## Kendall Tau similarity

dicty = {}
for a, b in itertools.combinations(output_rankings.keys(), 2):
    corr = kendalltau(
        output_rankings[a].rankle,
        output_rankings[b].rankle
    ).correlation
    parsed_a = "{0} - {1}".format(a[0], a[1])
    parsed_b = "{0} - {1}".format(b[0], b[1])
    try:
        dicty[parsed_a][parsed_b] = corr
    except:
        dicty[parsed_a] = {parsed_b: corr}

values = pd.DataFrame(dicty).values
fig, ax = plt.subplots(figsize=(12,8))
#mask = np.triu(np.ones_like(values, dtype=np.bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heatmap = sns.heatmap(values, cmap=cmap, ax=ax, square=True, vmin=-1, vmax=1, annot=True)
#plt.title('Heatmap')
plt.xlabel('N-grams generated')
plt.ylabel('N-grams generated')
ax.set_xticklabels(
    [
        "$({0}, {1})$".format(x.split('-')[0].strip(), x.split('-')[1].strip()) for x in pd.DataFrame(dicty).columns
    ],
    rotation=90,
    usetex=True
#    horizontalalignment='right'
)
ax.set_yticklabels(
    [
        "$({0}, {1})$".format(x.split('-')[0].strip(), x.split('-')[1].strip()) for x in pd.DataFrame(dicty).index
    ],
    rotation=0,
    usetex=True
#    horizontalalignment='right'
)

plt.title('Kendall tau distance between manifesto similarity rankings')

plt.tight_layout()
plt.savefig('kendall_tau_distance.png')
plt.show()


terms_matrix = outy.toarray().T
#
num_files = len(manifestos)
#
col_list = []


for x1, x2 in itertools.combinations(range(num_files), 2):
    thing = np.multiply(
        outy[x1].toarray(),
        outy[x2].toarray()
    )
    col_list.append(thing[0])
#
for x1, x2 in itertools.combinations(range(num_files), 2):
    t = np.ones((num_files, 1)) * 0
    t[x1] = 1
    t[x2] = 1
    col_list.append(t)
#
#
other_matrix = np.array(col_list).T
feature_names = tfidf.get_feature_names()
#
common_word_dict = {}
for index, files in enumerate(itertools.combinations(manifesto_names, 2)):
    word_list = []
    top_10_indices = np.argpartition(other_matrix[:, index], -10)[-10:]
    top_10_scores = other_matrix[:, index][top_10_indices]
    for index, score in sorted(
        zip(top_10_indices, top_10_scores),
        key=lambda x: x[1], reverse=True
    ):
        word = feature_names[index]
        word_list.append((word, score))

    common_word_dict["{0}-{1}".format(files[0], files[1])] = word_list

## Biggest words per each manifesto

for i, name in enumerate(manifesto_names):
    print(
        "{0}-{1}".format(
            name,
            feature_array[np.argsort(outy.toarray()[i])[::-1]][0]
        )
    )


indices_of_nhs = [i for i, word in enumerate(feature_names) if 'nhs' in word]
total_tfidf_of_nhs = np.sum(outy[:, indices_of_nhs].toarray(), axis=1)
