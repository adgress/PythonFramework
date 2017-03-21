import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_rcv1

twenty_ngs = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
rcv1 = fetch_rcv1()

raw_data = pickle.load(open('reuters.pkl'))
vectorizer = CountVectorizer()


print ''