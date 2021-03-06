import os
import gensim
import nltk
from nltk.tokenize import word_tokenize

#***************************************
# Author: Sudarsanan B S
#***************************************


raw_documents = ["Sometimes it's better to light a flamethrower than curse the darkness.",
                 "The situation was horrible, but not without its fascination.",
             "So we beat on, boats against the current, borne back ceaselessly into the past.",
             "So I walked back to my room and collapsed on the bottom bunk, thinking that if people were rain, I was drizzle and she was a hurricane.",
            "In a hole in the ground there lived a hobbit."]

#####Tokenize#####
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]

#####Create a dictionary that maps every word to a number#####
dictionary = gensim.corpora.Dictionary(gen_docs)

#####Building a corpus#####
# corpus      : list of "bag of words" of the document
# bag of words: a representation where each word is mapped to its frequency (TF) in the document
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

#####Building a TF-IDF model of the corpus#####
# TF : Term frequency 
#           Number of times a certain word occurs in the document
# IDF: Inverse Document Frequency 
#           Measure of how much information a word can hold
#           Inverse of "how many documents this term occurs in"
#           For example, more common words like 'the', 'a' need not be given too much importance
tf_idf = gensim.models.TfidfModel(corpus)

######Create a similarity measure object of the TF-IDF model#####
# The Similarity class splits the index into several -
# smaller sub-indexes ('shards'), which are disk-based.
# This is useful for large documents in order to prevent RAM overload.
# These temporary shards are to be stored in a sub directory (in the current working directory) called "shards_indices"
path= os.getcwd()
path = os.path.join(path ,'shards_indices')
sims = gensim.similarities.Similarity(path,tf_idf[corpus],
                                      num_features=len(dictionary))

#########################################################################################

#####Query document######

query = "The situation was overwhelming!"
print 'Query: '+query
print '======================'
query_doc = [w.lower() for w in word_tokenize(query)]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]

##########################################################################################

#Pass query doc for similarity comparison
result = sims[query_doc_tf_idf]

i=0
for value in result:
    print str(raw_documents[i]) + '==> ' + str(value)
    i+=1
