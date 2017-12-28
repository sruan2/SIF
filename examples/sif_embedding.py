import sys
sys.path.append('../src')
import data_io, params, SIF_embedding
import numpy as np

# input
#wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
wordfile = '/Users/sherryruan/data/glove/glove.6B/glove.6B.300d.txt' # sherry: use glove.6B instead
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
sentences = ['this is an example sentence', 
			 'this is another sentence that is slightly longer',
			 'this is the same sentence',
			 'this is an example phrase',
			 'this is me']

# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
params = params.params()
params.rmpc = rmpc
# get SIF embedding
embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i

s1 = np.array(embedding[0,:]).reshape((1,300))
s2 = np.array(embedding[1,:]).reshape((1,300))
s3 = np.array(embedding[2,:]).reshape((1,300))
s4 = np.array(embedding[3,:]).reshape((1,300))

print(s1.shape) # (300,)
print(s1.dtype)

# =========== Compute Coisine Similarity of first two sentences ===================
def semantic_similarity(vec_1, vec_2):
    return (np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))[0,0] # shape is (0,0)

#print(semantic_similarity(s1, s2))

print("similarity is: " + str(semantic_similarity(s1, s2)))
print("similarity is: " + str(semantic_similarity(s1, s3)))
print("similarity is: " + str(semantic_similarity(s1, s4)))
print("similarity is: " + str(semantic_similarity(s2, s3)))
print("similarity is: " + str(semantic_similarity(s2, s4)))
print("similarity is: " + str(semantic_similarity(s3, s4)))


