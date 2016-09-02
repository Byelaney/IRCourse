""" Assignment 2
"""
import abc
import math

import index


def idf(term, index):
    """ Compute the inverse document frequency of a term according to the
    index. IDF(T) = log10(N / df_t), where N is the total number of documents
    in the index and df_t is the total number of documents that contain term
    t.
    Params:
      terms....A string representing a term.
      index....A Index object.
    Returns:
      The idf value.
    >>> idx = index.Index(['a b c a', 'c d e', 'c e f'])
    >>> idf('a', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('d', idx) # doctest:+ELLIPSIS
    0.477...
    >>> idf('e', idx) # doctest:+ELLIPSIS
    0.176...
    """
    if index.doc_freqs[term] == 0 or len(index.documents) == 0:
        return 0
    return math.log(len(index.documents) *1. / index.doc_freqs[term],10)

class ScoringFunction:
    """ An Abstract Base Class for ranking documents by relevance to a
    query. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def score(self, query_vector, index):
        """
        Do not modify.
        Params:
          query_vector...dict mapping query term to weight.
          index..........Index object.
        """
        return


class RSV(ScoringFunction):
    """
    See lecture notes for definition of RSV.
    idf(a) = log10(3/1)
    idf(d) = log10(3/1)
    idf(e) = log10(3/2)
    >>> idx = index.Index(['a b c', 'c d e', 'c e f']) # doctest:+ELLIPSIS
    >>> rsv = RSV() # doctest:+ELLIPSIS
    >>> rsv.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.4771...
    """

    def score(self, query_vector, index):
        rsv = {}
        doc_id = 0
        for doc in index.documents:
            for query in query_vector:
                if query in index.documents[doc_id]:
                    tmp_rsv = math.log(len(index.documents) * 1. / index.doc_freqs[query], 10)
                    if (doc_id+1) in rsv.keys():
                        rsv[doc_id + 1] += tmp_rsv
                    else:
                        rsv[doc_id + 1] = tmp_rsv
            doc_id += 1
        return rsv




    def __repr__(self):
        return 'RSV'


class BM25(ScoringFunction):
    """
    See lecture notes for definition of BM25.
    log10(3) * (2*2) / (1(.5 + .5(4/3.333)) + 2) = log10(3) * 4 / 3.1 = .6156...
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f']) # doctest:+ELLIPSIS
    >>> bm = BM25(k=1, b=.5) # doctest:+ELLIPSIS
    >>> bm.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.61564032...
    """
    def __init__(self, k=1, b=.5):
        self.k = k
        self.b = b

    def score(self, query_vector, index):
        bm25 = {}
        doc_id = 0
        for doc in index.documents:
            for query in query_vector:
                if query in index.documents[doc_id]:
                    tf_i = index.documents[doc_id].count(query)
                    tmp_bm = math.log(len(index.documents) * 1. / index.doc_freqs[query], 10) * (self.k + 1) * tf_i / (
                    self.k * (1 - self.b + self.b * len(index.documents[doc_id])/index.mean_doc_length) + tf_i)
                    if (doc_id + 1) in bm25.keys():
                        bm25[doc_id + 1] += tmp_bm
                    else:
                        bm25[doc_id + 1] = tmp_bm
            doc_id += 1
        return bm25


    def __repr__(self):
        return 'BM25 k=%d b=%.2f' % (self.k, self.b)


class Cosine(ScoringFunction):
    """
    See lecture notes for definition of Cosine similarity.  Be sure to use the
    precomputed document norms (in index), rather than recomputing them for
    each query.
    >>> idx = index.Index(['a a b c', 'c d e', 'c e f']) # doctest:+ELLIPSIS
    >>> cos = Cosine() # doctest:+ELLIPSIS
    >>> cos.score({'a': 1.}, idx)[1]  # doctest:+ELLIPSIS
    0.792857...
    """
    def score(self, query_vector, index):
        cos = {}
        for word in query_vector:
            for tempList in index.index[word]:
                tmp_val = query_vector[word] * (1. + math.log(tempList[1],10)) * (idf(word, index))
                if tempList[0] in cos:
                    cos[tempList[0]] += tmp_val
                else:
                    cos[tempList[0]] = tmp_val

        for doc_id in cos.keys():
            cos[doc_id] /= index.doc_norms[doc_id]
        return cos

    def __repr__(self):
        return 'Cosine'
