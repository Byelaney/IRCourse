""" Assignment 2
"""
import abc

import numpy as np


class EvaluatorFunction:
    """
    An Abstract Base Class for evaluating search results.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, hits, relevant):
        """
        Do not modify.
        Params:
          hits...A list of document ids returned by the search engine, sorted
                 in descending order of relevance.
          relevant...A list of document ids that are known to be
                     relevant. Order is insignificant.
        Returns:
          A float indicating the quality of the search results, higher is better.
        """
        return


class Precision(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute precision.
        >>> Precision().evaluate([1, 2, 3, 4], [2, 4])
        0.5
        """
        if len(hits) == 0:
            return 0
        precision = 0
        for r in relevant:
            if r in hits:
                precision += 1
        return precision * 1. / len(hits)


    def __repr__(self):
        return 'Precision'


class Recall(EvaluatorFunction):

    def evaluate(self, hits, relevant):
        """
        Compute recall.
        >>> Recall().evaluate([1, 2, 3, 4], [2, 5])
        0.5
        """
        if len(relevant) == 0:
            return 0
        precision = 0
        for h in hits:
            if h in relevant:
                precision += 1
        return precision * 1. / len(relevant)


    def __repr__(self):
        return 'Recall'


class F1(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute F1.
        >>> F1().evaluate([1, 2, 3, 4], [2, 5])  # doctest:+ELLIPSIS
        0.333...
        """
        precision = Precision().evaluate(hits,relevant)
        recall = Recall().evaluate(hits,relevant)
        if (precision + recall) == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def __repr__(self):
        return 'F1'


class MAP(EvaluatorFunction):
    def evaluate(self, hits, relevant):
        """
        Compute Mean Average Precision.
        >>> MAP().evaluate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 4, 6, 11, 12, 13, 14, 15, 16, 17])
        0.2
        """
        temp = 0.0
        rel = 0.0
        for i in range(0, len(hits)):
            if hits[i] in relevant:
                rel += 1
                temp += (rel / (i + 1))
        return temp / len(relevant)


    def __repr__(self):
        return 'MAP'

