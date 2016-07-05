""" Assignment 0
You will implement a simple in-memory boolean search engine over the jokes
from http://web.hawkesnest.net/~jthens/laffytaffy/.
The documents are read from documents.txt.
The queries to be processed are read from queries.txt.
Your search engine will only need to support AND queries. A multi-word query
is assumed to be an AND of the words. E.g., the query "why because" should be
processed as "why AND because."
"""
from collections import defaultdict
import re


"""remove all punctuations"""
def bye_punctuation(document):
    doc = []
    punctuation = [',', '.', '?', '!', '(', ')', ';', ':', '\'', '-', '"', '`']
    for letter in document:
        if letter in punctuation:
            doc.append(' ')
        else:
            doc.append(letter.lower())
    return ''.join(doc)

def tokenize(document):
    """ Convert a string representing one document into a list of
    words. Remove all punctuation and split on whitespace.
    Params:
      document...a string to be tokenized
    Returns:
      A list of strings, one per token.
    Here is a doctest:
    >>> tokenize("Hi  there. What's going on?")
    ['hi', 'there', 'what', 's', 'going', 'on']
    """
    return re.findall(r"[\w']+", bye_punctuation(document))


# print(tokenize("Hi  there. What's going on?"))


def create_index(tokens):
    """
    Create an inverted index given a list of document tokens. The index maps
    each unique word to a list of document ids, sorted in increasing order.
    Params:
      tokens...A list of lists of strings
    Returns:
      An inverted index. This is a dict where keys are words and values are
      lists of document indices, sorted in increasing order.
    Below is an example, where the first document contains the tokens 'a' and
    'b', and the second document contains the tokens 'a' and 'c'.
    >>> index = create_index([['a', 'b'], ['a', 'c']])
    >>> sorted(index.keys())
    ['a', 'b', 'c']
    >>> index['a']
    [0, 1]
    >>> index['b']
    [0]
    """
    index = {}
    doc_id = 0
    for document in tokens:
        for token in document:
            if token not in index.keys():
                index[token] = [doc_id]
            else:
                tmp = index[token]
                if doc_id not in tmp:
                    tmp.append(doc_id)
                index[token] = tmp
        doc_id = doc_id + 1

    return index

# index = create_index([['a', 'b'], ['a', 'c']])
# sorted(index.keys())
# print(index['b'])


def intersect(list1, list2):
    """ Return the intersection of two posting lists. Use the optimize
    algorithm of Figure 1.6 of the MRS text. Your implementation should be
    linear in the sizes of list1 and list2. That is, you should only loop once
    through each list.
    Params:
      list1....A list of document indices, sorted in ascending order.
      list2....Another list of document indices, sorted in ascending order.
    Returns:
      The list of document ids that appear in both lists, sorted in ascending order.
    >>> intersect([1, 3, 5], [3, 4, 5, 10])
    [3, 5]
    >>> intersect([1, 2], [3, 4])
    []
    """
    intersection = []
    p1 = 0
    p2 = 0
    while p1 < len(list1) and p2 < len(list2):
        if list1[p1] < list2[p2]:
            p1 += 1
        elif list1[p1] > list2[p2]:
            p2 += 1
        else:
            intersection.append(list1[p1])
            p1 += 1
            p2 += 1

    return intersection

#print(intersect([1, 2], [3, 4, 5, 10]))


def sort_by_num_postings(words, index):
    """
    Sort the words in increasing order of the length of their postings list in
    index. You may use Python's builtin sorted method.
    Params:
      words....a list of strings.
      index....An inverted index; a dict mapping words to lists of document
      ids, sorted in ascending order.
    Returns:
      A list of words, sorted in ascending order by the number of document ids
      in the index.
    >>> sort_by_num_postings(['a', 'b', 'c'], {'a': [0, 1], 'b': [1, 2, 3], 'c': [4]})
    ['c', 'a', 'b']
    """
    tmp_lst = []
    tmp_index = {}
    for key in index.keys():
        tmp_index[key] = len(index[key])
    tmp_lst = sorted(tmp_index, key=tmp_index.__getitem__)
    tmp_words = words
    tmp_words.sort(key=lambda x: tmp_lst.index(x))
    return tmp_words

# print(sort_by_num_postings(['a', 'b', 'c'], {'a': [0, 1], 'b': [1, 2, 3], 'c': [4]}))

def search(index, query):
    """ Return the document ids for documents matching the query. Assume that
    query is a single string, possibly containing multiple words. The steps
    are to:
    1. tokenize the query
    2. Sort the query words by the length of their postings list
    3. Intersect the postings list of each word in the query.
    If a query term is not in the index, then an empty list should be returned.
    Params:
      index...An inverted index (dict mapping words to document ids)
      query...A string that may contain multiple search terms. We assume the
      query is the AND of those terms by default.
    E.g., below we search for documents containing 'a' and 'b':
    >>> search({'a': [0, 1], 'b': [1, 2, 3], 'c': [4]}, 'a b')
    [1]
    """
    result = []
    tmp_query = tokenize(query)
    if (1 == len(tmp_query)):
        result = index[tmp_query[0]]
    else:
        result = index[tmp_query[0]]
        for word in tmp_query[1:]:
            result = intersect(result, index[word])
    return result




def main():
    """ Main method. You should not modify this. """
    documents = open('documents.txt').readlines()
    tokens = [tokenize(d) for d in documents]
    index = create_index(tokens)
    queries = open('queries.txt').readlines()
    for query in queries:
        results = search(index, query)
        print('\n\nQUERY:%s\nRESULTS:\n%s' % (query, '\n'.join(documents[r] for r in results)))


if __name__ == '__main__':
    main()
