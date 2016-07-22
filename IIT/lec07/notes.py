import math

def intersect(list1,list2):
    return list(set(list1) & set(list2))

# list1 = [1,2,3,4,5]
# list2 = [2,3,6]
#
# print(intersect(list1,list2))
#
def set_intersection(set1,set2):
    smaller = set1 if len(set1) < len(set2) else set2
    larger = set2 if len(set1) < len(set2) else set1
    return set([x for x in smaller if x in larger])

# set1 = set(list1)
# set2 = set(list2)
#
# print(set_intersection(set1,set2))


# q = "dogs with hats"
# d = "I like hats"
#
# s1 = set(q.split())
# s2 = set(d.split())
#
# print(s1 | s2)
#

def Jaccard_coefficient(q,d):
    # q = "dogs with hats"
    # d = "big cats wearing hats"
    s1 = set(q.split())
    s2 = set(d.split())
    return len(s1&s2)*1.0/len(s1|s2)

print(Jaccard_coefficient(q,d))


# ---------------------------------------------------
# Term Frequency

def tft_d(t,d):
    # term t in document d
    # t = "dog" ; d = "dogs with hats"
    return d.split().count(t)

def w_td(t,d):
    tftd = tft_d(t,d)
    if tftd == 0:return 0
    return math.log(tftd,10) + (1 if tftd>0 else 0)


# t = "example"
# d = "this is another another example example example"
# q = "I have this one example example"
#
# print(tft_d(t,d))

def score_i(q,d_i):
    return sum([w_td(x,d_i) for x in set(q.split())])

#print(score_i(q,d))

# ---------------------------------------------------
# Document Frequency

def t_in_d(t,d):
    if t in d.split(): return True
    else: return False

def idf_t(t,all_d):
    N = len(all_d)
    dft = 0
    for d in all_d:
       if t_in_d(t, d):
           dft += 1
    if dft == 0: return 0
    return math.log(1.*N/dft,10)

# t = "example"
# d1 = "this is another another example example example"
# d2 = "I have this one "
# d = [d1,d2]
# print (idf_t(t,d))

# ---------------------------------------------------
# tf-idf

def tf_idf(t,all_d):
    scores = 0
    for d in all_d:
        scores += tft_d(t,d) * idf_t(t,all_d)
    return scores

# q = "example"
# d1 = "this is a a sample"
# d2 = "this is another another example example example"
# d = [d1,d2]


# d1 = "the dog barked"
# d2 = "the dog jumped"
# d3 = "a cat jumped"
# t = "jumped"
# d = [d1,d2,d3]
# print(tf_idf(t,d))
# print(math.log(1.5,10)*2)
