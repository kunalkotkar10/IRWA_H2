import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    # print('weights: ', weights)
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    # print('dictionary vec: ',dict(vec))
    # input()
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights):
    vec = defaultdict(float)
    n = len(read_docs('cacm.raw'))
    # print('n :', n)
    tf = compute_tf(doc, doc_freqs, weights)
    for i in tf:
        # j = tf[i]
        if i in doc_freqs:
            vec[i] = tf.get(i)  * np.log(n / doc_freqs.get(i))
            # print('vec for doc', vec[i])   
        else:
            vec[i] = 0
            # print('vec for 0:', vec[i])   
    # print('dictionary vec: ',dict(vec))
    return dict(vec)  # TODO: implement

def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(float)
    for word in doc.author:
            vec[word] = 1
    for word in doc.keyword:
            vec[word] = 1
    for word in doc.title:
            vec[word] = 1
    for word in doc.abstract:
            vec[word] = 1
    return dict(vec)  # TODO: implement

### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return (num / (norm(list(x.values())) * norm(list(y.values()))))

def dice_sim(x, y):
    '''
    Computes the dice similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return ((2*num) / (sum(list(x.values())) + sum(list(y.values()))))
    # return 0  # TODO: implement

def jaccard_sim(x, y):
    '''
    Computes the jaccard similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    if (sum(list(x.values())) + sum(list(y.values())) - num) == 0:
        return 1/10000000
    return (num / (sum(list(x.values())) + sum(list(y.values())) - num))
    # return 0  # TODO: implement

def overlap_sim(x, y):
    '''
    Computes the overlap similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return (num / min(sum(list(x.values())) , sum(list(y.values()))))
    # return 0  # TODO: implement


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''
    # print('recall: ',recall)
    # print('result:', results)
    # print('relevant', relevant)
    lower_r, upper_r = 0,0
    lower_p, upper_p = 1,1
    tp = 0
    c = 0
    # fp = 0
    # lower_flag = True
    for i in results:
        c += 1
        if i in relevant:
            tp += 1
            precision1 = tp / c
            recall1 = tp / len(relevant)
            if recall < recall1:
                upper_r = recall1
                upper_p = precision1
                break
            lower_p = precision1
            lower_r = recall1
        # else:
        #     fp += 1
        # precision1 = tp / c
        # recall1 = tp / len(relevant)
        # print('precision :', precision1)
        # print('recall :', recall1)
        # input()
        
            
        
    # print (lower_r, lower_p, upper_r, upper_p, recall, c, len(relevant), tp)
    # input()
    int_var = interpolate(lower_r, lower_p, upper_r, upper_p, recall)
    return int_var  # TODO: implement

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    summ = 0
    for i in range(1,11):
        summ += precision_at(0.1*i,results,relevant)
    return (0.1*summ)  # TODO: implement

def calc_rank(results, relevant):
    rank = []
    for i in range(0,len(relevant)):
        c=1
        for j in range(0,len(results)):
            if results[j] == relevant[i]:
                rank.append(c)
                break
            c = c+1
    return rank

def norm_recall(results, relevant):
    r_Rel = len(relevant)
    n_N = len(results)
    rank = calc_rank(results, relevant)
    # print('rank :',rank)
    sum_rank = 0
    sum_i = 0
    for i in range(1,len(relevant)+1):
        sum_rank += rank[i-1]
        sum_i += i
    # print('sum_rank :',sum_rank)
    # print('sum_i :',sum_i)

    r_norm = 1 - ((sum_rank - sum_i)/(r_Rel*(n_N - r_Rel)))

    return r_norm  # TODO: implement

def norm_precision(results, relevant):
    r_Rel = len(relevant)
    n_N = len(results)
    rank = calc_rank(results, relevant)
    log_rank_list = []
    for i in range(0,len(rank)):
        l = np.log(rank[i])
        log_rank_list.append(l)
    log_i_list = []
    for i in range(0,len(relevant)):
        l = np.log(i+1)
        log_i_list.append(l)


    log_i, log_rank = 0, 0
    for i in range(0,len(relevant)):
        log_rank += log_rank_list[i]
        log_i += log_i_list[i]
    
    log_n = n_N * np.log(n_N)
    n_rel_log = (n_N - r_Rel) * np.log(n_N - r_Rel)
    log_rel = r_Rel * np.log(r_Rel)
    # print('log_rank :',log_rank)
    p_norm = 1 - ((log_rank - log_i)/(log_n - n_rel_log - log_rel))
    return p_norm  # TODO: implement


### Extensions

# TODO: put any extensions here


### Search

def experiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'boolean': compute_boolean,
        # 'tfidf': compute_tfidf,
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):

        # print(sim)

        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        # return  # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()


if __name__ == '__main__':
    experiment()