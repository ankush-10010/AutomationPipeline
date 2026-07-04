import math
from collections import Counter

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        nd = {}
        num_doc = 0
        for doc in corpus:
            self.doc_len.append(len(doc))
            num_doc += len(doc)
            
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            
            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
                
        self.avgdl = num_doc / self.corpus_size if self.corpus_size > 0 else 0
        
        for word, freq in nd.items():
            idf_val = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf_val
            
    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (num / den)
        return scores

def reciprocal_rank_fusion(dense_scores, sparse_scores, k=60):
    dense_ranked = sorted(enumerate(dense_scores), key=lambda x: x[1], reverse=True)
    sparse_ranked = sorted(enumerate(sparse_scores), key=lambda x: x[1], reverse=True)
    
    rrf_scores = [0.0] * len(dense_scores)
    
    for rank, (clip_idx, _) in enumerate(dense_ranked):
        rrf_scores[clip_idx] += 1.0 / (k + rank + 1)
        
    for rank, (clip_idx, _) in enumerate(sparse_ranked):
        rrf_scores[clip_idx] += 1.0 / (k + rank + 1)
        
    return rrf_scores
