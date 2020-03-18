import random
import numpy as np
import utils as ut
import networkx as nx
import settings as st
import matplotlib.pyplot as plt

from eval.auc import AUC
from gensim.models import Word2Vec
from eval.multilabel_class_cv import Cross_val

class ANE(object):
    def __init__(self):
        if st.App == 0:
            self.G = nx.read_edgelist(st.TRAIN_INPUT_FILENAME, nodetype=str, data=(('weight', float),), create_using=nx.Graph)
        else:
            self.G = nx.read_edgelist(st.TRAIN_POS_FILENAME, nodetype=str, data=(('weight', float),), create_using=nx.Graph)
            self.train_edges = np.loadtxt(st.TRAIN_POS_FILENAME, dtype=str)
            self.train_edges_false = np.loadtxt(st.TRAIN_NEG_FILENAME, dtype=str)
            self.test_edges = np.loadtxt(st.TEST_POS_FILENAME, dtype=str)
            self.test_edges_false = np.loadtxt(st.TEST_NEG_FILENAME, dtype=str)

        self.model = None
        self.build_model()
#        self.G.remove_edges_from(nx.selfloop_edges(self.G))

    def build_model(self):
        if st.CP:               #### continue training from the checkpoint
            self.model = Word2Vec.load(st.MODEL_FILE)
        else:                    #### start a new train
            self.model = None
    
    def ANE_walk(self, rand=random.Random(),start_node1=None, start_node2=None, E=0):
        walk = [start_node1, start_node2]

        while len(walk) < st.walk_length:
            pre = walk[-2]
            cur = walk[-1]
            Npre = set(self.G.neighbors(pre))
            Ncur = set(self.G.neighbors(cur))
            D = list(Ncur - Npre - {pre})
            J = list(Ncur & Npre)
            w = (1. + len(Npre)/E) * st.la
#            w = st.la
            denom = len(D) + len(J) * w
            if denom == 0:
                next = pre
            else:
                th = len(D)/denom
                if rand.random() < th:
                    next = rand.choice(D)
                else:
                    next = rand.choice(J)

            walk.append(next)

        return [str(node) for node in walk]

    def Deepwalk(self, rand=random.Random(), start_node1=None, start_node2=None):
        walk = [start_node1, start_node2]

        while len(walk) < st.walk_length:
            cur = walk[-1]
            temp = list(self.G.neighbors(cur))
            next = rand.choice(temp)
            walk.append(next)

        return [str(node) for node in walk]

    def show(self, walks, w):
        count = [0] * 5
        it = 0
        for walk in walks:
            i = 0
            it = it + 1
            while i < st.walk_length:
                cur = walk[i]
                for j in range(w * 2 + 1):
                    t = i + j - w
                    if t < 0 or t == i or t >= st.walk_length:
                        continue

                    node = walk[t]
                    order = nx.shortest_path_length(self.G, source=cur, target=node)
                    if order > 4:
                        order = 4

                    count[order] = count[order] + 1
                i = i + 1
        print('order 0: ', count[0])
        print('order 1: ', count[1])
        print('order 2: ', count[2])
        print('order 3: ', count[3])
        print('order 4: ', count[4])

    #        ind = np.arange(len(count))
    #        fig, ax = plt.subplots()
    #        rect = ax.bar(ind, count)
    #        plt.show()

    def show2(self, walks, w):
        count = [0] * 5
        it = 0
        for walk in walks:
            i = 0
            it = it + 1
            while i < st.walk_length:
                cur = walk[i]
                for j in range(w + 1):
                    t = i + j
                    if t < 0 or t == i or t >= st.walk_length:
                        continue

                    node = walk[t]
                    order = nx.shortest_path_length(self.G, source=cur, target=node)
                    if order > 4:
                        order = 4

                    count[order] = count[order] + 1
                i = i + 1
        ind = np.arange(len(count))
        fig, ax = plt.subplots()
        rect = ax.bar(ind, count)
        plt.show()

    def getCorpus(self, verbose=True):
        walks = []
        nodes = list(self.G.nodes())
        rand = random.Random(st.seed)
        E = self.G.size()
        for walk_iter in range(st.num_walks):
            print('walk iter: ', str(walk_iter + 1), '/', str(st.num_walks))
            rand.shuffle(nodes)
            for node1 in nodes:
                temp = list(self.G.neighbors(node1))
                if temp:
                    node2 = rand.choice(temp)
                    walks.append(self.ANE_walk(rand=rand, start_node1=node1, start_node2=node2, E=E))
                else:
                    walks.append([node1] * st.walk_length)
                    print(node1, 'not exists!')

        if verbose:
            self.show2(walks, st.window_size)
        
        return walks

    def evaluation(self):
        if st.App == 0:
            f1_micro, f1_macro = Cross_val(st.EMB_OUTPUT_FILENAME)
            print('F1 (micro) = {}'.format(f1_micro))
            print('F1 (macro) = {}'.format(f1_macro))
        elif st.App == 1:
            roc_score_train, ap_score_train = AUC(self.model.wv, self.train_edges, self.train_edges_false)
            roc_score_test, ap_score_test= AUC(self.model.wv, self.test_edges, self.test_edges_false)
            print("roc-test:", roc_score_test, 'ap-test', ap_score_test)
            print('roc-train', roc_score_train, 'ap-train', ap_score_train)
        else:
            print('please reset App as 0 or 1!')

    def train(self):
        walks = self.getCorpus(verbose=st.verbose)
        if st.CP:
            self.model.train(walks, total_examples=len(walks), epochs=st.iter)
        else:
            self.model = Word2Vec(walks, size=st.FEATURE_SIZE, window=st.window_size, min_count=0, sg=1, hs=0, workers=st.workers, iter=st.iter, negative=st.negatives)
        self.model.save(st.MODEL_FILE)
        self.model.wv.save_word2vec_format(st.EMB_OUTPUT_FILENAME)

if __name__ == '__main__':
    if st.split:
        ut.split_data(st.FULL_FILENAME, st.TRAIN_POS_FILENAME, st.TRAIN_NEG_FILENAME, st.TEST_POS_FILENAME, st.TEST_NEG_FILENAME, st.test_frac)
    ane = ANE()
    ane.train()
    ane.evaluation()

