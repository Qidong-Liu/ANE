import os

####parameters for learning
iter = 1
workers = 8
seed = 0

la = 1.

###parameters for random walks
walk_length = 40
num_walks = 10
window_size = 10
negatives = 5
FEATURE_SIZE = 128              #dimensions

verbose = False

App = 1                         #0 denotes classification; 1 denotes link prediction
CP = False                      #Continue training from the checkpoint

workdir = os.path.abspath('.')  #workspace

######Classification Input
TRAIN_INPUT_FILENAME = workdir + "/data/blogcatalog/cf/blogcatalog_input2.txt"
TRAIN_LABEL_FILENAME = workdir + "/data/blogcatalog/cf/blogcatalog_label.txt"
format = 'cora'

######Link prediction Input
split = False
test_frac = 0.2
FULL_FILENAME = workdir + '/data/wiki-vote.txt'

TRAIN_POS_FILENAME = workdir + '/data/wiki-vote/wiki_train_pos.txt'
TRAIN_NEG_FILENAME = workdir + '/data/wiki-vote/wiki_train_neg.txt'

TEST_POS_FILENAME = workdir + "/data/wiki-vote/wiki_test_pos.txt"
TEST_NEG_FILENAME = workdir + "/data/wiki-vote/wiki_test_neg.txt"

#####Output
EMB_OUTPUT_FILENAME = workdir + "/emb/emb.txt"
MODEL_FILE = workdir + '/model/ANE.model'

