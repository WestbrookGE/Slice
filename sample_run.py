from xclib.evaluation import xc_metrics
from xclib.data import data_utils
import slice_lib

dir_name = './data'
score_dir = './score'
train_X, train_labels, num_examples, num_features, num_labels = data_utils.read_data(dir_name+'train.txt')
test_X, test_labels, num_examples, num_features, num_labels = data_utils.read_data(dir_name+'test.txt')
train_labels, test_labels = slice_lib.clean_data(train_labels, test_labels)

pre_mat = slice_lib.train_predict(train_X, train_labels, test_X, test_labels, save_path = ' ',k1 = 300, k2 = 300, num_threads = 1)

mask = (pre_mat > 0.1)
pre_cut = pre_mat.multiply(mask)
data_utils.write_sparse_file(pre_cut, score_dir+'score.txt')
