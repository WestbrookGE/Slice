import numpy as np
from sklearn.preprocessing import normalize
import hnswlib
from tqdm import tqdm
from xclib.evaluation import xc_metrics
from xclib.data import data_utils
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data(dir_name):
    train_X = np.loadtxt(dir_name+'train_X.txt')
    test_X = np.loadtxt(dir_name+'test_X.txt')
    train_labels = data_utils.read_sparse_file(dir_name+'train_lbl.txt', dtype = 'int')
    test_labels = data_utils.read_sparse_file(dir_name+'test_lbl.txt', dtype = 'int')
    return train_X, train_labels, test_X, test_labels

def clean_data(train_labels, test_labels):
    print('Cleaning Label with None Positive Example')
    train_labels = train_labels.tocsc()
    test_labels = test_labels.tocsc()
    num_labels = train_labels.shape[1]
    label_counts = np.sum(train_labels, axis = 0)
    cut_idx = np.ravel((label_counts> 0.5))
    train_labels = train_labels[:, cut_idx]
    test_labels = test_labels[:, cut_idx]
    train_labels = train_labels.tocsr()
    test_labels = test_labels.tocsr()
    new_num = train_labels.shape[1]
    print('Label Num. %d to %d'%(num_labels, new_num))
    return train_labels, test_labels


def get_label_example_lists(train_labels):
    label_csc = train_labels.tocsc()
    le_lists = np.split(label_csc.indices, label_csc.indptr)[1:-1]
    return le_lists

def compute_label_vector(train_X, le_lists):
    (num_examples, num_features) = train_X.shape
    num_labels = len(le_lists)
    lb_vectors = np.zeros((num_labels ,num_features))
    for i in range(num_labels):
        for idx in le_lists[i]:
            lb_vectors[i, :] += train_X[idx, :]
    lb_vectors = normalize(lb_vectors, norm='l2')
    return lb_vectors

def build_lb_HNSW(lb_vectors, ef, ef_construction = 200, M = 16, num_threads = 2):
    (num_labels, num_features) = lb_vectors.shape
    p = hnswlib.Index(space = 'ip', dim = num_features) 
    p.set_num_threads(num_threads)
    p.init_index(max_elements = num_labels, ef_construction = ef_construction, M = M)
    p.add_items(lb_vectors, num_threads = -1)
    p.set_ef(ef)
    return p

def compute_example_label_NN(train_X, p, k):
    X = normalize(train_X, norm='l2')
    el_NNs, distances = p.knn_query(X, k=k)
    return el_NNs, distances

def get_label_example_NN(el_NNs, train_labels):
    (num_examples, k) = el_NNs.shape
    (num_examples, num_labels) = train_labels.shape
    le_NNs = [[] for i in range(num_labels)]
    for i in range(num_examples):
        for j in range(k):
            idx = el_NNs[i, j]
            le_NNs[idx].append(i)
    return le_NNs

def train_classifer(train_X, train_labels, le_NNs, le_lists):
    (num_examples, num_labels) = train_labels.shape
    clfs = []
    for i in tqdm(range(num_labels)):
        example_list = list(le_lists[i]) + le_NNs[i]
        example_list = list(set(example_list))
        if len(example_list) == len(le_lists[i]):
            clfs.append('None')
            continue
        X = train_X[example_list, :]
        y = np.ravel(train_labels[example_list, i].toarray())
        clf = LinearSVC(dual = False)
        clf.fit(X, y)
        clfs.append(clf)
    return clfs

def approximate_gamma(clfs):
    num_labels = len(clfs)
    gamma = 0
    count = 0
    for i in range(num_labels):
        if clfs[i] == 'None':
            continue
        gamma += np.linalg.norm(clfs[i].coef_)
        count += 1
    return gamma/count

def pre_train(train_X, train_labels, k1, num_threads):
    le_lists = get_label_example_lists(train_labels)
    print('Computing label vector')
    lb_vectors = compute_label_vector(train_X, le_lists)
    print('Building label graph')
    p = build_lb_HNSW(lb_vectors, k1, num_threads = num_threads)
    print('Graph Querying')
    el_NNs, _ = compute_example_label_NN(train_X, p, k1)
    print('Get Inverted Index')
    le_NNs = get_label_example_NN(el_NNs, train_labels)
    return le_NNs, le_lists, p

def compute_discrimintaive_score(test_X, clfs, el_NNs):
    num_examples = test_X.shape[0]
    num_labels = len(clfs)
    rows,cols,datas = [], [], []
    for i in tqdm(range(num_examples)):
        for lb in el_NNs[i]:
            if clfs[lb] == 'None':
                score = 0
            else:
                score = clfs[lb].decision_function(test_X[i,:].reshape(1,-1))[0]
                score = sigmoid(score)
            rows.append(i)
            cols.append(lb)
            datas.append(score)
    pre_mat = csr_matrix((datas, (rows, cols)), shape=(num_examples, num_labels))
    return pre_mat

def compute_generative_score(test_X, clfs, el_NNs, distances):
    gamma = approximate_gamma(clfs)
    num_examples = test_X.shape[0]
    num_labels = len(clfs)
    rows,cols,datas = [], [], []
    for i in tqdm(range(num_examples)):
        t = 0
        for lb in el_NNs[i]:
            if clfs[lb] == 'None':
                score = 0
            else:
                score = gamma * (1-distances[i, t])
                score = sigmoid(score)
            rows.append(i)
            cols.append(lb)
            datas.append(score)
            t += 1
    pre_mat = csr_matrix((datas, (rows, cols)), shape=(num_examples, num_labels))
    return pre_mat

def prediction(test_X, clfs, p, save_path, k2):
    print('Graph Querying')
    el_NNs, distances = compute_example_label_NN(test_X, p, k2)
    print('Compute Discrimintaive Score')
    dis_score = compute_discrimintaive_score(test_X, clfs, el_NNs)
    print('Compute Generative Score')
    gen_score = compute_generative_score(test_X, clfs, el_NNs, distances)
    return (dis_score+gen_score, dis_score)

def evaluate(pre_mat, test_labels, train_labels, ps_A= 0.55, ps_B = 1.5, save_path = ' '):
    inv_propen = xc_metrics.compute_inv_propesity(train_labels, ps_A, ps_B)
    acc = xc_metrics.Metrics(true_labels=test_labels, inv_psp=inv_propen)
    args = acc.eval(pre_mat, 5)
    '''
    if save_path != ' ':
        with open(save_path,'w') as f:
            f.write('prec:'+str(args[0])+'\n')
            f.write('ndcg:'+str(args[1])+'\n')
            f.write('PSprec:'+str(args[2])+'\n')
            f.write('PSnDCG:'+str(args[3])+'\n')
    '''
    print(args[0])

def train_predict(train_X, train_labels, test_X, test_labels, save_path = ' ',k1 = 300, k2 = 300, num_threads = 2):
    le_NNs, le_lists, p = pre_train(train_X, train_labels, k1, num_threads)
    print('Training classifer')
    clfs = train_classifer(train_X, train_labels, le_NNs, le_lists)
    pre_mat = prediction(test_X, clfs, p, save_path, k2)
    return pre_mat


