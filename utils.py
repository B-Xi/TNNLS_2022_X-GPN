from __future__ import print_function
import tensorflow as tf
import scipy.sparse as sp
import numpy as np

from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import matplotlib.pyplot as plt

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array((adj > 0).sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array((adj > 0).sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_crossentropy(preds, labels):
    return np.mean(-np.log(np.clip(preds[range(labels.shape[0]),[labels]], 1e-8, 1)))

def sparse_accuracy(preds, labels):
    return np.mean(np.equal(labels, np.argmax(preds, 1)))

def evaluate_preds(preds, labels, indices, epoch, NB_EPOCH, weight_dis):
    sample_weight = logsig((epoch - NB_EPOCH / 2) / NB_EPOCH * 10)
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        loss=sparse_crossentropy(preds[idx_split], y_split[idx_split])
        loss = loss - sample_weight*weight_dis*np.mean(np.sum(np.multiply(np.clip(preds,1e-8,1), np.log(np.clip(preds,1e-8,1))),axis=1))
        split_loss.append(loss)
        split_acc.append(sparse_accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    exp_x = np.exp(x)
    s=np.reshape(np.sum(exp_x,axis=1),(exp_x.shape[0],-1))
    s=np.repeat(s,exp_x.shape[1],axis=1)
    softmax_x = exp_x * (1/s)
    #softmax_x = exp_x / np.sum(exp_x,axis=1)
    return softmax_x

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keep_dims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keep_dims=True)
    dist = f_2 - 2 * tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1, 0])
    return dist

# discrimination based entropy loss (dis)
def dis_loss(labels, logits):
    #y_sm = tf.nn.softmax(logits)
    y_split=tf.clip_by_value(logits,1e-8,1)
    mean_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_split, tf.log(y_split)), axis=1))
    return mean_loss

def eucliDist2(a, b):
    return np.sum(np.power((a - b), 2))

def normalize(X, axis=0):
    if axis == 1:
        for i in range(X.shape[0]):
            X[i] = (X[i] - X[i].min()) / (X[i].max() - X[i].min())
    else:
        for i in range(X.shape[1]):
            X[:, i] = (X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min())
    return X

def get_idx_train_val(labels, idx_train, num_classes, rate):
    val_idx = list()
    mask = np.ones((idx_train.shape[-1]), dtype=np.bool)
    for i in range(num_classes):
        idx_c = (np.where(np.argmax(labels, 1) == i))[0]
        num_c = idx_c.shape[0]
        idx=np.arange(num_c)
        np.random.shuffle(idx)
        val_idx.append(idx_c[idx[:round(num_c * rate)]])#random 10%
        #val_idx.append(idx_c[-round(num_c * rate):])#%the last 10%
    idx_val = np.concatenate(val_idx)
    mask[idx_val] = False
    return idx_train[(mask.tolist())], idx_val

# def get_idx_train_val(labels, idx_train, num_classes, rate):
#     # random 10%
#     permutation = np.random.permutation(idx_train.shape[0])
#     idx_val= permutation[0:round(idx_train.shape[0]*rate)]
#     mask = np.ones((idx_train.shape[-1]), dtype=np.bool)
#     mask[idx_val] = False
#     return idx_train[(mask.tolist())], idx_val

# logsig function
def logsig(x):
    return 1. / (1 + np.exp(-x))

def Get01Mat(mat1):
    [r, c] = np.shape(mat1)
    mat_01 = np.zeros([r, c])
    pos1 = np.argwhere(mat1 != 0)
    mat_01[pos1[:, 0], pos1[:, 1]] = 1
    return np.array(mat_01, dtype='float32')

def generate_map(prediction,idx,gt):
    maps=gt.reshape(gt.shape[0]*gt.shape[1],)
    labeled_loc = np.squeeze(np.array(np.where(maps>0)),axis=0)
    tr_test = maps[labeled_loc]
    tr_test[idx] = prediction
    maps[labeled_loc] = tr_test
    maps.reshape(gt.shape[0], gt.shape[1])
    return maps
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines   
    #ID=6:KSC
    #ID=7:Houston
    global palette
    global row
    global col
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 145
        col = 145
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])
        palette = palette*1.0/255  

    elif imageID == 6:
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette*1.0/255

    elif imageID == 7:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette*1.0/255
    elif imageID == 8:
        row = 601
        col = 2384
        palette = np.array([[0,208,0],
                            [128,255,0],
                            [50,160,100],
                            [0,143,0],
                            [0,76,0],
                            [160,80,40],
                            [0,236,236],
                            [255,255,255],
                            [216,191,216],
                            [255,0,0],
                            [192,180,170],
                            [114,133,124],
                            [170,0,0],
                            [80,0,0],
                            [237,164,24],
                            [255,255,0],
                            [250,190,21],
                            [245,0,245],
                            [0,0,236],
                            [179,197,222]])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
 

