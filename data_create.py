import numpy as np
import scipy.io as sio
import keras
from scipy import sparse
from utils import eucliDist2, normalize
import time

def calA(num_scale,alpha,row,col,num_nodes,pos1D,gt1D,x_all):
    adj = []
    for scl_idx in range(1,num_scale+1):
        scl=(2*scl_idx+1)
        x = np.arange(scl * scl) - int((scl ** 2 - 1) / 2)
        x = np.round(x / scl).reshape((scl, scl))
        idx_tmp0 = np.array([x.flatten(), x.T.flatten()])
        idx_tmp0 = np.delete(idx_tmp0, int(((scl ** 2 - 1) / 2)), axis=1)#mask of the scope of scl*scl
        rcd=0
        data=[]
        indices=[]
        indptr=[0]
        print('\rCreating {:d} of {:d} A matrix ...'.format(scl_idx,num_scale))
        index_pos1D=np.zeros(row*col)
        for i in range(num_nodes):
            index_pos1D[pos1D[i]]=i
        t = time.time()
        for i in range(num_nodes):
            pos_i = [(int)(pos1D[i] / col), pos1D[i] % col]
            idx_tmp = np.array([(pos_i[0] + idx_tmp0[0]), pos_i[1] + idx_tmp0[1]])
            # judge if the neighboring coordinate is legal
            idx_mask = np.array(
                (idx_tmp >= 0)[0] * (idx_tmp >= 0)[1] * (idx_tmp[0] < row) * (idx_tmp[1] < col))
            idx_tmp = np.squeeze(idx_tmp[:, np.where(idx_mask)], axis=1)#delete the illegal coordinates
            idx_tmp = np.transpose(idx_tmp)
            idx_tmp1D = idx_tmp[0] * col + idx_tmp[1]#transform the coordinates to 1D form
            A_i = np.zeros(num_nodes)
            tps=[]
            nei_idx=[]
            for j in range(idx_tmp.shape[0]):
                #check that if the neighboring pixels are in pos1D,
                #and calculate the distance between them and the central pixel
                x1=idx_tmp[j]
                x1D=int(x1[0]*col+x1[1])
                if gt1D[x1D]!=0:
                    tp = int(index_pos1D[x1D])
                    if alpha==0:
                        alpha=1
                    A_i[tp] = np.exp(-alpha*eucliDist2(x_all[i], x_all[tp]))
                    tps+=[tp]
                    nei_idx+=[j]
            tps=np.array(tps)
            nei_idx=np.array(nei_idx)
            #obtain the non-zero-pixel
            if len(tps)==0:
                print('warning: {:d} of lines is empty!'.format(i))
                indptr+=[indptr[-1]]
            else:
                pos=np.argsort(A_i[tps])
                nzero_num=len(tps)
                if nzero_num==0:
                    print('\r line {:d} is wrong..'.format(i))
                pos_in=pos[-nzero_num:]
                pos_out=pos[-nzero_num:-nzero_num]
                A_i[tps[pos_out]]=0
                #update the sparse matrix
                indices+=np.sort(tps[pos_in]).tolist()
                data+=(A_i[np.sort(tps[pos_in])].tolist())
                indptr+=[indptr[-1]+pos_in.shape[0]]
            #Dynamic output generation progress
            if int(i/num_nodes*100)!=rcd:
                rcd=int(i/num_nodes*100)
                print("\r[{0}{1}]->{2:02d}% {3:.2f}s".format('>'*round(rcd/2),'-'*round((100-rcd)/2),rcd,time.time()-t),end='',flush=True)
        print("\r[{0}{1}]->{2:02d}% {3:.2f}s".format('>'*50,'-'*0,100,time.time()-t),end='\n',flush=True)
        adj+=[sparse.csr_matrix((np.array(data),np.array(indices),np.array(indptr)),shape=(num_nodes,num_nodes))]  # The matrix is compressed in a row-first fashion
    return adj

def data_load_test(alpha,num_scale=1,path=u'data/', dataset=''):
    x_all = sio.loadmat(path+dataset+'/data.mat')['spectral_data']
    y_te = sio.loadmat(path+dataset+'/testingMap.mat')['testingMap']
    y_tr = sio.loadmat(path+dataset+'/trainingMap.mat')['trainingMap']
    y_all = y_te+y_tr #sio.loadmat('data/UP/PaviaU_gt.mat')['paviaU_gt']
    
    row,col = y_all.shape
    num_pixels = y_all.shape[0] * y_all.shape[1]
    num_classes = y_all.max()
    x_all = x_all.reshape(num_pixels, x_all.shape[2])
    y_all = y_all.reshape(num_pixels, )
    gt1D = y_all
    pos1D0=np.arange(num_pixels)
    
    # normaalization
    x_all = x_all.astype(float)
    x_all = normalize(x_all, axis=0)

    # delete 0 pixels
    idx = np.where(y_all>0)
    num_nodes = idx[0].shape[0]
    x_all = x_all[idx]
    y_all = y_all[idx]
    pos1D = pos1D0[idx] #deleted 0 pixels 1D position
    # s = s[idx]
    y_all -= 1
    y_tr = y_tr.reshape(num_pixels,)[idx]
    y_te = y_te.reshape(num_pixels,)[idx]

    ## generate training data
    ## --generate the index of training and test data
    idx_train = np.squeeze(np.array(np.where(y_tr>0)),axis=0)
    idx_test = np.squeeze(np.array(np.where(y_te>0)),axis=0)
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_test)
    sio.savemat(path+dataset+'/idx_test.mat',
                  {"idx_test": idx_test})
    
    idx_all=np.squeeze(np.hstack((idx_train, idx_test)))
    sio.savemat(path+dataset+'/sort_shuffle_idx_all.mat',
                  {"idx_all": idx_all})
    idx_all = sio.loadmat(path+dataset+'/sort_shuffle_idx_all.mat')["idx_all"]
    idx_all = np.squeeze(idx_all, axis=0)
    idx_train = idx_all[0:np.where(y_tr > 0)[0].shape[0]]
    idx_test = idx_all[np.where(y_tr > 0)[0].shape[0]:]
    y_train = np.zeros((num_nodes, num_classes))
    y_train[0:idx_train.shape[0]] = keras.utils.to_categorical(y_all[idx_train], num_classes)
    y_test = np.zeros((num_nodes, num_classes))
    y_test[idx_train.shape[0]:] = keras.utils.to_categorical(y_all[idx_test], num_classes)
    train_mask = np.zeros((num_nodes))
    train_mask[0:idx_train.shape[0]] = 1
    x_all = x_all[idx_all]
    y_all = y_all[idx_all]
    pos1D = pos1D[idx_all]
    adj =calA(num_scale,alpha*0.2,row,col,num_nodes,pos1D,gt1D,x_all)
    sio.savemat(path+dataset+'/sort_shuffle_A.mat', {"adj": adj})
    # the "adj" can be loaded but not calculate next time
    #A1 = sio.loadmat(path+dataset+'/sort_shuffle_A.mat')['adj']
    # adj=[]
    # for x in range(num_scale):
    #     adj_org=A1[0,x]
    #     adj_org.data = adj_org.data + adj_org.data.T * (adj_org.data.T > adj_org.data) - adj_org.data * (adj_org.data.T > adj_org.data)
    #     adj += [adj_org.tocsr() ]
    return np.mat(x_all), adj, y_all, y_train, y_test, range(idx_train.shape[0]), range(idx_train.shape[0], num_nodes),train_mask.astype(bool)