from __future__ import print_function
import scipy.io as sio
import keras.backend as K
from keras.layers import Input,Dense, Dropout, Lambda, Conv1D, concatenate,Add,Average, UpSampling1D, Flatten, AveragePooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from layers.centers import Centers
from layers.graph import GraphConvolution
from utils import *
from data_create import data_load_test
import time
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

# class Added_Weights(Layer):
#     def __init__(self,activation=None, **kwargs):
#         super(Added_Weights, self).__init__(**kwargs)
#         self.activation = activations.get(activation)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(3,1),
#                                       initializer = keras.initializers.Constant(value=0.33333),
#                                       trainable=True)
#         super(Added_Weights, self).build(input_shape)
#
#     def call(self, x, **kwargs):
#         # Implicit broadcasting occurs here.
#         # Shape x: (BATCH_SIZE, N, M)
#         # Shape kernel: (N, M)
#         # Shape output: (BATCH_SIZE, N, M)
#
#         return self.activation(x[0]*self.kernel[0]+x[1]*self.kernel[1]+x[2]*self.kernel[2])#(1-self.kernel[0]-self.kernel[1]))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#-------------------------------------------------------------------------
# Define parameters
#ID=1:Pavia University
#ID=2:Indian Pines   
#ID=6:KSC
#ID=7:Houston        
#DATASETS = {'Pavia':'UP','Indian Pines':'IP2018','KSC':'KSC','Houston':'HU2012',}
datasets=['','UP','IP2018','','','','KSC','HU2012']
ID=2
FILTER = 'chebyshev'  # 'chebyshev' 'localpool'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 1000
PATIENCE = 100  # early stopping patience
#-------------------------------------------------------------------------
# Model define
MULTI = True  # use Multi-scale (True) vs. Single_scale (False)
DCE_FLAG = True # use DCE_loss (True) vs. Cross_entropy_loss (False)
num_scale = 3  # Default numbers of scale
#-------------------------------------------------------------------------
# Training parameters
weight_dis = 1
T = 10
#-------------------------------------------------------------------------
# Random seed set
seed = 123
np.random.seed(seed)
tf.random.set_random_seed(seed)
#-------------------------------------------------------------------------

# Get data
X, A, y, y_train, y_test, idx_train, idx_test, train_mask = data_load_test(0.1,num_scale=num_scale,dataset=datasets[ID])
classNum = y.max()+1
idx_train, idx_val = get_idx_train_val(y_train[idx_train], np.array(idx_train), classNum, 0.1)  # 获取idx_train、idx_val
y_val = np.zeros((y_train.shape[0], classNum))
y_val[idx_val] = y_train[idx_val]
y_train[idx_val] = np.zeros((idx_val.shape[0], classNum))
train_mask[idx_val] = False
# Normalize X
X /= X.sum(1).reshape(-1, 1)
#-------------------------------------------------------------------------
# build model's inputs
if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    G = []
    graph = []
    support = 1
    for idx_scale in range(num_scale):
        A_ = preprocess_adj(A[idx_scale], SYM_NORM)
        graph += [A_]
        G += [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    G = []
    graph = []
    support = MAX_DEGREE + 1
    for idx_scale in range(num_scale):
        L = normalized_laplacian(A[idx_scale], SYM_NORM)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
        graph += T_k
        G += [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]
else:
    raise Exception('Invalid filter type.')

X_in = Input(shape=(X.shape[1],))
#-------------------------------------------------------------------------
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
def MGCN(X_in,G,num_scale,DCE_FLAG):
    Y=[]
    units1=20
    units2=16
    if DCE_FLAG==False:
        units1=32
        units2=classNum
    for i in range(num_scale):
        H = GraphConvolution(units1, support, normalization=True, activation='relu')([X_in] + G[i * support:(i + 1) * support])
        # H = Dropout(0.1)(H)#can be modified CE:0.5 DCE:0.1
        Encg=GraphConvolution(units2, support,normalization=True, activation='relu')([H] + G[i * support:(i + 1) * support])
        Encg = Dropout(0.2)(Encg)  # can be modified CE:0.5 DCE:0.10
        Y += [Encg]
    if DCE_FLAG:
        ADD = Lambda(lambda x:x[0]+x[1]+x[2])(Y)
        output = Centers(units=classNum, T=T, name='logits', activation='softmax')(ADD)
    else:
        output = Lambda(lambda x:tf.nn.softmax(x[0]+x[1]+x[2]))(Y)
    return output
def MGCN_Cross_Scale(X_in,G,num_scale,DCE_FLAG):#Epoch:450
    Y=[]
    Lms=[]
    Convs=[]
    units2=16
    if DCE_FLAG==False:
        units2=classNum
    for i in range(num_scale):
        H = GraphConvolution(32, support,normalization=True, activation='relu')([X_in] + G[i * support:(i + 1) * support])
        # H = Dropout(0.2)(H)#can be modified 0.5 32-20-32 0.8904 0.5 25-16-9 0.886# 0.1 0.8968 0.2 0.9099
        Lm=Lambda(lambda x:K.expand_dims(x,axis=-1))(H)
        Lms+=[Lm]
        Convs+=[Conv1D(1, 3, padding='same', activation='relu')(Lm)]#sigmoid
    for i in range(num_scale):
        concate1=[Lms[i],Convs[(i+1)%num_scale], Convs[(i+2)%num_scale]]
        Enc=concatenate(concate1,axis=1)
        # Enc = Add()(concate1)
        # Enc = Average()(concate1)
        Enc=Lambda(lambda x:tf.squeeze(x,axis=-1))(Enc)
        Encg= GraphConvolution(units2, support,normalization=True, activation='relu')([Enc] + G[i * support:(i + 1) * support])#,activation='sigmoid'
        Encg = Dropout(0.2)(Encg)#参数可调 0.2 32-20-32 0.8904 0.2 25-16-9 0.886#0.1 0.8968 0.1 0.9099
        #Encg = Graph_Wise_Normalization(support)([Encg]+G[i * support:(i + 1) * support])
        Y += [Encg]
    if DCE_FLAG:
        ADD = Lambda(lambda x:x[0]+x[1]+x[2])(Y)
        output = Centers(units=classNum, T=T, name='logits', activation='softmax')(ADD)
    else:
        output = Lambda(lambda x:tf.nn.softmax(x[0]+x[1]+x[2]))(Y)
    return output

def MGCN_Attention(X_in,G,num_scale,DCE_FLAG):
    Y=[]
    units2=16
    if DCE_FLAG==False:
        units2=classNum
    for i in range(num_scale):
        H = GraphConvolution(32, support,normalization=True, activation='relu')([X_in] + G[i * support:(i + 1) * support])
        #H = Dropout(0.2)(H)#can be modified
        H = GraphConvolution(units2, support,normalization=True, activation='relu')([H] + G[i * support:(i + 1) * support])#,activation='sigmoid'
        H = Dropout(0.2)(H)#can be modified
        Y+=[Lambda(lambda x:K.expand_dims(x,axis=-1))(H)]

    concate2 = concatenate(Y, axis = 2)
    AP1=AveragePooling1D(units2)(concate2)
    AP1=Lambda(lambda x:tf.squeeze(x,axis=-2),name='sq')(AP1)
    D1=Dense(10,activation='relu',kernel_initializer='random_uniform', use_bias=False)(AP1)
    D2=Dense(num_scale,activation='sigmoid',kernel_initializer='random_uniform', use_bias=False)(D1)
    Att=Lambda(lambda x:K.expand_dims(x,axis=-2),name='Att')(D2)
    APL = Lambda(lambda x :tf.squeeze(tf.matmul(1+x[0], x[1], transpose_b=True),axis=-2),name='APL')([Att,concate2])

    if DCE_FLAG:
        output = Centers(units=classNum, T=T, name='logits', activation='softmax')(APL)
    else:
        output = Lambda(lambda x:tf.nn.softmax(tf.squeeze(tf.matmul(1+x[0], x[1], transpose_b=True),axis=-2)),name='APL')([Att,concate2])
    return output

def MGCN_Cross_Scale_Attention(X_in,G,num_scale,DCE_FLAG):
    Y=[]
    Lms=[]
    Convs=[]
    units1 = 16
    units2=16
    if DCE_FLAG==False:
        units2=int(classNum)
    for i in range(num_scale):
        H = GraphConvolution(units1, support,normalization=True, activation='relu')([X_in] + G[i * support:(i + 1) * support])
        # H = Dropout(0.2)(H)#can be modified
        Lm=Lambda(lambda x:K.expand_dims(x,axis=-1))(H)
        Lms+=[Lm]
        Convs+=[Conv1D(1, 3, padding='same', activation='relu')(Lm)]#sigmoid
    for i in range(num_scale):
        concate1=[Lms[i],Convs[(i+1)%num_scale], Convs[(i+2)%num_scale]]
        Enc=concatenate(concate1,axis=1)
        Enc=Lambda(lambda x:tf.squeeze(x,axis=-1))(Enc)
        Encg= GraphConvolution(units2, support,normalization=True, activation='relu')([Enc] + G[i * support:(i + 1) * support])#,activation='sigmoid'
        Encg = Dropout(0.2)(Encg)#can be modified
        Y += [Lambda(lambda x:K.expand_dims(x,axis=-1))(Encg)]
    concate2 = concatenate(Y, axis = 2)
    AP1=AveragePooling1D(units2)(concate2)
    AP1=Lambda(lambda x:tf.squeeze(x,axis=-2),name='sq')(AP1)
    D1=Dense(10,activation='relu',kernel_initializer='random_uniform',use_bias=False)(AP1)
    D2=Dense(num_scale,activation='sigmoid',kernel_initializer='random_uniform',use_bias=False)(D1)
    Att=Lambda(lambda x:K.expand_dims(x,axis=-2),name='Att')(D2)
    APL = Lambda(lambda x:tf.squeeze(tf.matmul(1+x[0], x[1], transpose_b=True),axis=-2),name='APL')([Att,concate2])
    if DCE_FLAG:
        output = Centers(units=classNum,T=T,#regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001),
                         #initializer=keras.initializers.Orthogonal(gain=1.0, seed=None),
                             name='logits', activation='softmax')(APL)
    else:
        output = Lambda(lambda x:tf.nn.softmax(tf.squeeze(tf.matmul(1+x[0], x[1], transpose_b=True),axis=-2)),name='APL')([Att,concate2])
    return output

#-------------------------------------------------------------------------
#the output of the model
if MULTI:
    output = MGCN_Cross_Scale_Attention(X_in,G,num_scale,DCE_FLAG)
else:
    scale = 1   # '0':3*3  '1':5*5  '2':7*7
    if DCE_FLAG:
        output = Centers(units=classNum, T=T, name='logits', activation='softmax')(Y[scale])

    else:
        H = GraphConvolution(32, support, activation='relu')([X_in] + G[scale * support:(scale + 1) * support])
        H=Dropout(0.5)(H)
        output = GraphConvolution(classNum, support, activation='softmax')(
            [H] + G[scale * support:(scale + 1) * support])
#-------------------------------------------------------------------------

# IF ONEHOT DECODER
y_train=np.argmax(y_train,1)
y_test=np.argmax(y_test,1)
y_val=np.argmax(y_val,1)

# Compile model
model = Model(inputs=[X_in]+G, outputs=([output,output]))
model.summary()
model.compile(loss=(['sparse_categorical_crossentropy',dis_loss]),loss_weights=([1.0, weight_dis]), optimizer=Adam(lr=0.01))
#'sparse_categorical_crossentropy'[K.sparse_categorical_crossentropy]
# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999
print("-----------------------------------")
print("Train samples: {:d}".format(idx_train.shape[0]),
      "\nValidate samples: {:d}".format(idx_val.shape[0]),
      "\nTest samples: {:d}".format(idx_test.stop-idx_test.start))
print("-----------------------------------")
# Fit
f1=open('result/'+datasets[ID]+'/train_log.txt','w')
train_t = time.time()
for epoch in range(1, NB_EPOCH+1):
    # Log wall-clock time
    t = time.time()
    sample_weights = logsig((np.ones(y_train.shape[0], dtype='float32') - 1 + epoch-1 - NB_EPOCH / 2) / NB_EPOCH * 10)
    # Single training iteration (we mask nodes without labels for loss calculation)
    model.fit([X] + graph, [y_train ,y_train], sample_weight=([train_mask,sample_weights]),#, np.ones((y_train.shape[0]))
              batch_size=A[0].shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds,_ = model.predict([X]+graph, batch_size=A[0].shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val], epoch-1, NB_EPOCH, weight_dis)
    f1.write(str("Epoch: {:04d}".format(epoch)+
              " train_loss= {:.4f} ".format(train_val_loss[0])+
              " train_acc= {:.4f} ".format(train_val_acc[0])+
              " val_loss= {:.4f} ".format(train_val_loss[1])+
              " val_acc= {:.4f} ".format(train_val_acc[1])+
              " time= {:.4f} ".format(time.time() - t))+'\n')
    #if epoch % 50 == 0:
    print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))
    if epoch % 100 == 0:
        test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test], NB_EPOCH - 1, NB_EPOCH, weight_dis)
        print("Epoch: {:04d}".format(epoch),
              "Test set results:",
              "loss= {:.4f}".format(test_loss[0]),
              "accuracy= {:.4f}".format(test_acc[0]))
        f1.write(str("Epoch: {:04d} ".format(epoch)+
                      " Test set results:"+
                      " loss= {:.4f} ".format(test_loss[0])+
                      " accuracy= {:.4f} ".format(test_acc[0])))
        model.save_weights('result/' + datasets[ID] + '/model_weights' + str(epoch) + '.h5')
        # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        model.save_weights('result/'+datasets[ID]+'/model_weights.h5')
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            wait=0
            if K.get_value(model.optimizer.lr)>1e-4:
                K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)*(1.0))# 学习率衰减
            else:
                break
        wait += 1  
f1.close()
# model.save('model.h5')
model.load_weights('result/'+datasets[ID]+'/model_weights.h5')
training_time = time.time() - train_t
print("Training time =", str(time.time() - train_t))
# Testing
test_t = time.time()
preds,_ = model.predict([X]+graph, batch_size=A[0].shape[0])
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test], NB_EPOCH-1, NB_EPOCH, weight_dis)
test_time = time.time() - test_t
print("Testing time =", str(time.time() - test_t))

print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))

matrix = np.zeros((classNum, classNum))
data = np.argmax(preds[idx_test], 1)
test_labels = y_test[idx_test]
n = (idx_test.__len__())
with open('result/'+datasets[ID]+'/prediction.txt', 'w') as f:
    for i in range(n):
        pre_label = int(data[i])
        f.write(str(pre_label)+'\n')
        matrix[pre_label][test_labels[i]] += 1
np.savetxt('result/'+datasets[ID]+'/result_matrix.txt', matrix, fmt='%d', delimiter=',')
print(''+str(np.int_(matrix)))
print(np.sum(np.trace(matrix)))

OA = np.sum(np.trace(matrix)) / float(n)
# print('OA = '+str(OA)+'\n')
ua = np.diag(matrix)/np.sum(matrix, axis=0)
precision = np.diag(matrix)/np.sum(matrix, axis=1)
matrix = np.mat(matrix)
Po = OA
xsum = np.sum(matrix, axis=1)
ysum = np.sum(matrix, axis=0)
Pe = float(ysum*xsum)/(np.sum(matrix)**2)
Kappa = float((Po-Pe)/(1-Pe))

# print('ua =')
# print('precision =')
for i in range(classNum):
    print(ua[i])
print(np.sum(ua)/matrix.shape[0])
print(OA)
print(Kappa)

print()
for i in range(classNum):
    print(precision[i])
print(np.sum(precision)/matrix.shape[0])
print(training_time)
print(test_time)

f.close()


idx_test = sio.loadmat('data/'+datasets[ID]+'/idx_test.mat')["idx_test"]
gt = sio.loadmat('data/'+datasets[ID]+'/Gt.mat')['groundtruth']
resultpath = 'result/'+datasets[ID]+'/'#+datasets[ID]+'/'
pred_map=generate_map(data+1,idx_test,gt)

sio.savemat(resultpath+'pred_map.mat', {'pred_map': pred_map})

plt.figure()
img_gt = DrawResult(gt.reshape(pred_map.shape[0],1),ID)
plt.figure()
img = DrawResult(pred_map,ID)
plt.imsave(resultpath+'GCN'+'_'+repr(int(OA*10000))+'.png',img)
plt.show()