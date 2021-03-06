# Pointer network for TSP
# uses graph as input 

import numpy as np
from keras.models import Model
from keras.layers import LSTM, Input, Dense
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop
from keras import initializations

TSP_size = 10
hidden_size = 256

## Please download the dataset here: http://goo.gl/NDcOIG and put in tsp_data

# Input data loader
def get_TSP_data(TSP_size,mode):
	#TSP_size: size of TSP
	#mode: train/test
	mode = "_"+mode+"_exact.txt"
	#mode = "_"+mode+".txt"
	num_lines = 0
	with open("tsp_data/tsp_"+str(TSP_size)+mode) as infile:
		for line in infile:
			num_lines += 1
	x_train = np.zeros((num_lines,TSP_size,TSP_size))   
	y_train = np.zeros((num_lines,TSP_size,TSP_size))
	count = 0
	with open("tsp_data/tsp_"+str(TSP_size)+mode) as infile:
		for line in infile:
			temp= line.split(' output ')
			x_train_temp = np.array(temp[0].split(),dtype=np.float).reshape(-1,2)
			for i in range(TSP_size):
				#temp_dis = np.exp(-(np.linalg.norm(x_train_temp - x_train_temp[i,:],axis=1)))
				temp_dis = ((np.linalg.norm(x_train_temp - x_train_temp[i,:],axis=1)))
				temp_dis[i] = 1
				x_train[count,i,:] = temp_dis
			y_label_temp = np.array(temp[1].split()[1:],dtype=np.float) - 1 
			y_train[count,:,:] = to_categorical(y_label_temp)
			count += 1
	return x_train,y_train		    	

def get_path_loss(x,ind):
	err = 0
	ind = np.concatenate(([0], ind))
	for i in range(x.shape[0]-1):
		err = err + x[ind[i],ind[i+1]]
	#err = err + np.sqrt(np.sum((x[0,:] - x[-1,:]) ** 2))	# last city to first city
	return err	

x_train,y_train = get_TSP_data(TSP_size,'train')


def my_init(shape, name=None):
    return initializations.uniform(shape, scale=0.08, name=name)

main_input = Input( shape=(TSP_size, TSP_size), name='main_input' )
encoder = LSTM(output_dim = hidden_size,return_sequences = True,name="encoder1")(main_input)
#encoder = Dense(output_dim=hidden_size)(main_input)
#encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder2")(encoder)
#decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size,name="decoder")(encoder)

model = Model( input=main_input, output=decoder )

#model.summary()
checkpointer = ModelCheckpoint(filepath='./mlp_test1.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
lr=0.01

for i in range(0,2):
	sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True,clipvalue=2)
	rms = RMSprop()
	#adam = Adam()
	model.compile(optimizer=rms,loss='categorical_crossentropy',metrics=['accuracy'])
	#if i>0:
	#model.load_weights('./mlp_DNN_segmentation_basic.h5')
	model.fit( x_train, y_train, validation_split=0.1,nb_epoch = 100, batch_size = 512,callbacks=[early_stopping])	
	lr=lr/2

pred =model.predict(x_train,batch_size=512,verbose=True)
err_true = 0
err_pred = 0
for i in range(1000):
	x = x_train[i,:,:]
	ind_true = np.argmax(y_train[i,:,:],axis=1)
	result = beam_search_decoder(pred[i,:,:], 3)
	ind_pred = result[0][0]
	err_true += get_path_loss(x,ind_true)
	err_pred += get_path_loss(x,ind_pred)  

### debug
N = 1000
pred =model.predict(x_train[:N,:,:],batch_size=512,verbose=True)
err_true = np.zeros((N,1))
err_pred = np.zeros((N,1))
for i in range(1000):
	x = x_train[i,:,:]
	ind_true = np.argmax(y_train[i,:,:],axis=1)
	result = beam_search_decoder(pred[i,:,:], 3)
	ind_pred = result[0][0]
	err_true[i,:] = get_path_loss(x,ind_true)
	err_pred[i,:] = get_path_loss(x,ind_pred) 

# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	for row in data:
		all_candidates = list()
		for i in range(len(sequences)):
			seq, score = sequences[i]
			#for j in range(len(row)):
			for j in np.setdiff1d(np.arange(len(row)),seq):	
				candidate = [seq + [j], score - np.log(row[j])]
				all_candidates.append(candidate)
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		sequences = ordered[:k]
	return sequences

x_test,y_test = get_TSP_data(TSP_size,'test')
pred =model.predict(x_test,batch_size=512,verbose=True)
err_true = 0
err_pred = 0
for i in range(10000):
	x = x_test2[i,:,:]
	ind_true = np.argmax(y_test[i,:,:],axis=1)
	result = beam_search_decoder(pred[i,:,:], 3)
	ind_pred = result[0][0]
	err_true += get_path_loss(x,ind_true)
	err_pred += get_path_loss(x,ind_pred)  
