import numpy
import scipy.sparse

def loadData(filename):
	fData = numpy.loadtxt(filename, delimiter = ',')
	return(fData)

def loadTripleData(filename, nrow=0, ncol=0):
	'''
	laod triple data (row, column, value) to csc_matrix format
	'''
	fData = numpy.loadtxt(filename, delimiter=',')
	fData = fData.T

	num_row = max(int(fData[0].max())+1, nrow)
	num_col = max(int(fData[1].max())+1, ncol)
	fData = scipy.sparse.coo_matrix((fData[2],(fData[0],fData[1])), shape=(num_row, num_col))
	fData = scipy.sparse.csc_matrix(fData)
	return(fData)

def read_data(dataset_name):
	X_train = loadData('data/%s/train.txt' % (dataset_name))
	X_userFeat = loadData('data/%s/user.txt' % (dataset_name))
	X_itemFeat = loadData('data/%s/item.txt' % (dataset_name))
	X_test = loadData('data/%s/test.txt' % (dataset_name))

	Xs_trn = [X_train, X_userFeat, X_itemFeat]
	Xs_tst = [X_test, None, None]

	return [Xs_trn, Xs_tst]

def read_binary_data(dataset_name):
	X_train = loadData('data/%s/train.txt' % (dataset_name))
	X_userFeat = loadData('data/%s/user_binary.txt' % (dataset_name))
	X_itemFeat = loadData('data/%s/item_binary.txt' % (dataset_name))
	X_test = loadData('data/%s/test.txt' % (dataset_name))

	Xs_trn = [X_train, X_userFeat, X_itemFeat]
	Xs_tst = [X_test, None, None]

	return [Xs_trn, Xs_tst]

def read_triple_data(dataset_name):
	file_path = 'data/{}'.format(dataset_name)
	Dtrain = loadData('{}/train.txt'.format(file_path)).T
	Dtest = loadData('{}/test.txt'.format(file_path)).T

	# to avoid training and testing data with different shapes
	num_user = int(max(Dtrain[0].max(), Dtest[0].max())) + 1
	num_item = int(max(Dtrain[1].max(), Dtest[1].max())) + 1
	X_train = scipy.sparse.coo_matrix((Dtrain[2],(Dtrain[0],Dtrain[1])), shape=(num_user, num_item))
	X_test = scipy.sparse.coo_matrix((Dtest[2],(Dtest[0],Dtest[1])), shape=(num_user, num_item))
	# transform to csc format
	X_train = scipy.sparse.csc_matrix(X_train)
	X_test = scipy.sparse.csc_matrix(X_test)

	X_userFeat = loadTripleData('{}/user_binary.txt'.format(file_path), num_user, 0)
	X_itemFeat = loadTripleData('{}/item_binary.txt'.format(file_path), num_item, 0)

	Xs_trn = [X_train, X_userFeat, X_itemFeat]
	Xs_tst = [X_test, None, None]

	return [Xs_trn, Xs_tst] 

def get_config(Xs, rc_schema):
    '''
    get neccessary configurations of the given relation
    S = number of entity
    Ns = number of instances for each entity
    '''
    assert(len(Xs)==len(rc_schema)), "get_config: rc_schema lenth must be the same as input data."
    
    S = rc_schema.max() + 1
    Ns = -1 * numpy.ones(S, int)
    for i in range(len(Xs)):
        ri = rc_schema[i, 0]
        ci = rc_schema[i, 1]
        
        [m, n] = Xs[i].shape
        
        if Ns[ri] < 0:
            Ns[ri] = m
        else:
            assert(Ns[ri] == m), "get_config: rc_schema does not match data."
                            
        if Ns[ci] < 0:
            Ns[ci] = n
        else:
            assert(Ns[ci] == n), "get_config: rc_schema does not match data."
    return [S, Ns]

def RMSE(X, Y):
	'''
	X is prediction, Y is ground truth
	Both X and Y should be scipy.sparse.csc_matrix
	'''
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	if X.size > 0:
		return numpy.sqrt(sum(pow(X.data - Y.data, 2)) / X.size)
	else:
		return 0
	
def MAE(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr))
	if X.size > 0:
		return sum(abs(X.data - Y.data)) / X.size
	else:
		return 0

def check_modes(modes):
	for mode in modes:
		if mode != 'sparse' and mode != 'dense' and mode != 'log_dense':
			assert False, "Unrecognized mode: {}".format(mode)