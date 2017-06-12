import numpy
import scipy.sparse
import os.path

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

def read_triple_data(train, test, user, item):
	'''
	read data from three column format (row, column, value)
	'''
	Dtrain = loadData(train).T
	Dtest = loadData(test).T
	# to make sure training and testing data with the same shapes
	num_user = int(max(Dtrain[0].max(), Dtest[0].max())) + 1
	num_item = int(max(Dtrain[1].max(), Dtest[1].max())) + 1
	X_train = scipy.sparse.coo_matrix((Dtrain[2],(Dtrain[0],Dtrain[1])), shape=(num_user, num_item))
	X_test = scipy.sparse.coo_matrix((Dtest[2],(Dtest[0],Dtest[1])), shape=(num_user, num_item))
	# transform to csc format
	X_train = scipy.sparse.csc_matrix(X_train)
	X_test = scipy.sparse.csc_matrix(X_test)

	# user or item features
	if user != '' and item != '':
		X_userFeat = loadTripleData(user, num_user, 0)
		X_itemFeat = loadTripleData(item, num_item, 0)

		Xs_trn = [X_train, X_userFeat, X_itemFeat]
		Xs_tst = [X_test, None, None]
		
		rc_schema = numpy.array([[0, 1], [0, 2], [1, 3]])
		# [row entity number, column entity number]
		# 0=user, 1=item, 2=userFeat, 3=itemFeat

		modes = ['sparse', 'log_dense', 'log_dense']
		# modes of each relation: sparse, dense or log_dense
	    # dense if Wij = 1 for all ij 
	    # sparse if Wij = 1 if Xij>0
	    # log if link function = logistic

	elif user == '' and item != '':
		X_itemFeat = loadTripleData(item, num_item, 0)

		Xs_trn = [X_train, X_itemFeat]
		Xs_tst = [X_test, None]

		rc_schema = numpy.array([[0, 1], [1, 2]])
		# [row entity number, column entity number]
		# 0=user, 1=item, 2=itemFeat

		modes = ['sparse', 'log_dense']

	elif user != '' and item == '':
		X_userFeat = loadTripleData(user, num_user, 0)

		Xs_trn = [X_train, X_userFeat]
		Xs_tst = [X_test, None]

		rc_schema = numpy.array([[0, 1], [0, 2]])
		# [row entity number, column entity number]
		# 0=user, 1=item, 2=userFeat

		modes = ['sparse', 'log_dense']

	elif user == '' and item == '':
		assert False, "No user and item features. Please use LibMF."

	return [Xs_trn, Xs_tst, rc_schema, modes] 

def get_config(Xs, rc_schema):
    '''
    get neccessary configurations of the given relation
    S = number of entity
    Ns = number of instances for each entity
    '''
    assert(len(Xs)==len(rc_schema)), "rc_schema lenth must be the same as input data."
    
    S = rc_schema.max() + 1
    Ns = -1 * numpy.ones(S, int)
    for i in range(len(Xs)):
        ri = rc_schema[i, 0]
        ci = rc_schema[i, 1]
        
        [m, n] = Xs[i].shape
        
        if Ns[ri] < 0:
            Ns[ri] = m
        else:
            assert(Ns[ri] == m), "rc_schema does not match data."
                            
        if Ns[ci] < 0:
            Ns[ci] = n
        else:
            assert(Ns[ci] == n), "rc_schema does not match data."
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

def string2list(input_string, num, sep='-'):
	string_list = input_string.split(sep)
	assert( len(string_list) == num ), 'argument alphas must be the same length as numbers of relations.'
	return [float(x) for x in string_list]

def save_result(args, rmse):
	if args.user != '' and args.item != '':
		cmf_type = 'useritem'
	elif args.user == '' and args.item != '':
		cmf_type = 'item'
	elif args.user != '' and args.item == '':
		cmf_type = 'user'

	if args.out != '':
		if os.path.exists(args.out) == True:
			with open(args.out, 'a') as fp:
				fp.write('{},{},{},{},{},{},{:.4f}\n'.format(cmf_type, args.k, args.reg, args.lr, args.tol, args.alphas, rmse))
		else:
			with open(args.out, 'w') as fp:
				fp.write('type,k,reg,lr,tol,alphas,RMSE\n')
				fp.write('{},{},{},{},{},{},{:.4f}\n'.format(cmf_type, args.k, args.reg, args.lr, args.tol, args.alphas, rmse))