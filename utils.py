import numpy
import scipy.sparse
import os.path

def read_dense_data(train_file, test_file, user_file, item_file, feature_mat_type):
	return 

def loadTripleData(filename):
	'''
	laod triple data (row, column, value) to csc_matrix format
	'''
	fData = numpy.loadtxt(filename, delimiter=',').T
	fData = scipy.sparse.coo_matrix((fData[2],(fData[0],fData[1]))).tocsc()
	return(fData)

def read_triple_data(train, test, user, item, feature_mat_type):
	'''
	read data from three column format (row, column, value)
	'''
	assert( feature_mat_type in ['sparse', 'dense', 'log_dense'] ), 'Unrecognized link function'

	# need to make sure training & testing data with the same shapes as user and item features
	num_user = num_item = 0	
	if user != '':
		X_userFeat = loadTripleData(user)
		num_user = X_userFeat.shape[0]
	if item != '':
		X_itemFeat = loadTripleData(item)
		num_item = X_itemFeat.shape[0]

	Dtrain = numpy.loadtxt(train, delimiter = ',').T
	Dtest = numpy.loadtxt(test, delimiter = ',').T
	num_user = int( max(Dtrain[0].max(), Dtest[0].max(), num_user-1) ) + 1
	num_item = int( max(Dtrain[1].max(), Dtest[1].max(), num_item-1) ) + 1
	X_train = scipy.sparse.coo_matrix((Dtrain[2],(Dtrain[0],Dtrain[1])), shape=(num_user, num_item)).tocsc()
	X_test = scipy.sparse.coo_matrix((Dtest[2],(Dtest[0],Dtest[1])), shape=(num_user, num_item)).tocsc()
	# transform to csc format
	# X_train = scipy.sparse.csc_matrix(X_train)
	# X_test = scipy.sparse.csc_matrix(X_test)

	# user or item features
	if user != '' and item != '':
		Xs_trn = [X_train, X_userFeat, X_itemFeat]
		Xs_tst = [X_test, None, None]
		
		rc_schema = numpy.array([[0, 1], [0, 2], [1, 3]])
		# [row entity number, column entity number]
		# 0=user, 1=item, 2=userFeat, 3=itemFeat

		modes = ['sparse', feature_mat_type, feature_mat_type]
		# modes of each relation: sparse, dense or log_dense
	    # dense if Wij = 1 for all ij 
	    # sparse if Wij = 1 if Xij>0
	    # log if link function = logistic

	elif user == '' and item != '':
		Xs_trn = [X_train, X_itemFeat]
		Xs_tst = [X_test, None]

		rc_schema = numpy.array([[0, 1], [1, 2]]) # 0=user, 1=item, 2=itemFeat
		modes = ['sparse', feature_mat_type]

	elif user != '' and item == '':
		Xs_trn = [X_train, X_userFeat]
		Xs_tst = [X_test, None]

		rc_schema = numpy.array([[0, 1], [0, 2]]) # 0=user, 1=item, 2=userFeat
		modes = ['sparse', feature_mat_type]

	elif user == '' and item == '':
		assert False, 'No user and item features.'
		Xs_trn = [X_train]
		Xs_tst = [X_test]

		rc_schema = numpy.array([[0, 1]])
		modes = ['sparse']

	return [Xs_trn, Xs_tst, rc_schema, modes] 

def get_config(Xs, rc_schema):
    '''
    get neccessary configurations of the given relation
    ---------------------
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
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr) and X.size > 0)
	return numpy.sqrt(sum(pow(X.data - Y.data, 2)) / X.size)
	
def MAE(X, Y):
	assert(X.size == Y.size and all(X.indices == Y.indices) and all(X.indptr == Y.indptr) and X.size > 0)
	return sum(abs(X.data - Y.data)) / X.size

def check_modes(modes):
	for mode in modes:
		if mode != 'sparse' and mode != 'dense' and mode != 'log_dense':
			assert False, 'Unrecognized mode: {}'.format(mode)

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
	elif args.user == '' and args.item == '':
		cmf_type = 'none'

	if args.out != '':
		if os.path.exists(args.out) is False:
			with open(args.out, 'w') as fp:
				fp.write('type,k,reg,lr,tol,alphas,RMSE\n')
		with open(args.out, 'a') as fp:
			fp.write('{},{},{},{},{},{},{:.4f}\n'.format(cmf_type, args.k, args.reg, args.lr, args.tol, args.alphas, rmse))
	