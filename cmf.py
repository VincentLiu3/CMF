'''
Collective Matrix Factorization
Vincent Liu
2017 May
'''
import numpy
import time
import scipy.sparse
from anewton import logistic, newton_update
from utils import read_triple_data, read_binary_data, read_data, RMSE, MAE, check_modes, get_config

def learn(Xs, Xstst, rc_schema, alphas, modes, K, reg, learn_rate, max_iter=100, tol=0.4):
    assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2) # schema match data
    assert(numpy.all(rc_schema[:, 0] != rc_schema[:, 1])) # should not have symmetric relations
    assert(rc_schema.shape[0] == len(alphas))
    assert(rc_schema.shape[0] == len(modes))
    check_modes(modes)

    Xts = [None] * len(Xs)
    for i in range(len(Xs)):
        if Xs[i] is not None:
            Xts[i] = scipy.sparse.csc_matrix(Xs[i].T) # Transpose
            Xs[i] = scipy.sparse.csc_matrix(Xs[i]) # no Transpose
        if Xstst[i] is not None:
            Xstst[i] = scipy.sparse.csc_matrix(Xstst[i])
    
    [S, Ns] = get_config(Xs, rc_schema)

    # randomly initialize factor matrices with small values
    Us = [None] * S
    for i in range(S):
        Us[i] = numpy.random.rand(Ns[i], K) / K

    prev_loss = loss(Us, Xs, rc_schema, modes, alphas, reg)
    i = 0
    while i < max_iter:
        i += 1
        tic = time.time()
        for t in range(S):
            newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t)
        toc = time.time()

        print("[CMF] Iteration {}/{}. Time: {:.1f}".format(i, max_iter, toc - tic))

        training_loss = loss(Us, Xs, rc_schema, modes, alphas, reg)
        change_rate = (prev_loss-training_loss)/prev_loss * 100
        prev_loss = training_loss
        
        Ystst = predict(Us, Xstst, rc_schema, modes)
        testing_loss = RMSE(Xstst[0], Ystst[0])
        print("[CMF] Training Loss: {:.2f} (change {:.2f}%). Testing RMSE: {:.2f}".format(training_loss, change_rate, testing_loss))

        if change_rate < tol and i != 1:
            print("[CMF] Early terminate due to insufficient change in training loss!")
            break

    return Us

def loss(Us, Xs, rc_schema, modes, alphas, reg=0):
	'''
	Calculate objective loss
	See page 4: Generalizing to Arbitrary Schemas
	'''
	assert(rc_schema.shape[0] == len(Xs) and rc_schema.shape[1] == 2)

	Ys = predict(Us, Xs, rc_schema, modes)
	
	res = 0
	num_relation = len(Xs)
	# computing regularization for each latent factor
	for i in range(num_relation):
		for j in range(num_relation):
			if rc_schema[j, 0]==i or rc_schema[j, 1]==i:
				res += alphas[j] * reg * numpy.linalg.norm(Us[i].flat) # l2 norm

	# computing loss for each relation
	for j in range(num_relation):     
		alpha_j = alphas[j]
		if Xs[j] is None or Ys[j] is None or alpha_j == 0:
			continue

		X = scipy.sparse.csc_matrix(Xs[j])
		Y = scipy.sparse.csc_matrix(Ys[j])

		if modes[j] == 'sparse':
			assert( X.size == Y.size )
			res += alpha_j * numpy.sum(pow(X.data - Y.data, 2))

		elif modes[j] == 'dense' or modes[j] == 'log_dense':
			assert( numpy.all(Y.shape == X.shape) )
			res += alpha_j * numpy.sum(pow(X.toarray() - Y.toarray(), 2))   

	return res

def predict(Us, Xs, rc_schema, modes):
    '''
    see page 3: RELATIONAL SCHEMAS
    return a list of csc_matrix
    '''
    Ys = []
    for i in range(len(Xs)): # i = 1
        if Xs[i] is None:
        	# no need to predict Y
            Ys.append(None) 
            continue
        
        X = Xs[i]
        U = Us[rc_schema[i, 0]] 
        V = Us[rc_schema[i, 1]]

        if modes[i] == 'sparse':
            # predict only for non-zero elements in X
            X = scipy.sparse.csc_matrix(X)
            data = X.data.copy()
            indices = X.indices.copy()
            indptr = X.indptr.copy()
           
            for j in range(X.shape[1]): # for each column in X
                inds_j = indices[indptr[j]:indptr[j+1]]
                # indptr[j]:indptr[j+1] points to the data on j-th column of X
                if inds_j.size == 0:
                    continue
                data[indptr[j]:indptr[j+1]] = numpy.dot(U[inds_j, :], V[j, :])
            Y = scipy.sparse.csc_matrix((data, indices, indptr), X.shape)
            Ys.append(Y)

        elif modes[i] == 'dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

        elif modes[i] == 'log_dense':
            # predict for all elements in X
            Y = numpy.dot(U, V.T)
            Y = logistic(Y)
            Y = scipy.sparse.csc_matrix(Y)
            Ys.append(Y)

    return Ys

def run_cmf(Xs_trn, Xs_tst, rc_schema, alphas, modes, K, reg, learn_rate, T):
    print('----------------Running CMF----------------')
    Us = learn(Xs_trn, Xs_tst, rc_schema, alphas, modes, K, reg, learn_rate, T)

    Ys_tst = predict(Us, Xs_tst, rc_schema, modes)
    X = Xs_tst[0]
    Y = Ys_tst[0]
    print(X.data)
    print(Y.data)

    print('[CMF] Final Results:')
    print('[CMF] K = {}, reg = {:.5f}, learning rate = {:.5f} '.format(K, reg, learn_rate) )
    print('[CMF] RMSE = {:.4f} , MAE= {:.4f}'.format(RMSE(X, Y), MAE(X, Y)) )

if __name__ == "__main__":
    start_time = time.time()
    [Xs_trn, Xs_tst] = read_triple_data('ml-1m')
    end_time = time.time()
    print('[CMF] Finished loading data. Time: {:.1f} seconds'.format(end_time - start_time) )

    rc_schema = numpy.array([[0, 1], [0, 2], [1, 3]]) 
    # [row entity number, column entity number]
    # 0=user, 1=item, 2=userFeat, 3=itemFeat

    [S, Ns] = get_config(Xs_trn, rc_schema)
    print('----------------Data Summary----------------')
    print('[CMF] Number of entities = {}'.format(S))
    print('[CMF] Number of relations = {}'.format(len(Xs_trn)))
    print('[CMF] Number of instnace for each entity = {}'.format(Ns))
    print('[CMF] Training size = {}. Testing size = {}'.format(Xs_trn[0].size, Xs_tst[0].size))
    
    alphas = [0.8, 0.1, 0.1] 
    # alpha in [0, 1] weights the relative importance of relations

    modes = ['sparse', 'log_dense', 'log_dense']
    # modes of each relation: sparse, dense or log_dense
    # dense if Wij = 1 for all ij or sparse if Wij = 1 if Xij>0
    # log if F = logistic

    # reg = regulaizer
    # K = number of latent factors
    # learn_rate = learn rate
    # T = max number of iterations
    run_cmf(Xs_trn, Xs_tst, rc_schema, alphas, modes, K = 20, reg = 0.00001, learn_rate = 0.0062, T = 100)

    end_time = time.time()
    print('[CMF] Total Running Time: {:.0f} seconds'.format(end_time - start_time) )