import numpy
from functools import reduce

def logistic(vec):
	out_vec = 1.0 / (1.0 + numpy.exp(-1 * vec))
	return out_vec

def d_logistic(vec):
	log_vec = logistic(vec)
	out_vec = numpy.multiply(log_vec, 1-log_vec)
	return out_vec

'''
def loss_for_one_row(Xi, U, V, reg):
	Yi = numpy.dot(U, V.T)
	loss = sum( pow( Xi-Yi, 2) ) + reg * numpy.linalg.norm(U) / 2
	return loss

def Armijo_line_search(U, one_step, Xi, V, reg):
	prev_loss = 
	while True:
		U -= one_step
		loss = loss_for_one_row(Xi, U, V, reg)
		if prev_loss
'''

def newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t):
	nsize = Ns[t] # size for entity t t, e.g. number of user for type 0 (user) 
	U_t = Us[t] # random factor matrix for entity t

	A = numpy.zeros((K, K)) # place holders for hessian: q'(Ui)
	b = numpy.zeros(K) # place holders for gradient: q(Ui)

	for i in range(nsize): # randomly pick one instance in X
		A[:] = 0
		b[:] = 0
		for j in range(len(Xs)):  # for j = 1~number of relations

			if alphas[j] == 0:
				continue
			
			if rc_schema[j, 0] == t or rc_schema[j, 1] == t:
				# only need to update if type t is in relation j
				if rc_schema[j, 0] == t:
					# if type t = x-axis of relation j 
					X = Xts[j] # transpose X (n2, n1)
					U = U_t[i, :] # (1 * k)
					V = Us[rc_schema[j, 1]]  # (n2 * k)
				
					data = X.data # content of the matrix
					indptr = X.indptr 
					indices = X.indices 

					ind_i0, ind_i1 = (indptr[i], indptr[i+1]) 
					# Step 1: XiV
					if ind_i0 == ind_i1:
						if modes[j] == "sparse": # sparse -> no data on the i-th row of X -> no need to update
							continue
						else: # dense/log_dense -> 0 vector
						 	XiV = numpy.zeros(K) # (1 * k)
					else:
						inds_i = indices[ind_i0:ind_i1] # index to the non-zero elements on the i-th row of X
						data_i = data[ind_i0:ind_i1] # non-zero element on the i-th row of X (1 * x)
						XiV = numpy.dot(data_i, V[inds_i, :]) # (1 * x) (x * k) -> (1 * k)
	
					if modes[j] == "sparse":
						V = V[inds_i, :] # only need those column factors for non-zero element in the i-th row
						# Step 2: UVt
						UiVt = numpy.dot(U, V.T) # (1*k) (k*n2) -> (1 * x)
						# Step 3: UVtV
						UiVtV = numpy.dot(UiVt, V)  # (1 * k)
						# Step 4: VtDiV
						Hes = numpy.dot(numpy.multiply(V.T, UiVt), V)

					elif modes[j] == 'log_dense':
						UiVt = numpy.dot(U, V.T) # (1 * n2)
						UiVtV = numpy.dot(logistic(UiVt), V) # (1 * k)
						Hes = numpy.dot(numpy.multiply(V.T, d_logistic(UiVt)), V)
					
					elif modes[j] == 'dense':
						UiVt = numpy.dot(U, V.T) # (1 * n2)
						UiVtV = numpy.dot(UiVt, V)  # (1 * k)
						Hes = numpy.dot(numpy.multiply(V.T, UiVt), V)

					A += alphas[j] * Hes
					b += alphas[j] * (UiVtV - XiV)

				elif rc_schema[j, 1] == t:
					# if type t = x-axis of relation j 
					X = Xs[j] # (n1 * n2)
					U = Us[rc_schema[j, 0]] # (n1 * k)
					V = U_t[i, :] # (1 * k)

					data = X.data # content of the matrix
					indptr = X.indptr
					indices = X.indices

					ind_i0, ind_i1 = (indptr[i], indptr[i+1])
					if ind_i0 == ind_i1: 
						if modes[j] == "sparse": # no data on the i-th column of X -> no need to update
							continue
						else:
							XiU = numpy.zeros(K) # (1 * k)
					else:
						inds_i = indices[ind_i0:ind_i1] 
						data_i = data[ind_i0:ind_i1] # non-zero elements on the j-th column of X (1 * x)
						XiU = numpy.dot(data_i, U[inds_i, :]) # (1 * k)

					if modes[j] == "sparse":
						U = U[inds_i, :] # (x * k)
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(UVt.T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, UVt), U)

					elif modes[j] == 'log_dense':
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(logistic(UVt).T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, d_logistic(UVt)), U)

					elif modes[j] == 'dense':
						UVt = numpy.dot(U, V.T) # (x * k) (k * 1) -> (x * 1)
						UVtU = numpy.dot(UVt.T, U) # (1 * k)
						Hes = numpy.dot(numpy.multiply(U.T, UVt), U)

					A += alphas[j] * Hes
					b += alphas[j] * (UVtU - XiU)
			
		if numpy.all(b == 0):
			continue
			
		# regularizer
		A += reg * numpy.eye(K, K)
		b += reg * U_t[i, :].copy() # the previous factor for i-th data

		d = numpy.dot(b, numpy.linalg.inv(A))
		Us[t][i, :] -= learn_rate * d  

	# return change

def old_newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t):
	'''
	code from http://ihome.ust.hk/~zluab/code/
	'''
	assert(t <= len(Ns) and t >= 0)
	eyeK = reg * numpy.eye(K, K)
	N = Ns[t] # number of instances for type t
	V = Us[t] # U
	A = numpy.zeros((K, K)) # place holders for hessian
	b = numpy.zeros(K) # place holders for gradient
	UtUs = numpy.empty(len(Xs),object)
	# change = 0
	for j in range(len(Xs)):
		if modes[j] == 'dense':
			if rc_schema[j, 0] == t:		
				U = Us[rc_schema[j, 1]]
			else:
				U = Us[rc_schema[j, 0]] 
			UtUs[j] = numpy.dot(U.T,U) # UtUs = VtV
	for i in range(N):
		A[:] = 0
		b[:] = 0
		for j in range(len(Xs)):
			if alphas[j] == 0:
				continue
			if rc_schema[j, 0] == t or rc_schema[j, 1] == t:
				if rc_schema[j, 0] == t:
					X = Xts[j]
					U = Us[rc_schema[j, 1]] # V
				else:
					X = Xs[j]
					U = Us[rc_schema[j, 0]]
				data = X.data
				indptr = X.indptr
				indices = X.indices
				
				ind_i0, ind_i1 = (indptr[i], indptr[i+1])
				if ind_i0 == ind_i1:
					continue
				
				inds_i = indices[ind_i0:ind_i1] 
				data_i = data[ind_i0:ind_i1]
				
				if modes[j] == "dense": # square loss, dense binary representation
					UtU = UtUs[j]
					Utemp = U[inds_i, :]
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU,V[i,:])-numpy.dot(data_i, Utemp))
				elif modes[j] == "log_dense": # logistic loss
					Xi = numpy.dot(U, V[i, :])
					Yi = - 1 * numpy.ones(U.shape[0])
					Yi[inds_i] = 1
					# (sigma(yx)-1)
					Wi = 1.0 / (1 + numpy.exp(-1 * numpy.multiply(Yi, Xi))) - 1 
					Wi = numpy.multiply(Wi, Yi)
					gv = numpy.dot(Wi, U)
					# compute sigmoid(x)
					Ai = 1 / (1 + numpy.exp(-Xi))
					Ai = numpy.multiply(Ai, 1 - Ai)
					Ai = Ai.reshape(Ai.size, 1)
					AiU = numpy.multiply(Ai, U)
					Hv = numpy.dot(AiU.T, U)
					A += alphas[j] * Hv
					b += alphas[j] * gv
					
				elif modes[j] == "sparse": # square loss
					Utemp = U[inds_i, :]
					UtU = numpy.dot(Utemp.T, Utemp)
					A += alphas[j] * UtU
					b += alphas[j] * (numpy.dot(UtU, V[i,:])-numpy.dot(data_i, Utemp))
					
		A += eyeK
		b += reg*V[i, :]
		d = numpy.dot(numpy.linalg.inv(A), b)
		vi = V[i,:].copy()
		V[i, :] -= learn_rate*d
	# return change

# http://sebastianruder.com/optimizing-gradient-descent/