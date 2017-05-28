import numpy
from functools import reduce
# import scipy

def logistic(vec):
	out_vec = 1.0 / (1 + numpy.exp(-1 * vec))
	return out_vec

def d_logistic(vec):
	log_vec = logistic(vec)
	out_vec = numpy.multiply(log_vec, 1-log_vec)
	return out_vec

def loss_for_single_factor(X_i, U, V, modes):
	UV = numpy.dot(U, V.T)

def newton_update(Us, Xs, Xts, rc_schema, alphas, modes, K, reg, learn_rate, Ns, t):
	nsize = Ns[t] # size for entity t t, e.g. number of user for type 0 (user) 
	U_t = Us[t] # random factor matrix for entity t

	A = numpy.zeros((K, K)) # place holders for hessian: q'(Ui)
	b = numpy.zeros(K) # place holders for gradient: q(Ui)

	change = 0
	for i in range(nsize): # randomly pick one instance in X
		A[:] = 0
		b[:] = 0
		for j in range(len(Xs)):  # for j = 1~number of relations

			if alphas[j] == 0:
				continue
			
			if rc_schema[j, 0] == t or rc_schema[j, 1] == t:
				# if type t is in relation j
				if rc_schema[j, 0] == t:
					# if type t = x-axis of relation j 
					X = Xts[j]
					U = U_t[i, :] # (1 * k)
					V = Us[rc_schema[j, 1]]  # (n2 * k)
				
					data = X.data # content of the matrix
					indptr = X.indptr 
					indices = X.indices 

					ind_i0, ind_i1 = (indptr[i], indptr[i+1])
					if ind_i0 == ind_i1:
						if modes[j] == "sparse": # no data on the i-th row of X -> no need to update
							continue
						else:
							XiV = numpy.zeros(K) # (1 * k)
					else:
						inds_i = indices[ind_i0:ind_i1] 
						data_i = data[ind_i0:ind_i1] # non-zero element on the j-th row of X (1 * x)
						XiV = numpy.dot(data_i, V[inds_i, :]) # (1 * x) (x * k) -> (1 * k)

						if modes[j] == "sparse":
							# only need those column factors for non-zero element in the i-th row
							V = V[inds_i, :]

					UVt = numpy.dot(U, V.T) # (1 * n2)

					if modes[j] == 'log_dense':
						UVtV = numpy.dot(logistic(UVt), V)
					else:
						UVtV = numpy.dot(UVt, V) # (1 * k)

					if modes[j] == 'log_dense':
						Di = numpy.diagflat(d_logistic(UVt)) # (n2 * n2)
					else:
						Di = numpy.diagflat(UVt) # (n2 * n2)

					Hes = reduce(numpy.dot, [V.T, Di, V]) # (k * n2) (n2 * n2) (n2 * k) -> (k * k)

					A += alphas[j] * Hes
					b += alphas[j] * (UVtV - XiV)

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
						data_i = data[ind_i0:ind_i1] # value on the j-th column of X (1 * x)
						XiU = numpy.dot(data_i, U[inds_i, :]) # (1 * k)

						if modes[j] == "sparse":
							U = U[inds_i, :]

					VUt = numpy.dot(V, U.T) # (1 * n1)

					if modes[j] == 'log_dense':
						VUtU = numpy.dot(logistic(VUt), U) # (1 * k)
					else:
						VUtU = numpy.dot(VUt, U) # (1 * k)

					if modes[j] == 'log_dense':
						Di = numpy.diagflat(d_logistic(VUt)) # (n1 * n1)		
					else:
						Di = numpy.diagflat(VUt) # (n1 * n1)	

					Hes = reduce(numpy.dot, [U.T, Di, U]) # (k * n1)(n1 * n1)(n1 * k)=(k * k)	

					A += alphas[j] * Hes
					b += alphas[j] * (VUtU - XiU)
				
		ui_old = U_t[i,:].copy() # the previous factor for i-th data
		# regularizer
		A += reg * numpy.eye(K, K)
		b += reg * ui_old

		# A = [q'(Ui)]^-1, b = q(Ui)
		# Ui <- Ui - learn_rate * b A^-1
		d = numpy.dot(b, numpy.linalg.inv(A))
		Us[t][i, :] -= learn_rate * d  

	# return change

# http://sebastianruder.com/optimizing-gradient-descent/