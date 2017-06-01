import multiprocessing
import queue
from cmf import *
from utils import *

# rm nohup.out
# nohup python3 grid.py & 
# tail –f nohup.out
class hyperparameters:
	def __init__(self, name, k, reg, lr, tol=1.0, verbose=1):
		self.name = str(name)
		self.train = 'data/ml-1m/train.txt'
		self.test = 'data/ml-1m/test.txt'
		self.user = 'data/ml-1m/user.txt'
		self.item = 'data/ml-1m/item.txt'
		self.save = 'result.txt'
		self.alphas = '0.8-0.1-0.1'
		self.k = k
		self.reg = reg
		self.lr = lr
		self.max_iter = 100
		self.tol = tol
		self.verbose = verbose

class MyProcess(multiprocessing.Process):
	def __init__(self, id, task_queue, lock):
		multiprocessing.Process.__init__(self, name=str(id))
		self.task_queue = task_queue
		self.lock = lock

	def run(self):
		[Xs_trn, Xs_tst] = read_triple_data('data/ml-1m/train.txt', 'data/ml-1m/test.txt', 'data/ml-1m/user.txt', 'data/ml-1m/item.txt')
		rc_schema = numpy.array([[0, 1], [0, 2], [1, 3]])
		modes = ['sparse', 'log_dense', 'log_dense']

		while True:
			try:
				args = self.task_queue.get(block=True, timeout=1) # task_queue
			except queue.Empty:
				break

			print('[P{}] Run args {}'.format(self.name, args.name))
			rmse = run_cmf(Xs_trn, Xs_tst, rc_schema, modes, args)

			self.lock.acquire()
			print('[P{}] Finish args {}'.format(self.name, args.name))
			print('[P{}] k = {}. reg = {:.5f}. lr = {:.5f}.'.format(self.name, args.k, args.reg, args.lr))
			print('[P{}] RMSE = {:.4f}'.format(self.name, rmse) )
			save_result(args, rmse)
			self.lock.release()
		

if __name__ == "__main__":
	ncpus = multiprocessing.cpu_count()
	num_cores = min(4, ncpus)
	task_queue = multiprocessing.Queue()

	count = 1
	for reg in [0.1, 0.01]:
		for lr in [0.1, 0.01]:
			for k in [10, 15]:
				args = hyperparameters(count, k, reg, lr)
				task_queue.put(args)
				count += 1

	save_lock = multiprocessing.Lock()
	record = []
	for i in range(num_cores):
	    process = MyProcess(i, task_queue, save_lock)
	    process.start()
	    record.append(process)

	task_queue.close()

	for process in record:
	    process.join()
