import numpy as np
from random import Random


class Partition(object):
	""" Dataset-like object, but only access a subset of it. """

	def __init__(self, data, index):
		self.data = data
		self.index = index

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]


class DataPartitioner(object):
	""" Partitions a dataset into different chuncks. """

	def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0, dataset=None):
		self.data = data
		self.dataset = dataset
		if isNonIID:
			self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha)

		else:
			self.partitions = []
			self.ratio = sizes
			rng = Random()
			rng.seed(seed)
			data_len = len(data)
			indexes = [x for x in range(0, data_len)]
			rng.shuffle(indexes)

			for frac in sizes:
				part_len = int(frac * data_len)
				self.partitions.append(indexes[0:part_len])
				indexes = indexes[part_len:]

	def use(self, partition):
		return Partition(self.data, self.partitions[partition])

	def __getNonIIDdata__(self, data, sizes, seed, alpha):
		labelList = data.targets
		rng = Random()
		rng.seed(seed)
		a = [(label, idx) for idx, label in enumerate(labelList)]
		# Same Part
		labelIdxDict = dict()
		for label, idx in a:
			labelIdxDict.setdefault(label, [])
			labelIdxDict[label].append(idx)
		labelNum = len(labelIdxDict)
		labelNameList = [key for key in labelIdxDict]
		labelIdxPointer = [0] * labelNum
		# sizes = number of nodes
		partitions = [list() for i in range(len(sizes))]
		eachPartitionLen = int(len(labelList) / len(sizes))
		# majorLabelNumPerPartition = ceil(labelNum/len(partitions))
		majorLabelNumPerPartition = 2
		basicLabelRatio = alpha

		interval = 1
		labelPointer = 0

		# basic part
		for partPointer in range(len(partitions)):
			requiredLabelList = list()
			for _ in range(majorLabelNumPerPartition):
				requiredLabelList.append(labelPointer)
				labelPointer += interval
				if labelPointer > labelNum - 1:
					labelPointer = interval
					interval += 1
			for labelIdx in requiredLabelList:
				start = labelIdxPointer[labelIdx]
				idxIncrement = int(basicLabelRatio * len(labelIdxDict[labelNameList[labelIdx]]))
				partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start + idxIncrement])
				labelIdxPointer[labelIdx] += idxIncrement

		# random part
		remainLabels = list()
		for labelIdx in range(labelNum):
			remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
		rng.shuffle(remainLabels)
		for partPointer in range(len(partitions)):
			idxIncrement = eachPartitionLen - len(partitions[partPointer])
			partitions[partPointer].extend(remainLabels[:idxIncrement])
			rng.shuffle(partitions[partPointer])
			remainLabels = remainLabels[idxIncrement:]

		return partitions

	def __getDirichletData__(self, data, psizes, seed, alpha):
		n_nets = len(psizes)
		K = 10
		labelList = np.array(data.targets)
		min_size = 0
		N = len(labelList)
		np.random.seed(2020)

		net_dataidx_map = {}
		while min_size < K:
			idx_batch = [[] for _ in range(n_nets)]
			# for each class in the dataset
			for k in range(K):
				idx_k = np.where(labelList == k)[0]
				np.random.shuffle(idx_k)
				proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
				## Balance
				proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
				proportions = proportions / proportions.sum()
				proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
				idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
				min_size = min([len(idx_j) for idx_j in idx_batch])

		for j in range(n_nets):
			np.random.shuffle(idx_batch[j])
			net_dataidx_map[j] = idx_batch[j]

		net_cls_counts = {}

		for net_i, dataidx in net_dataidx_map.items():
			unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
			tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
			net_cls_counts[net_i] = tmp
		print('Data statistics: %s' % str(net_cls_counts))

		local_sizes = []
		for i in range(n_nets):
			local_sizes.append(len(net_dataidx_map[i]))
		local_sizes = np.array(local_sizes)
		weights = local_sizes / np.sum(local_sizes)
		print(weights)

		return idx_batch, weights
