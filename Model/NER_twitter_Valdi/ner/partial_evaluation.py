import numpy as np

#predicted = [[0,1,2,1,0,3,0,3],[0,1,2,0,0,2,0,0,3,4]]

#expected = [[0,1,1,0,3,3,3,4],[0,1,2,0,0,0,0,0,3,4]]


def to_tagged_dict(array, tag_b, tag_i):
	result = {}
	index = -1
	for i, sentence in enumerate(array):
		kode = i
		status = 0
		for j, tag in enumerate(sentence):
			if tag == tag_b:
				status = 1
				index+=1
				kode_new = (kode, j)
				result[index] = [kode_new]
			elif tag == tag_i:
				if status == 1:
					kode_new = kode, j
					result[index].append(kode_new)
				else:
					status = 1
					index+=1
					kode_new = (kode, j)
					result[index] = [kode_new]
			else:
				status = 0

	return result

def intersection(list_a, list_b):
	return [val for val in list_a if val in list_b]

def count_confussion(dict_predicted, dict_expected):
	flag_predicted = [False] * len(dict_predicted)
	flag_expected = [False] * len(dict_expected)

	tp = 0
	for i in dict_expected:
		item_expected = dict_expected[i]
		for j in dict_predicted:
			item_predicted = dict_predicted[j]
			if intersection(item_expected, item_predicted) and (not flag_expected[i]) and (not flag_predicted[j]):
				tp+=1
				flag_predicted[j] = True
				flag_expected[i] = True
	fp = flag_predicted.count(False)
	fn = flag_expected.count(False)
	return tp, fp, fn

def evaluate(list_predicted, list_expected, tag_a, tag_b):
	dict_predicted = to_tagged_dict(list_predicted, tag_a, tag_b)
	dict_expected = to_tagged_dict(list_expected, tag_a, tag_b)
	tp, fp, fn = count_confussion(dict_predicted, dict_expected)
	precision = tp * 1.0 / (tp + fp)
	recall = tp * 1.0 / (tp + fn)
	f1 = (2.0 * precision * recall) / (precision + recall)
	return precision, recall, f1