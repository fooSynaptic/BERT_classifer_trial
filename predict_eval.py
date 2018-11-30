

infer_res = 'output/test_results.tsv'
test_file = './data/test.csv'

def inference(infer_res, test_file):
	with open(infer_res) as f:
		res = f.readlines()
		res = [x.split() for x in res]
	score = []
	for r in res:
		zero_pred, one_pred = enumerate(r)
		if zero_pred[1] > one_pred[1]:
			score.append(float(0))
		else:
			score.append(float(1))

	with open(test_file) as f:
		labels = [x.split('<>')[0] for x in f.readlines()]
		labels = [float(x) for x in labels]

	print(score, labels)

	s = 0
	for pred, true in zip(score, labels):
		if pred == true:
			s += float(1)
		else:
			pass

	print(s)
	return s/len(labels)

print(inference(infer_res, test_file))	
