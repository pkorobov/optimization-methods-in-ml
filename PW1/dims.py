from sklearn.datasets import load_svmlight_file

datasets = ["w8a", "gisette_scale", "real-sim"]
for dataset in datasets:
	A, b = load_svmlight_file(dataset)
	print(A.shape, b.shape)
