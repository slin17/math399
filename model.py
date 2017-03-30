from __future__ import division
import csv
import sys
import collections
import re
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os.path
import ast
from collections import Counter
from spell_checker import correct_text
import nltk
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def regen_and_clean_correct_data(input_file= 'train.tsv', output_file= 'regen_clean_train.csv'):
	output_file = './my_data/'+output_file
	if not os.path.isfile(output_file):
		with open('./data/'+input_file, 'rb') as f, open(output_file, 'wb') as f2:
			reader = csv.DictReader(f, dialect = 'excel-tab')
			header_list = ['EssaySet', 'Score', 'EssayText']
			writer = csv.DictWriter(f2, header_list)
			writer.writeheader()
			for row in reader:
				clean_correct_text = correct_text(row['EssayText'])
				writer.writerow({'EssaySet': row['EssaySet'], 'Score': row['Score1'], \
					'EssayText': clean_correct_text})
				writer.writerow({'EssaySet': row['EssaySet'], 'Score': row['Score2'], \
					'EssayText': clean_correct_text})
	else:
		print >> sys.stderr, "cleaned data already regenerated."
	print output_file

def get_score_cutoff(essay_set):
	'''
	Rubric range: 0-3 (EssaySets: 1, 2, 5, 6)
				: 0-2 (EssaySets: 3, 4, 7, 8, 9. 10)
	'''
	if essay_set in set([1, 2, 5, 6]):
		return 2
	return 1

def get_num_stopwords(essay_text):
	global stops
	num_stopwords = 0
	for word in essay_text.split():
		if word in stops:
			num_stopwords += 1
	return num_stopwords

def write_features_vocabs(fwriter, dwriter, vectorizers, list_essay_text_lists, prev_essay_set, scores_list):
	list_data_features = []
	for i in xrange(2): # compute bag-of-words separately for those in low and in high score ranges
		essay_text_list = list_essay_text_lists[i]
		vectorizer = vectorizers[i]
		data_features = vectorizer.fit_transform(essay_text_list).toarray()
		list_data_features.append(data_features)
		vocabs = [vocab.encode('ascii') for vocab in vectorizer.get_feature_names()]
		dwriter.writerow({'EssaySet': prev_essay_set, 'Low/High': i, 'Vocabs': vocabs})

	start = 0
	for i in xrange(2):
		essay_text_list = list_essay_text_lists[i]
		other_data_features = vectorizers[(i+1)%2].transform(essay_text_list).toarray()	# get bag-of-words from vocabs in other range
		for j, elem in enumerate(zip(list_data_features[i], other_data_features)):
			if i == 0:
				low_features, high_features = elem[0], elem[1]
			else:
				high_features, low_features = elem[0], elem[1]
			fwriter.writerow({'EssaySet': prev_essay_set, 'Idx': j, 'CountVectorLow': low_features, 'CountVectorHigh': high_features, \
				'#Words': len(essay_text_list[j].split()), '#StopWords': get_num_stopwords(essay_text_list[j]), \
				'Length': len(essay_text_list[j]), 'Score': scores_list[start+j]})
		start = j

def generate_features(input_file= 'sorted_data.csv', features_file= 'features.csv', dicts_file= 'dicts.csv'):
	if not os.path.isfile('./my_data/'+features_file) or not os.path.isfile('./my_data/'+dicts_file):
		with open('./my_data/'+input_file, 'rb') as f:
			data = pd.read_csv(f, header = 0)
			num_rows = data['EssayText'].size

			vectorizers = []
			for _ in xrange(2):
				vectorizer = CountVectorizer(analyzer = 'word',   \
											 ngram_range = (1,2), \
									         tokenizer = None,    \
									         preprocessor = None, \
									         stop_words = None,   \
									         max_features = 20)
				vectorizers.append(vectorizer)

			prev_essay_set, low, scores_list = -1, True, []
			list_essay_text_lists = [[],[]]
			write_out_features = False

			with open('./my_data/'+features_file, 'wb') as f2, open('./my_data/'+dicts_file, 'wb') as f3:
				fwriter = csv.DictWriter(f2, ['EssaySet', 'Idx', 'CountVectorLow', 'CountVectorHigh', '#Words', \
								'#StopWords', 'Length', 'Score'])
				fwriter.writeheader()
				dwriter = csv.DictWriter(f3, ['EssaySet', 'Low/High', 'Vocabs'])
				dwriter.writeheader()

				essay_text_list = list_essay_text_lists[0]
				for i in xrange(num_rows):
					curr_essay_set, curr_score = data['EssaySet'][i], data['Score'][i]

					if prev_essay_set == -1 or curr_essay_set == prev_essay_set:
						cut_off_score = get_score_cutoff(int(curr_essay_set))
						if int(curr_score) >= cut_off_score and low:
							# write out the top 5 words corresponding to 'low' scores
							low = False
							essay_text_list = list_essay_text_lists[1]
					else:
						# we see a new essay_set, write out the top 5 words for previous essay_set and score
						low = True # reset low
						write_out_features = True
					
					if write_out_features:
						# print essay_text_list
						write_features_vocabs(fwriter, dwriter, vectorizers, list_essay_text_lists, prev_essay_set, scores_list)
						list_essay_text_lists = [[],[]]
						scores_list = []
						essay_text_list = list_essay_text_lists[0]
						write_out_features = False

					curr_essay_text = data['EssayText'][i]
					to_append = curr_essay_text if not pd.isnull(curr_essay_text) else ''  
					essay_text_list.append(to_append)
					scores_list.append(curr_score)
					prev_essay_set = curr_essay_set

				low = True if not low else False # if 'low' is False, then the current_score >= cut_off_score, so low_high = 1
				write_features_vocabs(fwriter, dwriter, vectorizers, list_essay_text_lists, prev_essay_set, scores_list)
	else:
		print >> sys.stderr, "already regenerated features and dictionaries based on data."

def make_list(fstr):
	fstr_split = fstr.split()
	flist = [re.sub(r'[\[\]]','',n) for n in fstr_split]
	return [int(n) for n in flist if len(n)]

def compute_cross_validation(essay_set, curr_flist, scores_list, random_forests, cv_scores):
	print "Cross-Validating the model on Essay Set:", essay_set
	X, y = np.array(curr_flist), np.array(scores_list)
	forest = random_forests[essay_set-1]
	kf = KFold(n_splits=10, shuffle=True, random_state=1)
	cv_score, iteration = 0, 10
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index] 
		cv_score += forest.fit(X_train, y_train).score(X_test, y_test)
	avg_cv_score = cv_score/iteration
	cv_scores.append(avg_cv_score)
	print "Average 10-fold CV score:", avg_cv_score

def train_models(features_file= 'features.csv'):
	random_forests = []
	for _ in xrange(10): # creating 10 random forests, one for each essay_set
		random_forests.append(RandomForestClassifier(n_estimators = 100))

	with open('./my_data/'+features_file, 'rb') as f0:
		freader = csv.DictReader(f0)
		curr_flist, scores_list = [], []
		prev_essay_set = -1

		cv_scores = [] # keep track of cross-validation scores

		for row in freader:
			curr_essay_set = int(row['EssaySet'])

			if prev_essay_set != -1 and curr_essay_set != prev_essay_set:
				# train random forest for prev_essay_set
				compute_cross_validation(prev_essay_set, curr_flist, scores_list, random_forests, cv_scores)
				curr_flist, scores_list = [], []

			to_append = make_list(row['CountVectorLow']) + make_list(row['CountVectorHigh']) + \
								[int(row['#Words']), int(row['#StopWords']), int(row['Length'])]
			curr_flist.append(to_append)
			scores_list.append(row['Score'])
			prev_essay_set = curr_essay_set
		
		compute_cross_validation(prev_essay_set, curr_flist, scores_list, random_forests, cv_scores)
	
	print "Finished training models!"
	print "The cross-validation scores are:", cv_scores 
	return random_forests

def load_dicts(dicts_file):
	d = collections.defaultdict(lambda : {'0': [], '1': []})
	with open(dicts_file, 'rb') as f:
		reader = csv.DictReader(f)
		for row in reader:
			d[row['EssaySet']][row['Low/High']] = ast.literal_eval(row['Vocabs'])
	return d

def predict_and_write(forest, features_list, ids_list, actual_scores, essay_set, writer):
	results = forest.predict(features_list)
	i = 0
	for iD, result in zip(ids_list, results):
		writer.writerow({'essay_set': essay_set, 'id': iD, 'score': result, 'actual_score': actual_scores[i]})
		i += 1
	print "Finished predicting and writing out results for EssaySet:", essay_set


def predict(input_file, features_file= 'features.csv', dicts_file= 'dicts.csv', results_file= 'results.csv'):
	forests = train_models(features_file)
	dict_of_dicts = load_dicts('./my_data/'+dicts_file)

	prev_essay_set = '-1'
	with open('./data/'+input_file, 'rb') as f0, \
					open('./results/'+results_file, 'wb') as f1:
		if input_file[-3:] == 'tsv':
			reader = csv.DictReader(f0, dialect = 'excel-tab')
		else:
			reader = csv.DictReader(f0)
		writer = csv.DictWriter(f1, ['essay_set', 'id', 'score', 'actual_score'])
		writer.writeheader()
		features_list, ids_list, actual_scores = [], [], []
		for row in reader:
			curr_essay_set = row['EssaySet']
			if prev_essay_set == '-1' or curr_essay_set != prev_essay_set:
				vectorizers = []
				for d_score in dict_of_dicts[curr_essay_set]:
					vocabs = dict_of_dicts[curr_essay_set][d_score]
					vectorizers.append(CountVectorizer(vocabulary = vocabs))
				if prev_essay_set != '-1':
					predict_and_write(forests[int(prev_essay_set)-1], features_list, ids_list, actual_scores, prev_essay_set, writer)
					features_list, ids_list, actual_scores = [], [], []

			essay_text = row['EssayText']
			features = []
			for vectorizer in vectorizers:	
				features += list(vectorizer.transform([essay_text]).toarray()[0])
			features += [len(essay_text.split()), get_num_stopwords(essay_text), len(essay_text)]
			features_list.append(features)
			ids_list.append(row['Id'])
			actual_scores.append(row['Score1'])

			prev_essay_set = curr_essay_set

		predict_and_write(forests[int(prev_essay_set)-1], features_list, ids_list, actual_scores, prev_essay_set, writer)

def eval_performance(results_file= 'results.csv'):
	total_count, exact_match, plus_minus_one = 0, 0, 0
	with open('./results/'+results_file, 'rb') as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row['score']) == int(row['actual_score']):
				exact_match += 1
				plus_minus_one += 1
			elif abs(int(row['score']) - int(row['actual_score'])) == 1:
				plus_minus_one += 1
			total_count += 1
	print 'exact match:', (exact_match*100.00)/total_count, '% +/- 1 match:', (plus_minus_one*100.00)/total_count, '%'












			
			







		







