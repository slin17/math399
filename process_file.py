import csv
import ast
from spell_checker import correct_text

# with open('./my_data/dicts.csv', 'rb') as f:
# 	reader = csv.DictReader(f)
# 	for row in reader:
# 		vocabs = row['Vocabs']
# 		vocabs_list = ast.literal_eval(vocabs)
# 		print vocabs, len(vocabs), len(vocabs_list)

with open('./data/public_leaderboard_rel_2.tsv', 'rb') as f0, \
		open('./data/public_leaderboard_solution.csv', 'rb') as f1, \
		open('./data/cc_mod_public_leaderboard.csv', 'wb') as f2:
	r0 = csv.DictReader(f0, dialect = 'excel-tab')
	r1 = csv.DictReader(f1)
	writer = csv.DictWriter(f2, ['Id', 'EssaySet', 'Score1', 'EssayText'])
	writer.writeheader()
	for row0, row1 in zip(r0, r1):
		writer.writerow({'Id': row0['Id'], 'EssaySet': row0['EssaySet'], 'Score1': row1['essay_score'], 
			'EssayText': correct_text(row0['EssayText'])})
