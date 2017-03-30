import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import enchant
from nltk.metrics import edit_distance

stops = set(stopwords.words("english"))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

TEXT = file('./my_data/big.txt').read()

def tokens(text):
	# List all the word tokens (consecutive letters) in a text. Normalize to lowercase.
    return re.findall('[a-z]+', text.lower())

WORDS = tokens(TEXT)
COUNTS = Counter(WORDS)

def known(words):
	global COUNTS
	# Return the subset of words that are actually in the dictionary.
	return {w for w in words if w in COUNTS}

def splits(word):
	# Return a list of all possible (first, rest) pairs that comprise word.
	return [(word[:i], word[i:]) for i in xrange(len(word)+1)]

def edits1(word):
	# Return all strings that are one edit away from this word.
	global alphabet
	pairs = splits(word)
	deletes = [a+b[1:] for (a,b) in pairs if b]
	transposes = [a+b[1]+b[0]+b[2:] for (a,b) in pairs if len(b)>1]
	replaces = [a+c+b[1:] for (a,b) in pairs for c in alphabet if b]
	inserts = [a+c+b for (a,b) in pairs for c in alphabet]
	return set(deletes+transposes+replaces+inserts)

def edits2(word):
	# Return all strings that are two edits away from this word.
	return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

def correct(word):
	# Find the best spelling correction for this word.
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
    candidates = (known({word}) or known(edits1(word)) or known(edits2(word)) or {word})
    return max(candidates, key=COUNTS.get)

# using enchant to do spelling correction

class EnchantSpellingCorrector():
	def __init__(self, dict_name= 'en', max_dist=2):
		self.spell_dict = enchant.Dict(dict_name)
		self.max_dist = max_dist

	def correct(self, word):
		if self.spell_dict.check(word):
			return word
		suggestions = self.spell_dict.suggest(word)
		if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
			return suggestions[0]
		return word

spelling_corrector = EnchantSpellingCorrector()

def correct_text(text, tag='enchant'):
	# global stops
	cleaned_text = re.sub("[^\w\d]", " ", text)
	words = cleaned_text.lower().split()
	if not tag:
		correct_words = [correct(word) for word in words]
	else:
		correct_words = [spelling_corrector.correct(word) for word in words]
	# meaningful_words = [w for w in correct_words if not w in stops]
	return ' '.join(correct_words)

