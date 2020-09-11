"""
Generate multiple area graphs and project them isometrically
"""
import shutil
import numpy
import csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plot
import numpy as np

from gensim.models import KeyedVectors

from backend.abstract.processor import BasicProcessor
from backend.lib.helpers import UserInput, convert_to_int
from backend.lib.exceptions import ProcessorInterruptedException

__author__ = "Stijn Peeters"
__credits__ = ["Stijn Peeters"]
__maintainer__ = "Stijn Peeters"
__email__ = "4cat@oilab.eu"

csv.field_size_limit(1024 * 1024 * 1024)

class HistWordsVectorSpaceVisualiser(BasicProcessor):
	"""
	Generate multiple area graphs, and project them on an isometric plane

	Allows for easy side-by-side comparison of prevalence of multiple
	attributes in a data set over time.
	"""
	type = "histwords-vectspace"  # job type ID
	category = "Visual"  # category
	title = "Chart vector space"  # title displayed in UI
	description = "Visualise a vector space(s) in a 2D graph "  # description displayed in UI
	extension = "svg"  # extension of result file, used internally and in UI

	input = "zip"
	output = "svg"

	accepts = ["align-word2vec", "generate-word2vec"]

	CMAP_MIN = 5

	options = {
		"words": {
			"type": UserInput.OPTION_TEXT,
			"help": "Words",
			"tooltip": "Separate with commas."
		},
		"num-words": {
			"type": UserInput.OPTION_TEXT,
			"help": "Amount of similar words",
			"min": 1,
			"default": 10,
			"max": 50
		},
		"threshold": {
			"type": UserInput.OPTION_TEXT,
			"help": "Similarity threshold",
			"tooltip": "Decimal value between 0 and 1; only words with a higher similarity score than this will be included",
			"default": "0.25"
		},
	}

	# a palette generated with https://medialab.github.io/iwanthue/
	colours = ["#eb010a", "#495dff", "#f35f00", "#5137e0", "#ffeb45", "#d05edf",
			   "#00cb3a", "#b200c7", "#d8fd5d", "#a058ff", "#b90fd4", "#6fb300",
			   "#ff40b5", "#9eff3b", "#022bc3"]
	colour_index = 0

	def process(self):
		input_words = self.parameters.get("words", "")
		if not input_words or not input_words.split(","):
			self.dataset.update_status("No input words provided, cannot look for similar words.", is_final=True)
			self.dataset.finish(0)
			return

		input_words = input_words.split(",")
		num_words = convert_to_int(self.parameters.get("num-words"), self.options["num-words"]["default"])
		try:
			threshold = float(self.parameters.get("threshold", self.options["threshold"]["default"]))
		except ValueError:
			threshold = float(self.options["threshold"]["default"])

		threshold = max(-1.0, min(1.0, threshold))

		# retain words that are common to all models
		staging_area = self.unpack_archive_contents(self.source_file)
		common_vocab = None
		models = {}
		for model_file in staging_area.glob("*.model"):
			if self.interrupted:
				shutil.rmtree(staging_area)
				raise ProcessorInterruptedException("Interrupted while processing word2vec models")

			model = KeyedVectors.load(str(model_file))
			models[str(model_file)] = model

			if common_vocab is None:
				common_vocab = set(model.vocab.keys())
			else:
				common_vocab &= set(model.vocab.keys())

			# prime model for further editing
			# if we don't do this "vectors_norm" will not be available later
			try:
				model.most_similar("4cat")
			except KeyError:
				pass

		# sort common vocabulary by combined frequency across all models
		common_vocab = list(common_vocab)
		common_vocab.sort(key=lambda w: sum([model.vocab[w].count for model in models.values()]), reverse=True)
		print("Common vocab: %i" % len(common_vocab))

		staging_area = self.unpack_archive_contents(self.source_file)
		relevant_vocab = set()
		for query in input_words:
			for model_file in staging_area.glob("*.model"):
				model = KeyedVectors.load(str(model_file))
				try:
					model_vocab = set([word[0] for word in model.most_similar(query, topn=1000)][:num_words])
					print(model_file.stem + ": " + repr(model_vocab))
				except KeyError:
					continue
				relevant_vocab |= model_vocab
				del model

		# take the last model and reduce it to only the relevant words
		last_model_file = sorted(list(staging_area.glob("*.model")), reverse=True)[0]
		model = KeyedVectors.load(str(last_model_file))

		# find vectors for relevant words
		word_vectors = np.empty((0, 100))
		for word in relevant_vocab:
			if word not in model.vocab:
				continue

			word_vector = model[word]
			word_vectors = np.append(word_vectors, [word_vector], axis=0)

		# find tsne coords for 2 dimensions
		tsne = TSNE(n_components=2, random_state=0)
		positions = tsne.fit_transform(word_vectors)

		raise RuntimeError