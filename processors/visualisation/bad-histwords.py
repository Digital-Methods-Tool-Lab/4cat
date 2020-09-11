"""
Generate multiple area graphs and project them isometrically
"""
import numpy
import csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plot

from backend.abstract.processor import BasicProcessor
from backend.lib.histwords.representations.sequentialembedding import SequentialEmbedding
from backend.lib.helpers import UserInput

__author__ = "Stijn Peeters"
__credits__ = ["Stijn Peeters"]
__maintainer__ = "Stijn Peeters"
__email__ = "4cat@oilab.eu"

csv.field_size_limit(1024 * 1024 * 1024)

class HistWordsVectorSpaceVisualiser():#BasicProcessor):
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

		staging_area = self.unpack_archive_contents(self.source_file)

		embeddings = SequentialEmbedding.load_files(list(staging_area.glob("*.model")))
		plot.figure(figsize=(20,20))

		for query in input_words:
			time_similarities, lookups, similarities = HistWordsVectorSpaceVisualiser.get_similarities(embeddings, query)
			words = list(lookups.keys())
			values = [lookups[word] for word in words]
			fitted = HistWordsVectorSpaceVisualiser.fit_tsne(values)

			# draw the words onto the graph
			color_map = HistWordsVectorSpaceVisualiser.get_color_map(len(time_similarities))
			arrows = HistWordsVectorSpaceVisualiser.plot_words(query, words, fitted, color_map, similarities)

			if arrows:
				HistWordsVectorSpaceVisualiser.draw_arrows(arrows)


		plot.savefig(str(self.dataset.get_results_path()))
		self.dataset.finish(self.parent.num_rows)

	@staticmethod
	def get_similarities(embeddings, query):
		time_similarities = {}
		lookups = {}
		similarities = {}
		for interval, model in embeddings.embeds.items():
			time_similarities[interval] = []

			for similarity, word in model.closest(query, n=15):
				word_item = "%s|%s" % (word, interval)
				if similarity > 0.3:
					time_similarities[interval].append((similarity, word_item))
					lookups[word_item] = model.represent(word)
					similarities[word_item] = similarity

		return time_similarities, lookups, similarities

	@staticmethod
	def fit_tsne(values):
		mat = numpy.array(values)
		model = TSNE(n_components=2, random_state=0, learning_rate=150, init='pca')
		fitted = model.fit_transform(mat)

		return fitted

	@staticmethod
	def get_color_map(n, name='YlGn'):
		return plot.cm.get_cmap(name, n + HistWordsVectorSpaceVisualiser.CMAP_MIN)

	@staticmethod
	def plot_words(query, words, positions, color_map, similarities):
		# TODO: remove this and just set the plot axes directly
		plot.scatter(positions[:, 0], positions[:, 1], alpha=0)
		plot.suptitle("%s" % query, fontsize=30, y=0.1)
		plot.axis('off')
		annotations = []
		for i in range(len(words)):
			position = positions[i]

			word_label, interval = [bit.strip() for bit in words[i].split("|")]
			# color = cmap((int(decade) - 1840) / 10 + CMAP_MIN)
			color = color_map((1990 - 1840) / 10 + HistWordsVectorSpaceVisualiser.CMAP_MIN)
			label_size = int(similarities[words[i]] * 30)

			# word1 is the word we are plotting against
			if word_label == query:
				annotations.append((word_label, interval, position))
				word_label = interval
				color = 'black'
				label_size = 15

			plot.text(position[0], position[1], word_label, color=color, size=label_size)

		return annotations

	@staticmethod
	def draw_arrows(annotations):
		# draw the movement between the word through the decades as a series of
		# annotations on the graph
		annotations.sort(key=lambda w: w[1], reverse=True)
		previous_position = annotations[0][-1]
		for word_label, decade, position in annotations[1:]:
			plot.annotate('', xy=previous_position, xytext=position,
						  arrowprops=dict(facecolor='blue', shrink=0.1, alpha=0.3, width=2, headwidth=15))
			previous_position = position


