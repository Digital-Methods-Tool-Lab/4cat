# original by William Leif Hamilton, https://github.com/williamleif/histwords
# See 'license' in the 'histwords' folder
# File changed to make import paths work for 4CAT

import collections
import random

from backend.lib.histwords.representations.embedding import Embedding, SVDEmbedding

class SequentialEmbedding:
    def __init__(self, year_embeds, **kwargs):
        self.embeds = year_embeds
 
    @classmethod
    def load(cls, path, years, **kwargs):
        embeds = collections.OrderedDict()
        for year in years:
            embeds[year] = Embedding.load(path + "/" + str(year), **kwargs)
        return SequentialEmbedding(embeds)

    # 4CAT ADDITION
    @staticmethod
    def load_files(files, **kwargs):
        embeds = {}
        for file in files:
            year = file.stem
            file.resolve()
            embeds[year] = Embedding.load(str(file), **kwargs)

        return SequentialEmbedding(embeds)
    # END 4CAT ADDITION

    def get_embed(self, year):
        return self.embeds[year]

    def get_subembeds(self, words, normalize=True):
        embeds = collections.OrderedDict()
        for year, embed in self.embeds.iteritems():
            embeds[year] = embed.get_subembed(words, normalize=normalize)
        return SequentialEmbedding(embeds)

    def get_time_sims(self, word1, word2):
       time_sims = collections.OrderedDict()
       for year, embed in self.embeds.iteritems():
           time_sims[year] = embed.similarity(word1, word2)
       return time_sims

    def get_seq_neighbour_set(self, word, n=3):
        neighbour_set = set([])
        for embed in self.embeds.itervalues():
            closest = embed.closest(word, n=n)
            for _, neighbour in closest:
                neighbour_set.add(neighbour)
        return neighbour_set

    def get_seq_closest(self, word, start_year, num_years=10, n=10):
        closest = collections.defaultdict(float)
        for year in range(start_year, start_year + num_years):
            embed = self.embeds[year]
            year_closest = embed.closest(word, n=n*10)
            for score, neigh in year_closest.iteritems():
                closest[neigh] += score
        return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]


class SequentialSVDEmbedding(SequentialEmbedding):

    def __init__(self, path, years, **kwargs):
        self.embeds = collections.OrderedDict()
        for year in years:
            self.embeds[year] = SVDEmbedding(path + "/" + str(year), **kwargs)


