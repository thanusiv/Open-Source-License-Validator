import gensim.parsing.preprocessing as gsp
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


class Preprocessor:
    """
    This class will be used for preprocessing the comment block texts. The self.filters are used for completing basic
    normalization tasks prior to vectorization
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

        self.filters = [
            gsp.strip_tags,
            gsp.strip_punctuation,
            gsp.strip_multiple_whitespaces,
            gsp.strip_numeric,
            gsp.remove_stopwords,
            gsp.strip_short,
            # self.lemmatize,
            gsp.stem_text
        ]

    def get_wordnet_pos(self, word):
        # Map POS (parts of speech) tags to first character lemmatize() accepts
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    # can use lemmatization over stemming if wanted
    def lemmatize(self, s):
        return " ".join([self.lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in word_tokenize(s)])

    def preprocess(self, s):
        s = s.lower()
        for f in self.filters:
            s = f(s)
        return s
