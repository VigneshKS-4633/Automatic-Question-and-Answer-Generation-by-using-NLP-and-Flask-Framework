import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from functools import lru_cache

class ObjectiveTest:
    # Grammar pattern as class variable for easier modification
    GRAMMAR_PATTERN = r"""
        CHUNK: {<NN>+<IN|DT>*<NN>+}
            {<NN>+<IN|DT>*<NNP>+}
            {<NNP>+<NNS>*}
    """

    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = int(noOfQues)

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        return [trivial for sent in sentences 
                if (trivial := self.identify_trivial_sentences(sent))]

    def identify_trivial_sentences(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        
        # Skip sentences that are too short or start with adverbs
        if tags[0][1] == "RB" or len(tokens) < 4:
            return None
        
        noun_phrases = self._extract_noun_phrases(tokens)
        if not noun_phrases:
            return None

        replace_nouns = self._get_replace_nouns(tags, noun_phrases)
        if not replace_nouns:
            return None

        return self._create_trivial_dict(sentence, replace_nouns)

    def _extract_noun_phrases(self, tokens):
        chunker = nltk.RegexpParser(self.GRAMMAR_PATTERN)
        pos_tokens = nltk.tag.pos_tag(tokens)
        tree = chunker.parse(pos_tokens)
        
        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                phrase = " ".join(word for word, _ in subtree)
                if not phrase.startswith("'"):
                    noun_phrases.append(phrase)
        return noun_phrases

    def _get_replace_nouns(self, tags, noun_phrases):
        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if word in phrase:
                    replace_nouns.extend(phrase.split()[-2:])
                    break
            if replace_nouns:
                break
        return replace_nouns if replace_nouns else [word for word, _ in tags[:1]]

    def _create_trivial_dict(self, sentence, replace_nouns):
        val = min(len(word) for word in replace_nouns)
        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________" * len(replace_nouns)).strip()
        question = re.sub(re.escape(replace_phrase), blanks_phrase, sentence, flags=re.IGNORECASE, count=1)
        
        trivial = {
            "Answer": replace_phrase,
            "Key": val,
            "Question": question
        }
        
        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []
            
        return trivial

    @staticmethod
    @lru_cache(maxsize=1000)
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")
        if not synsets:
            return []
            
        synset = synsets[0]
        if not synset.hypernyms():
            return []
            
        hypernym = synset.hypernyms()[0]
        hyponyms = hypernym.hyponyms()
        similar_words = []
        
        for hyponym in hyponyms:
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word != word:
                similar_words.append(similar_word)
            if len(similar_words) == 8:
                break
                
        return similar_words

    def generate_test(self):
        trivial_pairs = self.get_trivial_sentences()
        if not trivial_pairs:
            return [], []

        # Filter questions based on key value
        valid_questions = [q for q in trivial_pairs if q["Key"] > self.noOfQues]
        if not valid_questions:
            return [], []

        # Randomly select questions
        selected_indices = np.random.choice(len(valid_questions), 
                                         size=min(self.noOfQues, len(valid_questions)), 
                                         replace=False)
        
        questions = []
        answers = []
        for idx in selected_indices:
            questions.append(valid_questions[idx]["Question"])
            answers.append(valid_questions[idx]["Answer"])
            
        return questions, answers
