import numpy as np
import nltk as nlp

class SubjectiveTest:
    # Question patterns as class variable for easier modification
    QUESTION_PATTERNS = [
        "Explain in detail ",
        "Define ",
        "Write a short note on ",
        "What do you mean by "
    ]

    # Grammar pattern as class variable
    GRAMMAR_PATTERN = r"""
        CHUNK: {<NN>+<IN|DT>*<NN>+}
        {<NN>+<IN|DT>*<NNP>+}
        {<NNP>+<NNS>*}
    """

    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = int(noOfQues)
    
    @staticmethod
    def word_tokenizer(sequence):
        return [w for sent in nlp.sent_tokenize(sequence) 
                for w in nlp.word_tokenize(sent)]
    
    @staticmethod
    def create_vector(answer_tokens, tokens):
        return np.array([1 if tok in answer_tokens else 0 for tok in tokens])
    
    @staticmethod
    def cosine_similarity_score(vector1, vector2):
        def vector_value(vector):
            return np.sqrt(np.sum(np.square(vector)))
            
        v1 = vector_value(vector1)
        v2 = vector_value(vector2)
        if v1 == 0 or v2 == 0:
            return 0
            
        v1_v2 = np.dot(vector1, vector2)
        return (v1_v2 / (v1 * v2)) * 100
    
    def _extract_keywords(self, sentence):
        tagged_words = nlp.pos_tag(nlp.word_tokenize(sentence))
        tree = nlp.RegexpParser(self.GRAMMAR_PATTERN).parse(tagged_words)
        
        keywords = []
        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                keyword = " ".join(word for word, _ in subtree)
                keywords.append(keyword.upper())
        return keywords
    
    def generate_test(self):
        sentences = nlp.sent_tokenize(self.summary)
        question_answer_dict = {}
        
        # Extract keywords and build question-answer dictionary
        for sentence in sentences:
            if len(nlp.word_tokenize(sentence)) <= 20:
                continue
                
            keywords = self._extract_keywords(sentence)
            for keyword in keywords:
                if keyword not in question_answer_dict:
                    question_answer_dict[keyword] = sentence
                else:
                    question_answer_dict[keyword] += " " + sentence

        if not question_answer_dict:
            return [], []

        # Randomly select questions
        keywords = list(question_answer_dict.keys())
        if len(keywords) < self.noOfQues:
            return [], []

        selected_indices = np.random.choice(len(keywords), 
                                         size=min(self.noOfQues, len(keywords)), 
                                         replace=False)
        
        questions = []
        answers = []
        for idx in selected_indices:
            keyword = keywords[idx]
            answer = question_answer_dict[keyword]
            question_pattern = self.QUESTION_PATTERNS[idx % len(self.QUESTION_PATTERNS)]
            questions.append(question_pattern + keyword + ".")
            answers.append(answer)
            
        return questions, answers