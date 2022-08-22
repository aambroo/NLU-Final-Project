###########################################################################
############################## PREPROCESSING ##############################
###########################################################################
##   Bag of functions to support tasks that have to do with preprocessing##
##   Functions mainly cover tasks beneficial to dataset preprocessing    ##
##    which will be required for later purposes.                         ##
###########################################################################
##              Editor: Matteo Ambrosini - mat. #232885                  ##
###########################################################################


from tkinter import N
import joblib
import nltk
import numpy as np
from nltk.sentiment.util import mark_negation


def lol2str(doc):
    '''Transforms a document in the list-of-lists format into
    a block of text (str type).'''
    return " ".join([word for sent in doc for word in sent])


def mr2str(dataset):
    '''Transforms the Movie Reviews Dataset (or a slice) into a block of text.'''
    return [lol2str(doc) for doc in dataset]


def get_movie_reviews_dataset(mark_negs:bool = True) -> str:    #---> utils.preprocessing
    '''Uses the nltk library to download the 'Movie Reviews' dateset,
    splitting it into negative reviews and positive reviews.
    Toggle :param mark_neg: if u wish sentences to be mark-negated or not.'''
    #nltk.download("movie_reviews")
    from nltk.corpus import movie_reviews
    pos = movie_reviews.paras(categories="pos")
    neg = movie_reviews.paras(categories="neg")
    if mark_negs:
        pos = [[mark_negation(sent) for sent in doc] for doc in pos]
        neg = [[mark_negation(sent) for sent in doc] for doc in neg]
    return pos, neg


def hconcat(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:      #---> utils.preprocessing
    '''Applies horizontal concatenation to the X1 and X2 matrices, returning the concatenated matrix.'''
    assert len(X1.shape) == len(
        X2.shape) == 2, "function 'hconcat' only works with matrices (np.array with 2 dimensions)."
    assert X1.shape[0] == X2.shape[0], "In order to hconcat matrices, they must have the same number of rows."
    N = X1.shape[0]
    M = X1.shape[1] + X2.shape[1]
    X = np.ndarray(shape=(N, M))
    X[:, :X1.shape[1]] = X1
    X[:, X1.shape[1]:] = X2
    return X


def vconcat(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Applies vertical concatenation to the X1 and X2 matrices, returning the concatenated matrix."""
    assert len(X1.shape) == len(
        X2.shape) == 2, "function 'vconcat' only works with matrices (np.array with 2 dimensions)."
    assert X1.shape[1] == X2.shape[1], "In order to vconcat matrices, they must have the same number of columns."
    N = X1.shape[0] + X2.shape[0]  # sum of
    M = X1.shape[1]
    X = np.ndarray(shape=(N, M))
    X[:X1.shape[0], :] = X1
    X[X1.shape[0]:, :] = X2
    return X




#################################################################################
################################### LOGGER ######################################
#################################################################################
####                  Defining Custom-Colored logger.                        ####
#################################################################################
####              Author: Matteo Ambrosini - mat. #232885                    ####
#################################################################################

import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: yellow + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_neat_logger(logger_name="neat_logger"):
    '''Get Custom Color Logger'''
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger



#################################################################################
################################# DIFFPOSNEG ####################################
#################################################################################
####   Implementation of the DiffPosNeg feature as proposed in the orignal   ####
####    paper (link below or in the REPORT).                                 ####
#################################################################################
## Author: Nguyen et al - original paper:https://aclanthology.org/I13-1114.pdf ##
####              Editor: Matteo Ambrosini - mat. #232885                    ####
#################################################################################

from time import time
import numpy as np

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet

import multiprocessing as mp
from sklearn.base import BaseEstimator, TransformerMixin


pos2wn = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as an iterable of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        if pos == 'a':
            synsets = [ss for ss in synsets if str(ss.pos()) in ['a', 's']]
        else:
            synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense


def valence_count(sent, tokenizer, memory, update_mem):
    """Given a string :param: sent, returns the count of both
    positive and negative tokens in it."""
    tokens = tokenizer(sent)
    tagged_tokens = nltk.pos_tag(tokens, tagset="universal")
    tagged_tokens = [(t, pos2wn.get(pos_tag, None))
                     for (t, pos_tag) in tagged_tokens]
    sentence_counts = {"pos": 0, "neg": 0}
    for (t, pos_tag) in tagged_tokens:
        token_label = memory.get(t, None)
        if token_label is None:
            token_label = "neg"
            ss = lesk(tokens, t, pos=pos_tag)
            if ss:
                sense = swn.senti_synset(ss.name())
                if sense.pos_score() >= sense.neg_score():
                    token_label = "pos"
            if update_mem:
                memory[t] = token_label
        sentence_counts[token_label] += 1
    return sentence_counts


def swn_sentence_classification(sent, tokenizer, memory, update_mem):
    valence_counts = valence_count(sent, tokenizer, memory, update_mem)
    return 0 if valence_counts["neg"] > valence_counts["pos"] else 1

# Define DiffPosNeg Logger
dpn_logger = get_neat_logger('DiffPosNeg')

class DiffPosNegVectorizer(BaseEstimator, TransformerMixin):
    """Class for implementing the DiffPosNeg feature as described in https://aclanthology.org/I13-1114/
    through scikit-learn APIs."""
    
    def __init__(self, tokenizer=word_tokenize, lb=0, ub=1):
        """
        - :param tokenizer: Callable parameter, used to extract tokens from documents
        when vectorizing;
        - :param lb: lower bound for clipping absolute values of numerical distances once scaled;
        - :param rb: same as :param lb:, but upper bound.
        """
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self.tokenizer = tokenizer
        self.lb = lb
        self.ub = ub

    def diff_pos_neg_feature(self, doc, memory, update_mem=False, as_ratio=True) -> list:
        """Returns the DiffPosNeg feature of :param: doc.
        The feature is defined as the numerical distance between sentences
        with a positive orientation and sentences with a negative orientation."""
        pos_count, neg_count = 0, 0
        for sent in sent_tokenize(doc):
            sent_cls = swn_sentence_classification(
                sent, self.tokenizer, memory, update_mem)
            if sent_cls == 0:
                neg_count += 1
            else:
                pos_count += 1
        if pos_count >= neg_count:
            if as_ratio:
                return [abs(pos_count-neg_count)/(pos_count+neg_count), 1]
            return [abs(pos_count-neg_count), 1]
        if as_ratio:
            return [abs(pos_count-neg_count)/(pos_count+neg_count), 0]
        return [abs(pos_count - neg_count), 0]

    def fit(self, X, y=None, **fit_params):
        self.memory_ = {}
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, True) for doc in X]))
            self.memory_ = {k: v for k, v in mem.items()}
        distances = diff_pos_neg_feats[:, 0]
        self.min_ = np.amin(distances)
        self.max_ = np.amax(distances)
        return self

    def transform(self, X):
        start_time = time()
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            mem = {k: v for k, v in self.memory_.items()}
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, False) for doc in X]))
        distances = diff_pos_neg_feats[:, 0]
        prevalences = diff_pos_neg_feats[:, -1]

        # scale the values in the range [0,100], taking care of possible values outside the fitted min/max by clipping
        distances = np.clip((distances - self.min_) / (self.max_ -
                            self.min_ + np.finfo(float).eps), a_min=self.lb, a_max=self.ub)
        distances = np.int16(distances*100)

        # put components together and return
        distances = np.expand_dims(distances, axis=-1)
        prevalences = np.expand_dims(np.array(prevalences), axis=-1)
        dpn_logger.info(print_time(
            message = f'Transformed {len(X)} documents in', 
            start_time=start_time, 
            out_log=True
            ))
        return hconcat(distances, prevalences)

    def fit_transform(self, X, y=None, **fit_params):
        start_time = time()
        self.memory_ = {}
        # apply parallel execution of the 'diff_pos_neg' feature extraction function
        with mp.Manager() as manager:
            mem = manager.dict()
            with mp.Pool(processes=mp.cpu_count()) as pool:
                diff_pos_neg_feats = np.array(pool.starmap(
                    self.diff_pos_neg_feature, [(doc, mem, True) for doc in X]))
            self.memory_ = {k: v for k, v in mem.items()}
        distances = diff_pos_neg_feats[:, 0]
        prevalences = diff_pos_neg_feats[:, -1]
        dpn_logger.info(f'Number of positive documents: {np.count_nonzero(prevalences)}')

        # override stats inferred from the data
        self.min_ = np.amin(distances)
        self.max_ = np.amax(distances)

        # scaling the values of the distances in the range [0, 1]
        distances = (distances - self.min_) / \
            (self.max_ - self.min_ + np.finfo(float).eps)
        distances = np.int16(distances*100)

        # put the feature components back together after post-processing and return
        distances = np.expand_dims(distances, axis=-1)
        prevalences = np.expand_dims(prevalences, axis=-1)
        dpn_logger.info(print_time(
            message=f'Fitted Model and transformed {len(X)} documents in',
            start_time = start_time,
            out_log=True))
        return hconcat(distances, prevalences)




###################################################################################
############################ TWO STAGE CLASSIFIER #################################
###################################################################################
####   Implementation of the 2-stage classifier as proposed in the original    ####
####    paper.                                                                 ####
####   - First Stage Clf  --> sklearn estimator                                ####
####   - Second Stage Clf --> extended implementation of ClassifierMixin class ####
####   - note: dimensionality reduction is not implemented in the original paper ##
####           using LDA (Linear Discriminant Analysis)                          ##
###################################################################################
## Author: Nguyen et al - original paper:https://aclanthology.org/I13-1114.pdf ####
####              Editor: Matteo Ambrosini - mat. #232885                      ####
###################################################################################

# Imports
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from joblib import load
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted

# Paths
path_to_models = 'tmp/models'
first_stage_vectorizer_path     = os.path.join(path_to_models, 'first_stage_vectorizer.joblib')
second_stage_vectorizer_path    = os.path.join(path_to_models, 'second_stage_vectorizer.joblib')
reduce_dim_path                 = os.path.join(path_to_models, 'reduce_dim.joblib')
subj_detector_path              = os.path.join(path_to_models, 'count_bernoulli_subj_det_model.joblib')
subj_vectorizer_path            = os.path.join(path_to_models, 'count_bernoulli_subj_det_vectorizer.joblib')

# Setup Logger
logger = get_neat_logger('Two Stage Classifier')

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    """Implementation of Two Stage Classifier as proposed in https://aclanthology.org/I13-1114/"""

    def __init__(self, first_stage_vectorizer_path: joblib = first_stage_vectorizer_path,
                 second_stage_vectorizer_path: joblib = second_stage_vectorizer_path,
                 neg_cls_min_confidence: float = 0.6,
                 pos_cls_min_confidence: float = 0.6,
                 reduce_dim: bool =True,
                 use_subjectivity: bool = False,
                 subj_detector_path: os.path = subj_detector_path,
                 subj_vectorizer_path: os.path = subj_vectorizer_path):
        """
        - :param first_stage_vectorizer_path: dumped .joblib file, containing the fitted
        sklearn vectorizer to be used for the Naive Bayes classification at the first stage;
        - :param second_stage_vectorizer_path: analogous to :param first_stage_vectorizer_path:,
        but for the second stage classification with Support Vectors;
        - :param_neg_cls_min_confidence: minimum confidence required to the First Stage when classifying
        documents predicted to be negative. In case the prediction unsatisfies this requirement, the 
        Second Stage is invoked;
        - :param pos_cls_min_confidence: analogous to :param_neg_cls_min_confidence:, but for the positive class;
        - :param reduce_dim: if set to True, Linear Discriminant Analysis will be used to project input data along
        a direction maximizing the class separability before classifying with the Second Stage;
        - :param use_subjectivity: if set to True, extracts the subjectivity feature from documents and uses it as
        an additional component to the feature vector of input documents; 
        - :param subj_detector_path: dumped subjectivity detector to extract the subjectivity feature from input 
        documents. Unused if use_subjectivity=False;
        - :param subj_vectorizer_path: dumped joblib vectorizer to preprocess documents before extracting the subjectivity
        feature. Unused if use_subjectivity=False.
        """

        assert 0 < neg_cls_min_confidence < 1, \
            "Min confidence for the Negative Classifier must stay between (0,1)"
        assert 0 < pos_cls_min_confidence < 1, \
            "Min confidence for the Positive Classifier must stay between (0,1)"
        self.first_stage_clf = MultinomialNB()
        self.second_stage_clf = SVC()
        self.neg_cls_min_confidence = neg_cls_min_confidence
        self.pos_cls_min_confidence = pos_cls_min_confidence
        self.first_stage_vectorizer_path = first_stage_vectorizer_path
        self.second_stage_vectorizer_path = second_stage_vectorizer_path
        self.first_stage_vectorizer = load(self.first_stage_vectorizer_path)
        self.second_stage_vectorizer = load(self.second_stage_vectorizer_path)
        self.reduce_dim = reduce_dim
        self.dim_reducer = None
        if self.reduce_dim:
            self.dim_reducer = LinearDiscriminantAnalysis()
        self.use_subjectivity = use_subjectivity
        self.subj_detector_path = subj_detector_path
        self.subj_vectorizer_path = subj_vectorizer_path
        if self.use_subjectivity:
            self.subj_detector = load(self.subj_detector_path)
            self.subj_vectorizer = load(self.subj_vectorizer_path)

    def subjectivity_features(self, tokenized_corpus):
        feats = []
        for i, doc in enumerate(tokenized_corpus):
            sents = [" ".join(sent) for sent in doc]
            vectors = self.subj_vectorizer.transform(sents)
            y_hat = self.subj_detector.predict(vectors)
            feats.append(1 if np.count_nonzero(
                np.array(y_hat)) >= len(y_hat) else 0)
        return np.array(feats)

    def fit_first_stage_vectorizer(self, X, transform=False):
        if transform:
            return self.first_stage_vectorizer.fit_transform(X)
        return self.first_stage_vectorizer.fit(X)

    def fit_second_stage_vectorizer(self, X, transform=False):
        if transform:
            return self.second_stage_vectorizer.fit_transform(X)
        return self.second_stage_vectorizer.fit(X)

    def fit_first_stage(self, X: np.ndarray, y: np.ndarray) -> None:
        check_is_fitted(self.first_stage_vectorizer,
                        msg="""Make sure to fit the vectorizer of the 1st stage before fitting the respective classifier.
                        You can do it by calling the 'fit_first_stage_vectorizer' method.""")
        logger.info("Fitting First Stage Classifier")
        self.first_stage_clf.fit(X, y)

    def fit_second_stage(self, X: np.ndarray, y: np.ndarray, tokenized_corpus=None) -> None:
        check_is_fitted(self.second_stage_vectorizer,
                        msg="""Make sure to fit the Second Stage Vectorizer before fitting the respective classifier.
                        You can do it by calling the 'fit_second_stage_vectorizer' method.""")
        if isinstance(X, csr_matrix):
            X = X.toarray()
        if self.reduce_dim:
            logger.info("Fitting dimensionality reduction module")
            X = self.dim_reducer.fit_transform(X, y)

        if self.use_subjectivity and tokenized_corpus is not None:
            subj_features = self.subjectivity_features(tokenized_corpus)
            X = hconcat(X, np.expand_dims(subj_features, -1))

        logger.info("Fitting Second Stage Classifier")
        self.second_stage_clf.fit(X, y)

    def fit(self, X: list[list[str]], y: np.ndarray) -> None:
        start_time = time()
        X_los = mr2str(X)

        X_first_stage = self.first_stage_vectorizer.transform(X_los)
        self.fit_first_stage(X_first_stage, y)

        X_second_stage = self.second_stage_vectorizer.transform(X_los)
        self.fit_second_stage(X_second_stage, y, X)

        logger.info(print_time(
            message='Done fitting in', 
            start_time=start_time, 
            out_log=True))

        return self

    def reject_option(self, y) -> bool:
        cls = np.argmax(y)
        if cls == 0 and y[cls] < self.neg_cls_min_confidence:  # negative case
            return True
        if cls == 1 and y[cls] < self.pos_cls_min_confidence:  # positive case
            return True
        return False

    def predict(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        Y = []
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C_first_stage = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, cls_proba in enumerate(C_first_stage):
            if self.reject_option(cls_proba) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)
            else:
                y = np.argmax(cls_proba)
                Y.append(y)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.reduce_dim:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                Y.insert(rejected_idx, y)

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X)*100:.1f}% of the time")

        # memory cleanup and return
        del C_first_stage
        del rejected_samples_los
        del rejected_indices
        return np.array(Y)

    def predict_proba(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C = self.first_stage_clf.predict_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.reduce_dim:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a class probability of 1 to the class predicted by the SVM
                # 0 probability otherwise
                cls_probs = np.array([.0, .0])
                cls = np.argmax(y)
                cls_probs[cls] = 1.
                C[rejected_idx] = cls_probs

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X)*100:.1f}% of the time")

        # memory cleanup and return
        del rejected_samples_los
        del rejected_indices
        return np.array(C)

    def predict_log_proba(self, X: list[list[str]]) -> np.ndarray:
        # first predict with the first stage classifier and keep track
        # of which samples are too difficult to classify
        X_los = mr2str(X)
        X_transformed = self.first_stage_vectorizer.transform(X_los)
        C = self.first_stage_clf.predict_log_proba(X_transformed)
        rejected_samples_los, rejected_samples_lol, rejected_indices = [], [], []
        for i, c in enumerate(C):
            if self.reject_option(c) and self.second_stage_vectorizer is not None:
                rejected_sample_los = X_los[i]
                rejected_samples_los.append(
                    rejected_sample_los)  # i-th document in X
                rejected_sample_lol = X[i]
                rejected_samples_lol.append(rejected_sample_lol)
                rejected_indices.append(i)

        # predict classes for the rejected samples with the 2nd stage classifier
        if len(rejected_samples_los) > 0:
            rejected_vectors = self.second_stage_vectorizer.transform(
                rejected_samples_los)
            if isinstance(rejected_vectors, csr_matrix):
                rejected_vectors = rejected_vectors.toarray()
            if self.reduce_dim:
                rejected_vectors = self.dim_reducer.transform(rejected_vectors)
            if self.use_subjectivity:
                subj_features = self.subjectivity_features(
                    rejected_samples_lol)
                rejected_vectors = hconcat(
                    rejected_vectors, np.expand_dims(subj_features, axis=-1))
            Y_second_stage = self.second_stage_clf.predict(rejected_vectors)
            for rejected_idx, y in zip(rejected_indices, Y_second_stage):
                # assign a log-prob of 0 to the predicted class (log(1))
                # assign negative infinity to the other class (approx. log(0))
                log_probs = np.array([-np.inf, -np.inf])
                cls = np.argmax(y)
                log_probs[cls] = 0
                C[rejected_idx] = log_probs

        logger.info(
            f"SVC was called {len(rejected_samples_los)/len(X) * 100:.1f}% of the time")

        # memory cleanup and return
        del rejected_samples_los
        del rejected_indices
        return np.array(C)

    def score(self, X: list[list[str]], y):
        y_hat = self.predict(X)
        binary_acc_vector = [1 if y_gt_ == y_hat_ else 0 for (
            y_gt_, y_hat_) in zip(y, y_hat)]
        return sum(binary_acc_vector) / len(binary_acc_vector)



###########################################################################
############################## MISCELLANEOUS ##############################
###########################################################################
##   Bag of functions to support basic tasks required multiple times.    ##
##   Functions contained in this section are those used for several      ##
##    purposes.                                                          ##
###########################################################################
##              Author: Matteo Ambrosini - mat. #232885                  ##
###########################################################################


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
import os
from joblib import dump
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def switch_vectorizer(vectorizer_name='bow'):     #---> utils.miscellaneous
    assert vectorizer_name in ('bow', 'tfidf', 'diffposneg', 'bert')
    if vectorizer_name == 'bow':
        return CountVectorizer(tokenizer=word_tokenize)
    elif vectorizer_name == 'tfidf':
        return TfidfVectorizer(tokenizer=word_tokenize)
    elif vectorizer_name == 'diffposneg':
        return DiffPosNegVectorizer()


def fit_transform_save(vectorizer, dataset, path):
    X = vectorizer.fit_transform(mr2str(dataset))
    if isinstance(X, csr_matrix):  # in case it is a scipy.csr.csr_matrix
        X = X.toarray()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    dump(vectorizer, path)
    return vectorizer, X


def inference_time(X: list[list[str]], model, vectorizer=None, dim_reducer=None, subj_detector=None, subj_vectorizer=None):
    start_time = time()
    if subj_vectorizer and subj_detector:
        subj_features = []
        for i, doc in enumerate(X):
            sents = [" ".join(sent) for sent in doc]
            vectors = subj_vectorizer.transform(sents)
            y_hat = subj_detector.predict(vectors)
            subj_features.append(1 if np.count_nonzero(
                np.array(y_hat)) >= len(y_hat) else 0)
        subj_features = np.array(subj_features)
    if vectorizer:
        X = vectorizer.transform(mr2str(X))
    if isinstance(X, csr_matrix):
        X = X.toarray()
    if dim_reducer:
        X = dim_reducer.transform(X)
    if isinstance(X, csr_matrix):
        X = X.toarray()
    if subj_vectorizer and subj_detector:
        X = hconcat(X, np.expand_dims(subj_features, axis=-1))
    model.predict(X)
    elapsed = time() - start_time
    return f"{int(elapsed // 60)}m:{int(elapsed % 60)}s"


def print_time(start_time: time, end_time:time = None, message:str = 'Done in', out_log: bool = False):
    '''
    Prints to terminal (if :param out_log: = False) or returns a string (if :param out_log: = True)
    :param message: followed by minute(s), second(s).
        - :param start_time: start time of process
        - :param end_time: time at the end of process. Defaults to time() if not provided.
        - :param message: message to display in front of <min(s), sec(s)>
    '''
    
    end_time = time() if end_time is None else end_time
    mins = int( (end_time - start_time) // 60)
    secs = int( (end_time - start_time) %  60)
    
    if mins <= 0:
        if out_log: return f'{message} {secs} second' if secs == 1 else f'{message} {secs} seconds'
        else: print(f'{message} {secs} second' if secs == 1 else f'{message} {secs} seconds')
    
    elif mins == 1:
        if out_log: return f'{message} {mins} minute, {secs} second' if secs == 1 else f'{message} {mins} minute, {secs} seconds'
        else: print(f'{message} {mins} minute, {secs} second' if secs == 1 else f'{message} {mins} minute, {secs} seconds')
    
    else:
        if out_log: return f'{message} {mins} minutes, {secs} seconds' if secs == 1 else f'{message} {mins} minutes, {secs} seconds'
        else: print(f'{message} {mins} minutes, {secs} seconds' if secs == 1 else f'{message} {mins} minutes, {secs} seconds')


def plot_f1_score_results(scores, clf_name: str, path_to_save: os.path, show: bool = False) -> None:
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        plt.title(f'{clf_name} - F1 Score')
        plt.plot(np.sort(scores), scores)
        if show: 
            plt.show()
            plt.close()
        else: 
            plt.savefig(path_to_save+f'{clf_name}_f1.png')
            plt.close()