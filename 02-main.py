# Shared Imports
from copy import deepcopy
import os
from matplotlib.pyplot import plot
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
# Load Dataset
from joblib import load
from src.utils import get_movie_reviews_dataset, fit_transform_save, switch_vectorizer, hconcat
# 2-stage clf
from src.utils import TwoStageClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from src.utils import inference_time
# Timing
from time import time
from src.utils import print_time
# Plotting
from src.utils import plot_f1_score_results




def main(
    first_stage_vectorizer_name='count',
    second_stage_vectorizer_name='tfidf',
    subj_det='filter',
    subj_detection_name='count_multinomial',
    reduce_dim = True,
    kfold_split_size=5,
    ):

    start_time = time()

    # checks
    assert subj_det in ('filter', 'aggregate'), 'subj_det should be either `filter` or `aggregate`'
    assert len(subj_detection_name.split('_')) == 2
    assert subj_detection_name.split('_')[0] in ('count', 'tfidf')
    assert subj_detection_name.split('_')[1] in ('multinomial', 'bernoulli')


    # download possible missing nltk package: `perceptron_tagger`
    nltk.download('averaged_perceptron_tagger')

    # download dataset
    # movie reviews
    pos, neg = get_movie_reviews_dataset(mark_negs=False)
    movie_reviews_ds = list(neg + pos)

    # labels for movie reviews
    y = np.array([0] * len(neg) + [1] * len(pos))
    # print("Labels shape: ", y.shape)

    # rejecting objective sentences using subjectivity detection
    path_to_models = 'tmp/models/'
    # vectorizer
    subj_vectorizer_path = os.path.join(path_to_models, f"{subj_detection_name}_subj_det_vectorizer.joblib")
    if not os.path.exists(subj_vectorizer_path):
        raise FileNotFoundError(f'Cannot find file {subj_vectorizer_path}_subj_det_vectorizer.joblib')
    subj_vectorizer = load(subj_vectorizer_path)
    # detector
    subj_detector_path = os.path.join(path_to_models, f'{subj_detection_name}_subj_det_model.joblib')
    subj_detector = load(subj_detector_path)

    filter_mins, filter_secs = 0, 0
    if subj_det == 'filter':
        from itertools import compress
        # classify each and every sentence in Movie Reviews dataset,
        # then prune objective ones.
        filter_start = time()
        removed, total = 0, 0
        for i, doc in tqdm(enumerate(deepcopy(movie_reviews_ds)), total=len(movie_reviews_ds)):
            sents = [" ".join(sent) for sent in doc]
            vectors = subj_vectorizer.transform(sents)
            y_hat = subj_detector.predict(vectors)
            movie_reviews_ds[i] = list(compress(doc, y_hat))
            removed += len(doc) - len(movie_reviews_ds[i])
            total += len(doc)
        print(f'Removed {removed}/{total} using subjectivity detection filtering.')
        filter_elapsed = time() - filter_start
        filter_mins = int(filter_elapsed // 60)
        filter_secs = int(filter_elapsed %  60)


    # representing document for FIRST STAGE vectorizer
    first_stage_vectorizer = switch_vectorizer(first_stage_vectorizer_name)
    first_stage_vectorizer_path = os.path.join(
        path_to_models, "first_stage_vectorizer.joblib")
    first_stage_vectorizer, X = fit_transform_save(
        first_stage_vectorizer, movie_reviews_ds, first_stage_vectorizer_path)

    # representing document for SECOND STAGE vectorizer
    second_stage_vectorizer = switch_vectorizer(second_stage_vectorizer_name)
    second_stage_vectorizer_path = os.path.join(
        path_to_models, "second_stage_vectorizer.joblib")
    second_stage_vectorizer, X_second_stage = fit_transform_save(
        second_stage_vectorizer, movie_reviews_ds, second_stage_vectorizer_path)

    # instantiate 2-stage classifier w/ pretrained first-second stage vectorizers
    two_stage_clf_params = {
        "first_stage_vectorizer_path": first_stage_vectorizer_path,
        "second_stage_vectorizer_path": second_stage_vectorizer_path,
        "reduce_dim": reduce_dim
        }
    
    if subj_det == "aggregate":
        two_stage_clf_params["use_subjectivity"] = True
        two_stage_clf_params["subj_vectorizer_path"] = subj_vectorizer_path
        two_stage_clf_params["subj_detector_path"] = subj_detector_path

    
    ########## TWO-STAGE-CLASSIFIER ##########
    clf = TwoStageClassifier(**two_stage_clf_params)
   
    # Evaluate
    scores = cross_validate(clf, movie_reviews_ds, y,
                            cv=StratifiedKFold(n_splits=kfold_split_size),
                            scoring=['f1_micro'],
                            return_estimator=True)
    

    # Save F1 Score Results
    PATH_TO_OUTPUTS = 'tmp/outputs/'
    #plot_f1_score_results(scores=scores['test_f1_micro'], path_to_save=PATH_TO_OUTPUTS, clf_name='two_stage_clf')

    two_stage_clf_average = sum( scores['test_f1_micro']) / len(scores['test_f1_micro'] )
    print("Two Stage Classifier F1 Score: {:.3f}".format(
        two_stage_clf_average))

    # Inference Time
    best_2_stage = scores["estimator"][np.argmax( np.array(scores["test_f1_micro"]) )]
    two_stage_clf_elapsed = inference_time(movie_reviews_ds, best_2_stage)

    
    ########## MULTINOMIAL-NAIVE-BAYES ##########
    # Evaluate
    scores = cross_validate(clf.first_stage_clf, X, y,
                            cv=StratifiedKFold(n_splits=kfold_split_size),
                            scoring=['f1_micro'],
                            n_jobs=-1,
                            return_estimator=True)

    # Save F1 Score Results
    #plot_f1_score_results(scores=scores['test_f1_micro'], path_to_save=PATH_TO_OUTPUTS, clf_name='naive_bayes_clf')

    naive_bayes_average = sum( scores['test_f1_micro'] ) / len( scores['test_f1_micro'] )
    print("Multinomial Naive Bayes F1 Score: {:.3f}".format(naive_bayes_average))
    
    # Inference Time
    naive_bayes_best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    naive_bayes_elapsed = inference_time( movie_reviews_ds, naive_bayes_best, first_stage_vectorizer )
    if subj_det == "filter":
        naive_bayes_mins = int(naive_bayes_elapsed.split(":")[
                                  0][:-1]) + filter_mins
        naive_bayes_secs = int(naive_bayes_elapsed.split(":")[
                                  1][:-1]) + filter_secs
        naive_bayes_elapsed = f"{naive_bayes_mins}m:{naive_bayes_secs}s"


    ########## SVC ##########
    # Evaluate
    if reduce_dim:
        X_second_stage = best_2_stage.dim_reducer.transform(X_second_stage)

    if subj_det == "aggregate":
        subj_features = []
        for i, doc in enumerate(deepcopy(movie_reviews_ds)):
            sents = [" ".join(sent) for sent in doc]
            vectors = best_2_stage.subj_vectorizer.transform(sents)
            y_pred = best_2_stage.subj_detector.predict(vectors)
            subj_features.append(1 if np.count_nonzero(
                np.array(y_pred)) >= len(y_pred) else 0)
        subj_features = np.array(subj_features)
        X_second_stage = hconcat( X_second_stage, np.expand_dims(subj_features, axis=-1) )

    scores = cross_validate(clf.second_stage_clf, X_second_stage, y,
                            cv=StratifiedKFold(n_splits=kfold_split_size),
                            scoring=['f1_micro'],
                            return_estimator=True,
                            n_jobs=-1)

    # Save F1 Score Results
    #plot_f1_score_results(scores=scores['test_f1_micro'], path_to_save=PATH_TO_OUTPUTS, clf_name='svc_clf')

    svc_average = sum(scores['test_f1_micro']) / len(scores['test_f1_micro'])
    print("SVC F1 Score: {:.3f}".format(svc_average))

    # Inference Time
    svc_best = scores["estimator"][np.argmax(np.array(scores["test_f1_micro"]))]
    svc_elapsed = inference_time(movie_reviews_ds, svc_best, second_stage_vectorizer,
                                   dim_reducer=best_2_stage.dim_reducer if reduce_dim else None,
                                   subj_detector=best_2_stage.subj_detector if subj_det == "aggregate" else None,
                                   subj_vectorizer=best_2_stage.subj_vectorizer if subj_det == "aggregate" else None)

    # total execution time
    print_time(start_time=start_time, message='Done in')

    # return performances
    return [
        first_stage_vectorizer_name,
        second_stage_vectorizer_name,
        subj_det,
        reduce_dim,
        round(two_stage_clf_average, 3),
        round(naive_bayes_average, 3),
        round(svc_average, 3),
        two_stage_clf_elapsed,
        naive_bayes_elapsed,
        svc_elapsed,
    ]




if __name__ == '__main__':
    
    # experiment with different architectures
    first_stage_vec_bets = ["diffposneg", "count"]
    second_stage_vec_bets = [["count", "tfidf"], ["tfidf"]]
    subj_det_bets = ["aggregate", "filter"]
    dim_red_bets = [True, False]

    summary = []
    for i, first_stage_vec_bet in enumerate(first_stage_vec_bets):
        for second_stage_vec_bet in second_stage_vec_bets[i]:
            for subj_det_bet in subj_det_bets:
                for dim_red_bet in dim_red_bets:
                    params = {
                        "first_stage_vectorizer_name": first_stage_vec_bet,
                        "second_stage_vectorizer_name": second_stage_vec_bet,
                        "subj_det": subj_det_bet,
                        "reduce_dim": dim_red_bet
                    }
                    data = main(**params)
                    summary.append(data)
    print(summary)

    # save experiment results
    path_to_summary = 'tmp/summary/'
    summary_save_path = os.path.join(path_to_summary, 'summary.csv')
    if not os.path.isdir(path_to_summary):
        os.makedirs(path_to_summary)
    print(f'Saving summary at: {summary_save_path}')

    cols = ["Vec 1", "Vec 2", "Subj Det", "Reduce Dim", "Two Stage F1", "Multinomial F1",
               "SVC F1", "Two Stage Elaps", "Multinomial Elaps", "SVC Elaps"]

    pd.DataFrame(summary, columns=cols).to_csv(summary_save_path, index=False)
