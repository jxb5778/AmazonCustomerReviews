from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.pipeline import Pipeline

from customer_review_API_lib import run_test_predictions


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/FinalProject/data/'

run_test_predictions(
    toys_file='{}{}'.format(DIRECTORY, 'amazon_reviews_us_Apparel_v1_00.tsv'),
    apparel_file='{}{}'.format(DIRECTORY, 'amazon_reviews_us_Toys_v1_00.tsv'),
    min_words=12,
    n_reviews=200000,
    pipeline=Pipeline([
            ('tfidf', TfidfVectorizer(norm='l2', max_df=0.6, min_df=75, ngram_range=(1, 1), stop_words='english')),
            ('feature-extract', SelectKBest(chi2, k=20)),
            ('clf', AdaBoostClassifier(
                base_estimator=LogisticRegression(
                    C=0.1,
                    class_weight='balanced',
                    solver='liblinear'
                ),
                n_estimators=10,
                learning_rate=5
            ))
        ]),
    gridsearch_args=dict(
        param_grid={
            'clf__n_estimators': [2, 5, 10, 15],
            'clf__learning_rate': [0.1, 1, 5]
        },
        scoring='f1'
    )
)
