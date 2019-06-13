from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
from buffer import Buffer


DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/FinalProject/data/'

REVIEW_COUNT = 10000

context = Buffer()

context.value = pd.read_csv(
        '{}{}'.format(DIRECTORY, 'amazon_reviews_us_Apparel_v1_00.tsv'),
        sep='\t',
        usecols=['star_rating', 'review_body'],
        nrows=500000
)

context.value = context.value.dropna()


context.value['num_words'] = [len(review.split(' ')) for review in context.value['review_body']]
context.value = context.value.query('num_words > 12')
context.value = context.value[['star_rating', 'review_body']]


context.value = {
    '1': context.value.query('star_rating == 1')[:REVIEW_COUNT],
    '2': context.value.query('star_rating == 2')[:REVIEW_COUNT],
    '3': context.value.query('star_rating == 3')[:REVIEW_COUNT],
    '4': context.value.query('star_rating == 4')[:REVIEW_COUNT],
    '5': context.value.query('star_rating == 5')[:REVIEW_COUNT]
}

context.value = pd.concat([
    context.value['1'],
    context.value['2'],
    context.value['3'],
    context.value['4'],
    context.value['5'],
])

print('Finished reading in data...')

pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(norm='l2', max_df=0.7, min_df=5, ngram_range=(1, 1), stop_words='english')),
        ('feature-extract', SelectKBest(chi2, k=15)),
        ('clf', AdaBoostClassifier(
            base_estimator=LogisticRegression(
                C=0.2,
                class_weight='balanced',
                solver='liblinear'
            ),
            n_estimators=10,
            learning_rate=5
        ))
    ])

search = GridSearchCV(
    estimator=pipeline,
    param_grid={
        'clf__base_estimator__C': [0.1, 0.2, 0.5, 1]
    },
    scoring='accuracy'
)

print('Pipeline and gridsearch configured...')

search.fit(context.value['review_body'], context.value['star_rating'])

print("Best parameters set found on development set:")
print(search.best_params_)
print("Grid scores on development set:")
print(search.best_score_)


X_train, X_test, y_train, y_test = train_test_split(context.value['review_body'], context.value['star_rating'], test_size=0.2)

best_estimator = search.best_estimator_
best_estimator.fit(X_train, y_train)

y_pred = best_estimator.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))

