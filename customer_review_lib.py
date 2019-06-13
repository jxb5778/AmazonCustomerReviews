from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import math


def test_prediction(context, pipeline, gridsearch_args):
    search = GridSearchCV(estimator=pipeline, **gridsearch_args)

    num_reviews = len(context.value['review_body'])

    search.fit(
        context.value['review_body'][:math.floor(num_reviews/2)],
        context.value['product_category'][:math.floor(num_reviews/2)]
    )

    print("Best parameters set found on development set:")
    print(search.best_params_)
    print("Grid scores on development set:")
    print(search.best_score_)

    best_estimator = search.best_estimator_
    best_estimator.fit(
        context.value['review_body'][:math.floor(num_reviews/2)],
        context.value['product_category'][:math.floor(num_reviews/2)]
    )

    y_pred = best_estimator.predict(context.value['review_body'][math.floor(num_reviews/2):])
    print('Classification Report:')
    print(classification_report(context.value['product_category'][math.floor(num_reviews/2):], y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(context.value['product_category'][math.floor(num_reviews/2):], y_pred))

    return
