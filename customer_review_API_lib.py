import pandas as pd

from buffer import Buffer
from customer_review_lib import *


def generate_test_dataset(context, toys_file, apparel_file, min_words=10, n_reviews=25000):

    context.value = dict(
        toys_df=pd.read_csv(toys_file, sep='\t', usecols=['product_category', 'review_body'], nrows=3000000),
        apparel_df=pd.read_csv(apparel_file, sep='\t', usecols=['product_category', 'review_body'], nrows=3000000)
    )

    impute_context_dataframe(context=context, df_keyword='apparel_df', min_words=min_words, n_reviews=n_reviews)
    impute_context_dataframe(context=context, df_keyword='toys_df', min_words=min_words, n_reviews=n_reviews)

    context.value = pd.concat([context.value['toys_df'], context.value['apparel_df']])

    context.value['product_category'] = [
        1 if category == 'Apparel' else 0 for category in context.value['product_category']
    ]

    context.value = context.value.sample(frac=1).reset_index(drop=True)

    return


def impute_context_dataframe(context, df_keyword, min_words=10, n_reviews=25000):

    context.value[df_keyword]['num_words'] = [
        len(str(review).split(' ')) for review in context.value[df_keyword]['review_body']
    ]
    context.value[df_keyword] = context.value[df_keyword].query('num_words > @min_words')

    context.value[df_keyword] = context.value[df_keyword][['product_category', 'review_body']]

    context.value[df_keyword] = context.value[df_keyword].dropna()

    context.value[df_keyword] = context.value[df_keyword][:n_reviews]

    return


def run_test_predictions(toys_file, apparel_file, pipeline, gridsearch_args, min_words=10, n_reviews=25000):

    context = Buffer()

    generate_test_dataset(
        context=context,
        toys_file=toys_file,
        apparel_file=apparel_file,
        min_words=min_words,
        n_reviews=n_reviews
    )

    test_prediction(context, pipeline, gridsearch_args)

    return
