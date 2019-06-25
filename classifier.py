# classifier.py

import inspect
from functools import reduce

from config import *

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import DecisionTreeClassifier


def generate_resample_params_list(data, label_column):
    '''
    Generates the parameters for each label class to use in resample function
    '''
    params_list = data.groupBy(label_column).agg(F.count('*').alias('freq'))

    max_freq = int(params_list.agg(F.mean('freq').alias('max_freq')).collect()[0]['max_freq'])
    print('Max items for each label class:')
    print(max_freq)

    params_list = params_list.withColumn('max_freq', F.lit(max_freq))
    params_list = params_list.withColumn('fraction', F.round((max_freq / F.col('freq')), 2))
    params_list = params_list.withColumn('fraction', F.round(F.col('fraction') * F.col('freq')) / F.col('freq'))
    params_list = params_list.withColumn('with_replacement', F.col('freq') < max_freq)
    params_list = list(i.asDict() for i in params_list.collect())
    return params_list


def resample(data, label_column, seed=None):
    '''
    Performs data resampling according to the frequency of each class of the label to deal with class imbalance.
    It uses both upsampling and downsampling methods:
        - If a frequency of a label class is smaller than the average frequency use upsampling (with substitution)
        - otherwise use downsampling (without substitution).

    The main goal of this method is to avoid noise increasement by copying / adding data randomly via upsampling (pseudo-random).

    Ref:
        https://stackoverflow.com/a/53990745/4159153
    '''
    class_params_list = generate_resample_params_list(data, label_column)

    assert(label_column in data.columns)

    sample_dataframes = []

    for i, class_params in enumerate(class_params_list):
        sample_class = data.filter(F.col(label_column) == class_params[label_column])

        if class_params[label_column] is None or class_params['fraction'] in [None, 0.0]:
            continue

        sample_class = sample_class.sample(class_params['with_replacement'], float(class_params['fraction']), seed=seed)
        sample_dataframes.append(sample_class)

    return reduce(DataFrame.unionAll, sample_dataframes)


def assemble_vector(dataframe, input_cols, output_col):
    '''
    Combines a given list of columns into a single vector column in a a pyspark dataframe

    https://spark.apache.org/docs/latest/ml-features.html#vectorassembler
    '''
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    return assembler.transform(dataframe.na.drop())


def build_param_grid_maps(classifier, param_grid_dict):
    '''
    Converts a grid search parameters from python dict format to ParamGridBuilder.build() result

    https://spark.apache.org/docs/2.2.0/ml-tuning.html
    https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.ml.tuning.ParamGridBuilder
    https://spark.apache.org/docs/2.2.0/api/scala/index.html#org.apache.spark.ml.param.ParamMap
    '''
    estimator_param_maps = ParamGridBuilder()

    # https://stackoverflow.com/a/9058322/4159153s
    classifier_attributes = inspect.getmembers(classifier, lambda a:not(inspect.isroutine(a)))
    classifier_attributes = [a for a in classifier_attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
    classifier_attributes = dict(classifier_attributes)

    for param_key, param_value in param_grid_dict.items():
        estimator_param_maps.addGrid(classifier_attributes[param_key], param_value)

    return estimator_param_maps.build()


def train_multiclass_classifier(classifier_cls, training, testing, feature_col_list, label_col, param_grid, metric='accuracy'):
    '''
    Trains a tunned multiclass classifier using cross-validation and grid search params
    '''
    features_col = 'features'
    prediction_col= 'prediction'

    assert(features_col not in training.columns)
    assert(features_col not in testing.columns)
    assert(features_col not in feature_col_list)
    assert(features_col != label_col)
    assert(isinstance(param_grid, dict))

    # include features into a single column vector for training and testing purpose
    training = assemble_vector(training, feature_col_list, features_col)
    testing = assemble_vector(testing, feature_col_list, features_col)

    # training
    classifier = classifier_cls(featuresCol = features_col, labelCol = label_col)
    param_grid_maps = build_param_grid_maps(classifier, param_grid)
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName=metric) 
    validator = CrossValidator(estimator=classifier, estimatorParamMaps=param_grid_maps, evaluator=evaluator, numFolds=5)
    model = validator.fit(training)

    # evaluation
    testing_predictions = model.transform(testing)
    testing_score = evaluator.evaluate(testing_predictions)

    training_predictions = model.transform(training)
    training_score = evaluator.evaluate(training_predictions)

    best_param_map = model.bestModel.extractParamMap()

    return model, testing_score, training_score, testing_predictions, training_predictions, best_param_map