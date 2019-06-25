
# coding: utf-8

# Author: Fernando Felix do Nascimento Junior 
# 
# Last update: 24/06/2019

# # Addressable market challange

# This notebook is divided in the following topics:

# Summary:
#     - Load libs and modules
#     - Load raw data sets
#     - Split actual and addressable customers
#     - EDA
#         - Descriptive statistics analsysis
#         - Customer type size analysis
#         - Outlier analysis
#         - Univariate distribution analysis
#     - Model building
#         - Customer segmentation
#         - Decision-tree classifier
#         - Customer scorer (pearson similiarity)
#     - Generate Deliverables
#         - Deliverable 1
#         - Deliverable 2
#         - Sanity Check
#     - Improvements (TODO)
#     - Cluster based classifier and scorer (deprecated)

# It runs on top of IBM Watson Studio with the following hardware config:
#     - 2 Executors: 1 vCPU and 4 GB RAM, Driver: 1 vCPU and 4 GB RAM
#     
# Link (Private):
# - https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/89a5651d-87f7-440d-b6b4-7b831cefcad4/view?projectid=8d5fdbe5-355b-4c65-bbf1-1d1abba0077f&context=wdp

# # Load libs

# In[ ]:


# requirements.txt
get_ipython().system(u'pip install --upgrade wget')
get_ipython().system(u'pip install --upgrade plotly')


# In[ ]:


# config.py

import ibmos2spark
# @hidden_cell
credentials = {
    'endpoint': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'service_id': 'iam-ServiceId-3c5cb5d7-3787-4432-bd0d-8815e65be261',
    'iam_service_endpoint': 'https://iam.bluemix.net/oidc/token',
    'api_key': 'bW3TRvTFioeaJ2FBZYJ3djHqsWlilR2G1eppRuyPRd7z'
}

configuration_name = 'os_25e3d57cda46474ca2767163f7a810b0_configs'
cos = ibmos2spark.CloudObjectStorage(sc, credentials, configuration_name, 'bluemix_cos')


import wget

import numpy as np
import pandas as pd

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import HiveContext

import pyspark.sql.types as T
import pyspark.sql.functions as F

sc = SparkContext.getOrCreate()
sqlContext = HiveContext(sc)
spark = sqlContext.sparkSession

SEED = 27


# In[ ]:


# utils.py


def load_csv_as_dataframe(filename):
    df = spark.read        .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')        .option('header', 'true')        .option('inferSchema', 'true')        .load(filename)

    df = checkpoint(df, filename + '.parquet')

    return df



def load_dataframe_from_url(link_to_data):
    filename = wget.download(link_to_data)

    df = spark.read        .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')        .option('header', 'true')        .option('inferSchema', 'true')        .load(filename)

    df = checkpoint(df, filename + '.parquet')

    return df


def load_dataframe(filename):
    return spark.read.option("mergeSchema", "true").parquet(filename)


def checkpoint(df, filename):
    '''
    Saves a dataframe from memory to a parquet file then read the file
    '''
    df.write.mode('overwrite').parquet(filename)
    return spark.read.option("mergeSchema", "true").parquet(filename)


def inspect_dataframe(df, n=20):
    '''
    Shows data dimension, schema and samples.
    '''
    print('Dimension:', df.count(), len(df.columns))
    print('Schema:')
    df.printSchema()
    print('Sample n={n}:'.format(n=n))
    df.show(n)
    return df


def apply_agg_fn(fn, data, columns):
    '''
    Applies an aggregate function (eg, mean, variance, count, etc.) to a list of columns of a spark dataframe
    '''
    dfagg = data.select(columns).agg(*([fn(c) for c in columns])).toDF(*columns)

    return dfagg.select(*[F.col(c).cast(T.DecimalType(18, 2)).alias(c) for c in dfagg.columns]) # avoid sci notation


def transpose_as_pd(data, index=None):
    '''
    Performs transposition of a spark datraframe and then transforms into a pandas dataframe
    '''
    return data.toPandas().set_index(index).transpose()


def describe(data, columns):
    '''
    Summarizes a spark dataframe applying the following aggregate functions: count, mean, stddev, minmax, var.
    '''
    dfagg = transpose_as_pd(data.select(columns).describe(), index='summary')
    dfvar = apply_agg_fn(F.variance, data, columns).toPandas().rename(index={0: 'var'}).transpose()

    df = pd.concat([dfagg, dfvar], axis=1)
    display(df)
    return data


# In[ ]:


# viz.py

# https://plot.ly/python/apache-spark/
# https://plot.ly/python/offline/
import plotly.offline as pyo
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff

init_notebook_mode(connected=True)


def grouped_bar_plot(df_list, title_list, group_col):
    '''
    https://plot.ly/python/bar-charts/#grouped-bar-chart
    '''
    bar_data = []

    for df, title in zip(df_list, title_list):
        df_by = df.groupBy(group_col).agg(F.count('*').alias('count'))

        #print('Number of items for each group in {title} dataset:'.format(title=title))
        #df_by.show()

        df_by = df_by.collect()

        trace = go.Bar(x = [i[group_col] for i in df_by], y = [i['count'] for i in df_by], name=title)

        bar_data.append(trace)

    layout = go.Layout(barmode='group')

    fig = go.Figure(data=bar_data, layout=layout)
    iplot(fig, filename='grouped-bar')


def boxplots(data, columns):
    '''
    https://plot.ly/python/box-plots/
    https://dataplatform.cloud.ibm.com/exchange/public/entry/view/d80de77f784fed7915c14353512ef14d
    '''
    data_pd = data.select(columns).toPandas()

    traces = []

    for colname in columns:
        traces.append(go.Box(y = data_pd[colname], name = colname))
    
    return iplot(traces)


def dist_plots(data, columns, show_hist=True):
    '''
    - https://plot.ly/python/distplot/
    - https://en.wikipedia.org/wiki/Kernel_density_estimation
    '''
    hist_data = []
    colors = ['#333F44', '#37AA9C', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4', '#94F3E4']

    for colname in columns:
        df = data.select(colname).toPandas()[colname]
        hist_data.append(df)

    fig = ff.create_distplot(hist_data, columns, show_hist=show_hist, show_rug=False)
    fig['layout'].update(title='KDE curve plots')

    iplot(fig, filename='Kernel density estimation curve plots')


def line_plot(x, y, title, x_title, y_title, x_range=None, y_range=None):
    '''
    https://plot.ly/python/line-charts/#simple-line-plot
    '''
    xaxis = dict(title = x_title, range=x_range)
    yaxis = dict(title = y_title, range=y_range)
    layout = dict(title = title, xaxis = xaxis, yaxis = yaxis)
    data = [go.Scatter(x = x, y = y)]
    fig = dict(data=data, layout=layout)
    iplot(fig)


# In[ ]:


# clustering.py

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans


def tsse(df, feature_cols):
    '''
    Total sum of squared error for multivariate data

    https://stackoverflow.com/a/21385702/4159153
    '''
    df = df.agg(*([(F.variance(F.col(colname)) * (F.count('*') - 1)).alias(colname) for colname in feature_cols]))
    df = df.withColumn('tsse', sum(df[colname] for colname in feature_cols))
    return df.select('tsse')


def cluster_data(dataset, feature_cols, k, max_iter=100, seed=None):
    '''
    https://spark.apache.org/docs/2.2.0/ml-clustering.html#k-means
    https://spark.apache.org/docs/latest/ml-features.html#vectorassembler

    TODO use spark pipeline
    '''
    dataset = dataset.drop('features')
    dataset = dataset.drop('prediction')
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

    dataset = assembler.transform(dataset)

    # Trains a k-means model.
    kmeans = KMeans().setK(k).setMaxIter(max_iter).setSeed(seed)
    model = kmeans.fit(dataset)

    # Make predictions
    dataset = model.transform(dataset)

    # Centroids
    centers = model.clusterCenters()

    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse_score = model.computeCost(dataset)

    dataset = dataset.drop('features')
    return {'model': model, 'predictions': dataset, 'centers': centers, 'wssse_score': wssse_score}


def elbow_curve(df, feature_cols, max_k=90, seed=None):
    '''
    Generates a 2-d tuple containing:
    - a set (list) of number of cluster k >= 1 and k <= max_k. 
    - a list containing a within set sum of squared errors (wssse) for each k >= 2 and k <= max_k
        - if k == 1, it returns the total sum of squared errors of the dataframe[feature_cols]
    '''
    k_list = []
    k_scores = []

    for k in range(1, max_k):
        if k == 1:
            score = tsse(df, feature_cols).collect()[0]['tsse']
            print("Total Sum of Squared Errors k=1; no clustered data: {score}".format(score=score))
        else:
            cluster_results = cluster_data(df, feature_cols, k, seed=seed)
            score = cluster_results['wssse_score']
            print("Within Set Sum of Squared Errors k={k}: {score}".format(k=k, score=score))

        k_list.append(k)
        k_scores.append(score)

    return k_list, k_scores


def elbow_curve_plot(k_list, k_scores, variability_reduction_rate=False):
    '''
    Generates an elbow curve
    '''
    if not variability_reduction_rate:
        return line_plot(k_list, k_scores, 'Elbow curve - WSSSE', 'Number of clusters k', 'Within-clusters Set Sum of Squared Errors - WSSSE')

    k_scores = (pd.Series(k_scores)[0] - pd.Series(k_scores)) / pd.Series(k_scores)[0]
    title = 'Elbow curve - variability reduction rate'
    x_title = 'Number of clusters k'
    y_title = 'WSSSE reduction rate over TSSE: (TSSE - WSSSE / TSSE)'
    return line_plot(k_list, k_scores, title, x_title, y_title)


# In[72]:


# classifier.py

import inspect
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
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


# # Load raw data sets

# Let's load the raw datasets from customer CRM and Neoway firmographic.

# In[ ]:


# customer_crm = load_dataframe_from_url('https://gist.githubusercontent.com/fernandojunior/e30bbab8298fb8a57c78f52079503fd8/raw/0f8995acab136324237b17a670fa083db674462d/customer_CRM_2019-05-17.csv')
# neoway_db = load_dataframe_from_url('https://gist.githubusercontent.com/fernandojunior/e30bbab8298fb8a57c78f52079503fd8/raw/0f8995acab136324237b17a670fa083db674462d/Neoway_database_2019-05-17.csv')


# In[88]:


customer_crm = load_csv_as_dataframe('customer_CRM_2019-05-17.csv')
neoway_db = load_csv_as_dataframe('Neoway_database_2019-05-17.csv')


# Inspect customer_crm dataset (dimension, schema, sample):

# In[8]:


customer_crm = inspect_dataframe(customer_crm, n=3)


# Inspect customer_crm dataset (dimension, schema, sample): 

# In[9]:


neoway_db = inspect_dataframe(neoway_db, n=3)


# Notes:
# - The data size, schema and sample correspond the expected format from CSV files.

# # Split actual and addressable customers

# In this section, we will carry out the following activities:
# - Merge raw datasets into a single dataset (all_customers)
# - Remove duplicate data items (distinct)
# - Identify and count actual customers (addressable == False) and addressable customers (addressable == True)
# - Split the data between actual and addressable customers

# In[21]:


# merge data and create new column to identify actual or addresable customer
customer_crm = customer_crm.withColumn('addressable', F.lit(False))
all_customers = neoway_db.join(customer_crm, ['id'], 'left_outer')
all_customers = all_customers.withColumn('addressable', F.coalesce(F.col('addressable'), F.lit(True)))

all_customers = all_customers.distinct() # removendo dados duplicados

all_customers.groupBy('addressable').agg(F.count('*')).show()


# Notes:
# - Actual customer size: 18001
# - Addressable customer size: 2000

# Now, let's split the all_customers dataset into actual_customers and addressable_customers datasets.

# In[22]:


# split all customers into actual_customers and addressable_customers
actual_customers = all_customers.filter(F.col('addressable') == False)
addressable_customers = all_customers.filter(F.col('addressable') == True)


# In[23]:


# checkpoints to refresh spark DAG
all_customers = checkpoint(all_customers, 'all_customers.parquet')
actual_customers = checkpoint(actual_customers, 'actual_customers.parquet')
addressable_customers = checkpoint(addressable_customers, 'addressable_customers.parquet')


# # EDA

# Before performing EDA, let's identify the features and customer types:

# In[24]:


feature_cols = ['feat_0', 'feat_1', 'feat_2', 'feat_3',  'feat_4',  'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9']
customer_type_list = [i.type for i in all_customers.select('type').distinct().filter(F.col('type').isNotNull()).collect()]

print('Feature columns: ', feature_cols)
print('Customer types:', customer_type_list)


# ## Descriptive statistics analsysis

# Let's summarize the descriptive statistics for each dataset: actual_customers, addressable_customers.

# In[26]:


print('Descriptive statistics for actual customers dataset:')
actual_customers = describe(actual_customers, feature_cols)

print('Descriptive statistics for addressable customers dataset:')
addressable_customers = describe(addressable_customers, feature_cols)


# Notes:
# - The features have the same data size for each dataset. This indicates that features don't have missing values (null, nan)
# - The basic statistics of the addressable_customers and actual_customers datasets are similar, except for variance.
# - The mean and variance of actual_customers is close to 0 and 1, respectively. This indicates that the data was standardized.
# - The addressable_customers dataset appeared to have been standardized using z-score parameters (mean, variance) from actual_customers dataset.

# ## Customer type size analysis
# 
# Let's analyse the number of customers by type for each dataset: actual customers, addressable customers, all customers.

# In[28]:


grouped_bar_plot([actual_customers, addressable_customers, all_customers], ['actual customers', 'addressable customers', 'all customers'], 'type')


# Notes:
# - The actual_customers dataset doesn't have all customer types.
#      - actual_customers types: church, supermarket, restaurant, school
#      - addressable_customers types: church, supermarket, restaurant, school, c-store, bar, hair saloon
# - The type sizes aren't balanced for both datasets
# - Therefore, in our modeling, we can't perform segmentation based on type field. We will use k-means clustering to find the most fitting segments derived from customer characteristics.

# ## Outlier analysis
# 
# Let's perform a simple outlier analysis by applying boxplot for each customer type on merged dataset (actual dataset + addressable dataset).

# In[29]:


for customer_type in customer_type_list:
    print('Boxplot for ', customer_type)
    boxplots(all_customers.filter(F.col('type') == customer_type), feature_cols)


# - Embora o dataset possua outliers, a distribuicao dos outliers nos boxplots nao parecem ter comportamento estranho, como presenca de dados oriundos de erros ou outliers extremos que possam interferir na estatistica dos dados. 
# - Os outliers parecem estar ajudando a discriminar o comportamento de cada tipo de customer. Por exemplo, alguns tipos tem mais outliers em determinadas features
# - Portanto, por enquanto, nenhum tratamento vai ser realizado para remover outliers

# ## Univariate distribution analysis

# In[31]:


dist_plots(all_customers.na.drop(), feature_cols, show_hist=False)


# Notes:
# - By analyzing the kurtosis and skewness of the curves visually, we can say that in general the features are normally distributed.
#     - Some features more than the others.
# - We could apply the Shapiro-Wilk test in each of the features to validate.
# - Therefore, if necessary, we could apply some parametric statistical tests that require the data to be distributed normally [link] (https://www.originlab.com/doc/Origin-Help/Normality-Test)

# # Model building

# ## Customer segmentation

# The main goal of k-means is to decrease the data variability by grouping similar items.
# To find the optimal number of k, we will use a custom elbow method, which aims to analyze the proportion of how much the WSSE decreases over the TSSE for each k.

# In[37]:


k_list, k_scores = elbow_curve(actual_customers, feature_cols, max_k=70, seed=SEED)
elbow_curve_plot(k_list, k_scores, variability_reduction_rate=True)


# Notes:
# - As we can see, the  elbow curve varies between k> = 10 and k <= 12
# - Considering that 70% is a reasonable thrashold for variability reduction, we will choose k = 11 as the optimal number of clusters

# The following figure summarizes the size of the actual customer in each cluster.

# In[38]:


actual_customers = actual_customers.drop('cluster')
cluster_results = cluster_data(actual_customers, feature_cols, 11, seed=SEED)
actual_customers = cluster_results['predictions'].withColumnRenamed('prediction', 'cluster')

#boxplots(actual_customers.groupBy('segment').agg(F.count('*').alias('count')), ['count'])
grouped_bar_plot([actual_customers], ['actual customers'], 'cluster')


# Note:
# - As we can see, two segments can be considered outliers as they have fewer customer sizes.

# ## Decision-tree classifier

# Based on clustered data, we will train a decision tree classisifier:

# - Split training and test datasets
# - handle imbalanced classes hybrid method: upsampling and downsampling (threshold = avg of class freq).
# - analyse cluster label sizes after resampling
# - train a decision tree classifier using cross validation and grid search
# - classfier offline validation using accuracy

# In[46]:


label_col = 'cluster'
prediction_label = 'prediction'

(training, testing) = actual_customers.randomSplit([0.7, 0.3], seed=SEED)

training = checkpoint(training, 'training.parquet')
testing = checkpoint(testing, 'testing.parquet')

training = resample(training, label_col, seed=SEED)
training = checkpoint(training, 'training_resampled.parquet')

grouped_bar_plot([training], ['training'], label_col)

param_grid = {
    'maxDepth': [5, 8, 13],
    'maxBins': [13, 21, 34],
    'impurity': ['entropy', 'gini'],
    'seed': [SEED]
}

(model, testing_score, training_score, testing, training, best_param_map) = train_multiclass_classifier(
    DecisionTreeClassifier,
    training,
    testing,
    feature_cols,
    label_col,
    param_grid,
    metric='accuracy')

print('testing_score vs training_score:', testing_score, training_score)


# Notes:
# - The classifier performs well without overfitting and underfitting

# ## Customer scorer

# In this section we will perform the following tasks:
# - predict the addressable customers' clusters  using previously trained classifier
# - estimate score by applying pearson correlation between predict cluster centroids and features for each  addressable customer.

# In[73]:


@F.udf("double")
def udf_corr_scoring(point, cluster):
    point = pd.Series([float(i) for i in point])
    corr_series = centers.corrwith(point)
    result = corr_series[int(cluster)]

    return float(result)

# predict addressable_customers' clusters and estimate correlation core 
addressable_customers = addressable_customers.drop('features', 'prediction', 'rawPrediction', 'probability')
addressable_customers = model.transform(assemble_vector(addressable_customers, feature_cols, 'features'))
addressable_customers = addressable_customers.withColumn('score', udf_corr_scoring(F.array(feature_cols), prediction_label))

addressable_customers.show(5)


# # Generate Deliverables

# In[93]:


def save_deliberable(df, filename):
    filename = cos.url(filename, 'potentialmarketranking-donotdelete-pr-cej2kccafd4zxc')
    df.repartition(1).write.mode('overwrite').option("header", "true").csv(filename)


def load_deliberable(filename):
    filename = cos.url(filename, 'potentialmarketranking-donotdelete-pr-cej2kccafd4zxc')
    
    return spark.read      .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')      .option('header', 'true')      .load(filename)


# Finally, let's generate the challange deliverables.

# ## Deliverable 1

# In[78]:


save_deliberable(addressable_customers.select('id'), 'addressable_ids.csv')


# ## Deliverable 2

# In[81]:


save_deliberable(training.select('id'), 'training_ids.csv')
save_deliberable(testing.select('id'), 'testing_ids.csv')
save_deliberable(addressable_customers.select('id', 'score').orderBy(F.desc('score')), 'addressable_ranking.csv')
print('OK')


# ## Sanity Check

# Let's check if the deliverable datasets were generated and saved correctly.

# In[94]:


print('training_ids.csv')
load_deliberable('training_ids.csv').show(5)

print('testing_ids.csv')
load_deliberable('testing_ids.csv').show()

print('addressable_ranking.csv')
load_deliberable('addressable_ranking.csv').show()


# # Improvements
# 
# TODO:
#      - Redistribute customers from small clusters into other clusters
#      - persona characterization (improve explicability) - analyze centroids to describe and differentiate each cluster behavior
#      - create files to store python modules: utils, config, clustering, etc.
#      - save models (clustering, classsifier)
#      - use spark pipelines
#      - compute processing time for clustering and predictions
#      - try to use euclidian distance to compute score

# # Cluster based classifier and scorer (deprecated)

# Clustering is not able to classify instances of companies in itself. Instead, a simple classification model was built on top of the result of the clustering ([reference](https://pdfs.semanticscholar.org/9813/21d9f06a51110d6585bce4dcc14f624acbc0.pdf)). This was conducted with the following algorithm
# 
# 1. Split customers c1, c2, . . . , c644 into a training set (80%) and a test set (20%) 
# 2. Cluster the training-data = actual-customers using the K-means algorithm
# 3. Calculate the centroids for each cluster by taking the distance to the data point in the training set
# 4. Predict the closest cluster for training-set (actual customers) using pearson similiraty method
# 5. Predict the closest cluster for unlabeled companies (addressable customers = testing-set) using pearson similiraty method
# 6. Compare the predicted centroids for both training-set and unlabeled companies
# 7. Rank addressable customers using pearson similiraty

# In[40]:


centers = pd.DataFrame(cluster_results['centers']).T


def assemble_vector(dataframe, input_cols, output_col):
    '''
    Combine a given list of columns into a single vector column in a a pyspark dataframe

    https://spark.apache.org/docs/latest/ml-features.html#vectorassembler
    '''
    assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
    return assembler.transform(dataframe.na.drop())


def find_cluster(point):    
    '''
    Find the most correlated cluster of a given point

    1. Compute the pairwise correlation between each row of a clusters' centers (pd.Dataframe) vs a dimensional point (pd.Series).

    2. Rank the correlations

    3. Return a tuple containing: the most correlated cluster of point, correlation estimation.

    Ref:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corrwith.html
        - https://stackoverflow.com/a/38468711/4159153
        - https://stackoverflow.com/a/35459076/4159153

    TODO: Use euclidian distance to compute score
        ```
        from scipy.spatial import distance
        a = (1, 2, 3)
        b = (4, 5, 6)
        distance.euclidean(a, b)
        ```
    '''
    point = pd.Series([float(i) for i in point])
    corr_series = centers.corrwith(point)
    corr_series = corr_series.sort_values(ascending=False)

    result = list(zip(corr_series.index, corr_series))[0] #  first result from ranking

    return (float(result[0]), float(result[1])) # (cluster, correlation)


@F.udf("double")
def udf_estimate_cluster(point):
    return find_cluster(point)[0]


@F.udf("double")
def udf_estimate_score(point):
    return find_cluster(point)[1]


def predict_cluster(df):
    df = assemble_vector(df, feature_cols, 'features')
    df = df.withColumn('prediction', udf_estimate_cluster(F.col('features')))
    df = df.withColumn('score', udf_estimate_score(F.col('features')))
    df = df.drop('features')

    return df


training = checkpoint(actual_customers.distinct(), 'training.parquet')
testing = checkpoint(addressable_customers.distinct(), 'testing.parquet')

training = predict_cluster(training)
testing = predict_cluster(testing)

print('Actual customers predictions based on cluster centroids correlation')
training.show(3)

print('Addressable customers predictions based on cluster centroids correlation')
testing.show(3)


# ## Off-line evaluation

# In[43]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_multiclass_model(predictions_data, label_col, prediction_col='prediction', metric_name='accuracy'):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName=metric_name)
    score = evaluator.evaluate(predictions_data)
    return score


eval_metrics = ['accuracy', 'weightedPrecision', 'weightedRecall']
label_col = 'cluster'
prediction_col = 'prediction'

for metric in eval_metrics:
    global_train_score = evaluate_multiclass_model(training, label_col, prediction_col)
    print('{metric} for training : {global_train_score}'.format(metric=metric,global_train_score=global_train_score))


# Sanity check

# In[44]:


training.groupBy('cluster').agg({x: "avg" for x in feature_cols}).show()
training.groupBy('prediction').agg({x: "avg" for x in feature_cols}).show()
testing.groupBy('prediction').agg({x: "avg" for x in feature_cols}).show()

