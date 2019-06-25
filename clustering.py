# clustering.py

from config import *
from viz import line_plot

from pyspark.ml.feature import VectorAssembler
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
