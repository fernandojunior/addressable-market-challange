# utils.py

from config import *


def load_csv_as_dataframe(filename):
    df = spark.read\
        .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
        .option('header', 'true')\
        .option('inferSchema', 'true')\
        .load(filename)

    df = checkpoint(df, filename + '.parquet')

    return df



def load_dataframe_from_url(link_to_data):
    filename = wget.download(link_to_data)

    df = spark.read\
        .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
        .option('header', 'true')\
        .option('inferSchema', 'true')\
        .load(filename)

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
