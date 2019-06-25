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
