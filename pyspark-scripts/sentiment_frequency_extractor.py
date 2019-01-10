from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, avg
import sys
from functools import reduce

# Sentiment extractor, based on a set of vocabulary provided in a one-word-per-line text file.
# Kévin Dumanoir and Lea Uhliarova, Stockholm University, BIGDATA HT2018
# MIT License

if __name__ == "__main__":
    sc = SparkContext(appName="Sentiment Extractor")
    sqlContext = SQLContext(sc)

    # Part I : Extracting tokens from tweets

    tweet_files = sc.wholeTextFiles('hdfs://quickstart.cloudera:8020/user/cloudera/final_project/' + sys.argv[1] + '/*')
    tweets = sqlContext.read.option("multiline", "true").json(tweet_files.values())

    tokenizer = Tokenizer(inputCol='text', outputCol='words')
    tokenized_tweets = tokenizer.transform(tweets)

    # Part II : Take the reference vocabulary

    vocab_file = sc.textFile('hdfs://quickstart.cloudera:8020/user/cloudera/final_project/' + sys.argv[2])
    vocab_df = vocab_file.map(lambda str: ([str],)).toDF(['words'])

    # Setup the CountVectorizer

    cv = CountVectorizer(inputCol='words', outputCol='BoW')
    cv_model = cv.fit(vocab_df)

    # Part III : Count words in tweets from vocabulary and create frequency and create a new DataFrame

    tweets_with_bow = cv_model.transform(tokenized_tweets)
    tweets_stats = tweets_with_bow.map(lambda tuple: (tuple['BoW'].values.sum().item(), len(tuple['words']), tuple['timestamp']))
    tweets_stats = tweets_stats.map(lambda tuple: tuple + (tuple[0]/tuple[1],))
    tweets_stats_df = tweets_stats.toDF(['vocab_words_count', 'words_count', 'timestamp', 'frequency'])

    # Part IV : Saving as CSV... or calculating "density"

    if len(sys.argv) > 3: # If we directly want to get the frequency average, just add a third argument containing anything.
    	tweets_stats_avg_freq = tweets_stats_df.agg(avg(col('frequency')))
    	tweets_stats_avg_freq.toPandas().to_csv('MEAN_' + sys.argv[2].split('.')[0] + '_frequency_' + sys.argv[1] + '.csv')
    else:
    	tweets_stats_df.toPandas().to_csv(sys.argv[2].split('.')[0] + '_frequency_' + sys.argv[1] + '.csv')
