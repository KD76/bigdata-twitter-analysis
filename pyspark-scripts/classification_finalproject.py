from pyspark import SparkContext
from pyspark.sql import SQLContext, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import sys
from functools import reduce

# Tweets classifier
# Based on "Assignment 3" work done at BIGDATA course, Stockholm University
# Kévin Dumanoir and Lea Uhliarova, Stockholm University, BIGDATA HT2018
# MIT License 

# Function for combining multiple DataFrames row-wise
def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

if __name__ == "__main__":
    # Create a SparkContext and an SQLContext
    sc = SparkContext(appName="Tweets Classification")
    sqlContext = SQLContext(sc)

    # Load data
    # wholeTextFiles(path, [...]) reads a directory of text files from a filesystem
    # Each file is read as a single record and returned in a key-value pair
    # The key is the path and the value is the content of each file
    
    tweet_files = sc.wholeTextFiles('hdfs://quickstart.cloudera:8020/user/cloudera/final_project/*/*')

    # Tweets are stored in form of multiple JSON Array files. Before continuing we need to extract columns and rows from these json files.
    tweets = sqlContext.read.option("multiline", "true").json(tweet_files.values())
    # We obtain directly a DataFrame (wholeTextFiles method and map provides an RDD), which constitutes our tweets.
    # Now, we have to extract the information.

    #extract information: is the text file from trump or not and create key-value pairs
    #where (1.0, textfile) if the text file is from trump tweets and (0.0, textfile) otherwise
    tweets_categorized = tweets.map(lambda tuple: (1.0 if 'trump' in tuple["screen_name"].lower() else 0.0, tuple["text"]))
    dataset = tweets_categorized.toDF(['class_label', 'tweet'])

    # ----- PART II: FEATURE ENGINEERING -----

    # Tokenize the review text column into a list of words
    tokenizer = Tokenizer(inputCol='tweet', outputCol='words') 
    words_data = tokenizer.transform(dataset)

    # Randomly split data into a training set, a development set and a test set
    # train = 60% of the data, dev = 20% of the data, test = 20% of the data
    # The random seed should be set to 42
    (train, dev, test) = words_data.randomSplit([.6, .2, .2], seed = 42)

    # TODO: Count the number of instances in, respectively, train, dev and test
    # Print the counts to standard output
#we do not need to count the instances
    #train_instances_count = train.count()
    #dev_instances_count = dev.count()
    #test_instances_count = test.count()

    #print("Number of train instances: ", train_instances_count)
    #print("Number of dev instances: ", dev_instances_count)
    #print("Number of test instances: ", test_instances_count)

    # TODO: Count the number of positive/negative instances in, respectively, train, dev and test
    # Print the class distribution for each to standard output
    # The class distribution should be specified as the % of positive examples
    
    #train_pos_instances_count = train[train.class_label == 1].count()
    #dev_pos_instances_count = dev[dev.class_label == 1].count()
    #test_pos_instances_count = test[test.class_label == 1].count()

    #print("Class distribution of training set: ", train_pos_instances_count*100/train_instances_count)
    #print("Class distribution of dev set: ", dev_pos_instances_count*100/dev_instances_count)
    #print("Class distribution of test set: ", test_pos_instances_count*100/test_instances_count)

    # TODO: Create a stopword list containing the 100 most frequent tokens in the training data
    # Hint: see below for how to convert a list of (word, frequency) tuples to a list of words
    # stopwords = [frequency_tuple[0] for frequency_tuple in list_top100_tokens]
    train_word_counts_sorted = train.rdd.flatMap(lambda row: row.words).map(lambda word : (word, 1)).reduceByKey(lambda a,b: a+b).sortBy(lambda tuple: tuple[1], ascending=False)
    stopwords = [tuple[0] for tuple in train_word_counts_sorted.take(100)]

    remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=stopwords)

    # Remove stopwords from all three subsets
    train_filtered = remover.transform(train)
    dev_filtered = remover.transform(dev)
    test_filtered = remover.transform(test)

    # Transform data to a bag of words representation, only include tokens that have a minimum document frequency of 2
    cv = CountVectorizer(inputCol='words_filtered', outputCol='BoW', minDF=2.0)
    cv_model = cv.fit(train_filtered)
    train_data = cv_model.transform(train_filtered)
    dev_data = cv_model.transform(dev_filtered)
    test_data = cv_model.transform(test_filtered)
    
    # TODO: Print the vocabulary size (to STDOUT) after filtering out stopwords and very rare tokens
    # Hint: Look at the parameters of CountVectorizer
    print("Vocabulary size of the testing set: ", len(cv_model.vocabulary)) 

    # Create a TF-IDF representation of the data
    idf = IDF(inputCol='BoW', outputCol='TFIDF')
    idf_model = idf.fit(train_data)
    train_tfidf = idf_model.transform(train_data)
    dev_tfidf = idf_model.transform(dev_data)
    test_tfidf = idf_model.transform(test_data)

    # ----- PART III: MODEL SELECTION -----

    # Provide information about class labels: needed for model fitting
    # Only needs to be defined once for all models (but included in all pipelines)
    label_indexer = StringIndexer(inputCol = 'class_label', outputCol = 'label')

    # Create an evaluator for binary classification
    # Only needs to be created once, can be reused for all evaluation
    evaluator = BinaryClassificationEvaluator()

    # Train a decision tree with default parameters (including maxDepth=5)
    dt_classifier_default = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=5)

    # Create an ML pipeline for the decision tree model
    dt_pipeline_default = Pipeline(stages=[label_indexer, dt_classifier_default])

    # Apply pipeline and train model
    dt_model_default = dt_pipeline_default.fit(train_tfidf)

    # Apply model on development data
    dt_predictions_default_dev = dt_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_dt_default_dev = evaluator.evaluate(dt_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev))

    # TODO: Check for signs of overfitting (by evaluating the model on the training set)
    # [FIX ME!] Write code below

    dt_predictions_default_train = dt_model_default.transform(train_tfidf)
    auc_dt_default_train = evaluator.evaluate(dt_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train))

    # TODO: Tune the decision tree model by changing one of its hyperparameters
    # Build and evalute decision trees with the following maxDepth values: 3 and 4.

    dt_classifier_default_4 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=4)
    dt_pipeline_default_4 = Pipeline(stages=[label_indexer, dt_classifier_default_4])
    dt_model_default_4 = dt_pipeline_default_4.fit(train_tfidf)
    dt_predictions_default_dev_4 = dt_model_default_4.transform(dev_tfidf)
    auc_dt_default_dev_4 = evaluator.evaluate(dt_predictions_default_dev_4, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev_4))

    dt_predictions_default_train_4 = dt_model_default_4.transform(train_tfidf)
    auc_dt_default_train_4 = evaluator.evaluate(dt_predictions_default_train_4, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train_4))

    dt_classifier_default_3 = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth=3)
    dt_pipeline_default_3 = Pipeline(stages=[label_indexer, dt_classifier_default_3])
    dt_model_default_3 = dt_pipeline_default_3.fit(train_tfidf)
    dt_predictions_default_dev_3 = dt_model_default_3.transform(dev_tfidf)
    auc_dt_default_dev_3 = evaluator.evaluate(dt_predictions_default_dev_3, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev_3))

    dt_predictions_default_train_3 = dt_model_default_3.transform(train_tfidf)
    auc_dt_default_train_3 = evaluator.evaluate(dt_predictions_default_train_3, {evaluator.metricName: 'areaUnderROC'})
    print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train_3))

    # Train a random forest with default parameters (including numTrees=20)
    rf_classifier_default = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=20)

    # Create an ML pipeline for the random forest model
    rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

    # Apply pipeline and train model
    rf_model_default = rf_pipeline_default.fit(train_tfidf)

    # Apply model on development data
    rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

    # Evaluate model using the AUC metric
    auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

    # Print result to standard output
    print('Random Forest, Default Parameters, Development Set, AUC:' + str(auc_rf_default_dev))

    # TODO: Check for signs of overfitting (by evaluating the model on the training set)
    rf_predictions_default_train = rf_model_default.transform(train_tfidf)
    auc_rf_default_train = evaluator.evaluate(rf_predictions_default_train, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, Default Parameters, Training Set, AUC:' + str(auc_rf_default_train))


    # TODO: Tune the random forest model by changing one of its hyperparameters
    # Build and evalute (on the dev set) another random forest with the following numTrees value: 100.
    # [FIX ME!] Write code below

    rf_classifier_default_100 = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)
    rf_pipeline_default_100 = Pipeline(stages=[label_indexer, rf_classifier_default_100])
    rf_model_default_100 = rf_pipeline_default_100.fit(train_tfidf)
    rf_predictions_default_dev_100 = rf_model_default_100.transform(dev_tfidf)
    auc_rf_default_dev_100 = evaluator.evaluate(rf_predictions_default_dev_100, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, Default Parameters, Development Set, AUC:' + str(auc_rf_default_dev_100))

    rf_predictions_default_train_100 = rf_model_default_100.transform(dev_tfidf)
    auc_rf_default_train_100 = evaluator.evaluate(rf_predictions_default_train_100, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, Default Parameters, Training Set, AUC:' + str(auc_rf_default_train_100))

    # ----- PART IV: MODEL EVALUATION -----

    # Create a new dataset combining the train and dev sets
    traindev_tfidf = unionAll(train_tfidf, dev_tfidf)

    # TODO: Evalute the best model on the test set
    # Build a new model from the concatenation of the train and dev sets in order to better utilize the data
    rf_classifier_default_100 = RandomForestClassifier(labelCol = 'label', featuresCol = 'TFIDF', numTrees=100)
    rf_pipeline_default_100 = Pipeline(stages=[label_indexer, rf_classifier_default_100])
    rf_model_default_100 = rf_pipeline_default_100.fit(traindev_tfidf)
    rf_predictions_default_test_100 = rf_model_default_100.transform(test_tfidf)
    auc_rf_default_test_100 = evaluator.evaluate(rf_predictions_default_test_100, {evaluator.metricName: 'areaUnderROC'})
    print('Random Forest, Default Parameters, Test Set, AUC:' + str(auc_rf_default_test_100))