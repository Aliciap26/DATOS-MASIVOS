

# Final project.  

[PDF link]()  
[SVM scala file]()  
[Decision tree scala file]()  
[Logistic regression scala file]()  
[Multilayer perceptron scala file]()  

## Introduction.
This project consists of 4 algorithms which we are going to compare to determine which is the most efficient and convenient to use in various situations.
We finished the subject of Massive Data which consists of 4 units, which were very interesting and gave us a lot of knowledge regarding the topic of big data, we learned a lot and what we highlight the most are the topics of machine learning, since as we know , ML is one of the topics that sounds the most in our society, so it is necessary to be at the forefront to implement these technologies and not be left behind technologically speaking, we, as future data scientists, must know better than anyone, this type of algorithms, next, we are going to present said algorithms, we will define them, we will show them, we will compare them to determine, which one is the best, in terms of speed and efficiency.
# Conceptual framework.

## Support Vector Machine (SVM): 
It is a supervised learning algorithm used in many classification and regression problems, including medical applications of signal processing, natural language processing, and image and speech recognition.

The goal of the SVM algorithm is to find a hyperplane that best separates two different classes of data points. "As best as possible" implies the hyperplane with the widest margin between the two classes, represented by the plus and minus signs in the figure below. The margin is defined as the maximum width of the region parallel to the hyperplane that has no interior data points. The algorithm can only find this hyperplane in problems that allow linear separation; in most practical problems, the algorithm maximizes the soft margin by allowing a small number of misclassifications.


Reference: 
- MATLAB. (2015, 2 marzo). Support Vector Machine (SVM). MATLAB & Simulink. Recuperado 5 de junio de 2022, de https://es.mathworks.com/discovery/support-vector-machine.html


## Decision Tree:
Decision trees are statistical algorithms or machine learning techniques that allow us to build predictive data analytics models for Big Data based on their classification according to certain characteristics or properties, or on regression through the relationship between different variables to predict the value of another.

In classification models we want to predict the value of a variable by classifying the information based on other variables (type, group membership...). For example, we want to predict which people will buy a certain product, by classifying between customers and non-customers, or which brands of laptops each person will buy by classifying between different brands. The values ​​to be predicted are predefined, that is, the results are defined in a set of possible values.

In regression models, an attempt is made to predict the value of a variable based on other variables that are independent of each other. For example, we want to predict the sale price of the land based on variables such as its location, surface, distance to the beach, etc. The possible result is not part of a predefined set, but can take any possible value.

Reference: 
- Unir, V. (2021, 19 octubre). Árboles de decisión: en qué consisten y aplicación en Big Data. UNIR. Recuperado 5 de junio de 2022, de https://www.unir.net/ingenieria/revista/arboles-de-decision/


## Logistic Regression: 
Logistic Regression is a statistical method to predict binary classes. The outcome or target variable is dichotomous in nature. Dichotomous means that there are only two possible classes. For example, it can be used for cancer screening problems or calculating the probability of an event occurring.

Logistic Regression is one of the simplest and most widely used Machine Learning algorithms for two-class classification. It is easy to implement and can be used as a baseline for any binary classification problem. Logistic Regression describes and estimates the relationship between a binary dependent variable and the independent variables.

Reference: 
- Gonzalez, L. (2020, 21 agosto). Regresión Logística - Teoría. 🤖 Aprende IA. Recuperado 5 de junio de 2022, de https://aprendeia.com/regresion-logistica-multiple-machine-learning-teoria/#:%7E:text=La%20Regresi%C3%B3n%20Log%C3%ADstica%20es%20uno,cualquier%20problema%20de%20clasificaci%C3%B3n%20binaria.

## Multilayer perceptron:
The multilayer perceptron (MLP) is a complement to the feedforward neural network. It consists of three types of layers: the input layer, the output layer, and the hidden layer. The input layer receives the input signal to be processed. The output layer performs the required task, such as prediction and classification. An arbitrary number of hidden layers that are placed between the input and output layers are the real computational engine of the MLP. Similar to a forward network in an MLP, data flows in the forward direction from the input layer to the output layer. The neurons in the MLP are trained with the backpropagation learning algorithm. MLPs are designed to approximate any continuous function and can solve problems that are not linearly separable. The main use cases of MLP are pattern classification, recognition, prediction, and approximation.

Reference: 
- Sciencedirect. (2014a, abril 1). Multilayer Perceptron. Recuperado 5 de junio de 2022, de https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron


## Implementation.
To carry out the implementation of the aforementioned algorithms, we used the spark/scala programming language, since it is a very powerful tool for Big Data topics and, furthermore, it is relatively simple to use. actually, we compare it more or less with python, since it is not such a complex language compared to others. The possibilities it offers us are infinite, being for us, one of the top tools to work with massive data.

# Code
## SVM
~~~
//First we must tell spark to start counting the time since we run it, then we must import the libraries, start a simple session in spark
val start = System.currentTimeMillis
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)



val data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("C:/Users/x/Documents/projet/bank-full.csv")

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

//Transform the categorical data to numeric
val labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
val indexed = labelIndexer.transform(data).withColumnRenamed("indexedLabel", "label") 

//In order to avoid error we need to create 2 new columns label and features
//Here we create them using StringIndexer and VectorAssembler
val assembler = new VectorAssembler().setInputCols(Array("age", "balance", "day", "duration", "previous")).setOutputCol("features")
val features = assembler.transform(indexed)

val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)

val lsvcModel = lsvc.fit(training)

val results = lsvcModel.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy

val error = 1 - metrics.accuracy

println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

//Get the total time of the program execution
val totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

//Get the total of MB used 
val runtime = Runtime.getRuntime
val mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemory) / mb + " MB")
~~~  
![svm](/img/svm1.PNG)  
![svm](/img/svm1.PNG)  
![svm](/img/svm1.PNG)  

## Decision Tree
~~~
//First we must tell spark to start counting the time since we run it, then we must import the libraries, start a simple session in spark
val start = System.currentTimeMillis

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema","true").option("delimiter",";")csv("C:/Users/x/Documents/projet/bank-full.csv")


//Transform the categorical data to numeric, merges the new data with the previous values
//this time with the numeric data and renames the loan column as label to use it in the execution
val labelIndexer = new StringIndexer().setInputCol("loan").setOutputCol("indexedLabel").fit(data)
val indexed = labelIndexer.transform(data).drop("loan").withColumnRenamed("indexedLabel", "label") 
val assembler = (new VectorAssembler().setInputCols(Array("age", "balance", "day", "duration", "previous")).setOutputCol("features"))
val features = assembler.transform(indexed)
val filter = features.withColumnRenamed("loan", "label")

val finalData = filter.select("label", "features")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels into the index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(finalData)
// Automatically identify categorical features and then index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(finalData)
// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = finalData.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train the model, this also runs the indexers.
val model = pipeline.fit(trainingData)

// Make the predictions.
val predictions = model.transform(testData)

// Select example rows to display. In this case there was only 5 rows to show.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label)
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
// Compute the test error.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// Show by stages the classification of the tree model
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

//Get the total time of the program execution
val totalTime = System.currentTimeMillis - start
println("Elapsed time: %1d ms".format(totalTime))

//Get the total of MB used 
val runtime = Runtime.getRuntime
val mb = 1024*1024
println("Used memory: " + (runtime.totalMemory - runtime.freeMemory) / mb + " MB")
~~~  
![svm](/img/dtc1.PNG)  
![svm](/img/dtc2.PNG)  
![svm](/img/dtc3.PNG)  

## Logistic Regression
~~~
// Starting timer
val timerstar = System.currentTimeMillis()

//Import necessary libraries

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Create spark session
val spark = SparkSession.builder().getOrCreate()

// Load dataframe
val data = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// Adding index labels
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)
val indexed = labelIndexer.transform(data).drop("y").withColumnRenamed("indexedLabel", "label")

// Creating a vector to add the info into the array
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))

// Transforming the previous vector into a new index variable
val features = vectorFeatures.transform(indexed)

// Renaming label of features
val featuresLabel = features.withColumnRenamed("y", "label")

// A new variable is created  selecting some columns
val dataIndexed = featuresLabel.select("label","features")

// Use randomSplit to create 70/30 split test and train data
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3), seed = 12345)

// The Logistic Regression is created with the parameters sent
val logisticAlgorithm = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

// The model is trained
val logisticModel = logisticAlgorithm.fit(training)

// Calculating precission of the test data
val results = logisticModel.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Showing confusionMatrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Showing Accuracy and error
println("Accuracy: "+metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")
// Stoping the timer and calculated the time that took to run the algorithm
val timerstop = System.currentTimeMillis()
val duration = (timerstop - timerstar) / 1000

// Printing the time in seconds that took the algorithm to run
println(duration)


~~~  
![svm](/img/lr1.PNG)  
![svm](/img/lr2.PNG)  

 ## Multilayer perceptron.
 ~~~
 //Start timer
val starttimer = System.currentTimeMillis()

// Import necessary libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.IntegerType

// Creating spark session
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// Loading dataframe
val dataframeMP  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("bank-full.csv")

// displaying the data
dataframeMP.columns 
dataframeMP.printSchema() 
dataframeMP.head(5) 
dataframeMP.describe().show() 

// Indexing labels
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(dataframeMP)
val indexed = labelIndexer.transform(dataframeMP).drop("y").withColumnRenamed("indexedLabel", "label")
indexed.describe().show() 

// Creating assemble vector 
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val features = assembler.transform(indexed)

// The label columns are indexed and the data is displayed
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(indexed)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")
features.show()

// Data is divided into training and testing
val splits = features.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// The layers of the neural network are specified:
val layers = Array[Int](5, 4, 1, 2)

// Training parameters are set
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Training the model
val model = trainer.fit(train)

// Calculating precission
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// Printing the model accuracy
println(s"Test Accuracy = ${evaluator.evaluate(predictionAndLabels)}")
println(s"Test Error = ${(1.0 - evaluator.evaluate(predictionAndLabels))}")

// Stoping the timer
val endtimer = System.currentTimeMillis()
val duration = (endtimer - starttimer) / 1000

//Print the time that took the algorithm to run
println(duration)
 ~~~  
 
 ![svm](/img/mp.PNG)  
 ![svm](/img/mp3.PNG)  
 ![svm](/img/mp4.PNG  
 
 
# Conclusions.

We can notice how each algorithm has different predictions and what is its margin of error, which is interesting since we worked with the same data file (csv), it is interesting how a large amount of data can be handled with, this type algorithms, we learn how to use these tools and the large number of things we can do. In conclusion, according to effectiveness, the best of the algorithms is that of SVM, since in the averages it shows a smaller range of error.  

# Repository link: 
https://github.com/Aliciap26/DATOS-MASIVOS/tree/Unit-4


# Video link (Youtube):  






# References.

- MATLAB. (2015, 2 marzo). Support Vector Machine (SVM). MATLAB & Simulink. Recuperado 5 de junio de 2022, de https://es.mathworks.com/discovery/support-vector-machine.html

- Unir, V. (2021, 19 octubre). Árboles de decisión: en qué consisten y aplicación en Big Data. UNIR. Recuperado 5 de junio de 2022, de https://www.unir.net/ingenieria/revista/arboles-de-decision/

- Gonzalez, L. (2020, 21 agosto). Regresión Logística - Teoría. 🤖 Aprende IA. Recuperado 5 de junio de 2022, de https://aprendeia.com/regresion-logistica-multiple-machine-learning-teoria/#:%7E:text=La%20Regresi%C3%B3n%20Log%C3%ADstica%20es%20uno,cualquier%20problema%20de%20clasificaci%C3%B3n%20binaria.

- Sciencedirect. (2014a, abril 1). Multilayer Perceptron. Recuperado 5 de junio de 2022, de https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron






