# **Unit # 2 branch. Big Data course.**  

## Practice#1  
[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practice%231%2C%20unit%232.pdf)  
[Scala File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/correlation.scala)  


## Input  
~~~
package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
object CorrelationsExample {

def main(){

    val conf = new SparkConf().setAppName("CorrelationsExample")
    val sc = new SparkContext(conf)

    // $example on$
    val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))  // a series
    // must have the same number of partitions and cardinality as seriesX
    val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))

    // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
    // method is not specified, Pearson's method will be used by default.
    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(s"Correlation is: $correlation")

    val data: RDD[Vector] = sc.parallelize(
    Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0))
    )  // note that each Vector is a row and not a column

    // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method
    // If a method is not specified, Pearson's method will be used by default.
    val correlMatrix: Matrix = Statistics.corr(data, "pearson")
    println(correlMatrix.toString)
    // $example off$

    sc.stop()
    }
}
~~~
## Output  
![logo](/images/corre.png)  


## Practice #3  

[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practice%203%20u2.pdf)  
[Scala File(Clasification)](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/classification.scala)  
[Scala File(Regression)](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/regressiont.scala)  


## Random Forest concept.  
Random Forest, is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both classification and regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and improve model performance. As the name suggests, "Random Forest is a classifier that contains a series of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on a decision tree, the random forest takes the prediction of each tree and based on the majority votes of the predictions, predicts the final result.

## Input code (Clasification).  
~~~ 
import org.apache.spark.mllib.tree.RandomForest  
import org.apache.spark.mllib.tree.model.RandomForestModel  
import org.apache.spark.mllib.util.MLUtils  

// Load and parse the data file.  
val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")  
// Split the data into training and test sets (30% held out for testing)  
val splits = data.randomSplit(Array(0.7, 0.3))  
val (trainingData, testData) = (splits(0), splits(1))  

// Train a RandomForest model.  
// Empty categoricalFeaturesInfo indicates all features are continuous.  
val numClasses = 2  
val categoricalFeaturesInfo = Map[Int, Int]()  
val numTrees = 3 // Use more in practice.  
val featureSubsetStrategy = "auto" // Let the algorithm choose.  
val impurity = "gini"  
val maxDepth = 4  
val maxBins = 32  

val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)  

// Evaluate model on test instances and compute test error  
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}  
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()  
println(s"Test Error = $testErr")
println(s"Learned classification forest model:\n ${model.toDebugString}")  

// Save and load model
model.save(sc, "target/tmp/myRandomForestClassificationModel")
val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")  
~~~

## Output.  
~~~
Tree 0:

If (feature 245 <= 16.0)

If (feature 315 <= 64.0)

If (feature 467 <= 70.0)

Predict: 1.0

Else (feature 467 > 70.0)

Predict: 0.0

Else (feature 315 > 64.0)

Predict: 0.0

Else (feature 245 > 16.0)

Predict: 0.0

Tree 1:

If (feature 512 <= 1.5)

If (feature 688 <= 14.0)

Predict: 1.0

Else (feature 688 > 14.0)

Predict: 0.0

Else (feature 512 > 1.5)

Predict: 0.0

Tree 2:

If (feature 356 <= 19.0)If (feature 301 <= 27.0)

If (feature 351 <= 2.0)

Predict: 0.0

Else (feature 351 > 2.0)

Predict: 1.0

Else (feature 301 > 27.0)

Predict: 0.0

Else (feature 356 > 19.0)

Predict: 0.0

scala> 

~~~
## Input code (Regression).  
~~~
import org.apache.spark.mllib.tree.RandomForest  
import org.apache.spark.mllib.tree.model.RandomForestModel  
import org.apache.spark.mllib.util.MLUtils  

// Load and parse the data file.  
val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")  
// Split the data into training and test sets (30% held out for testing)  
val splits = data.randomSplit(Array(0.7, 0.3))  
val (trainingData, testData) = (splits(0), splits(1))  

// Train a RandomForest model.  
// Empty categoricalFeaturesInfo indicates all features are continuous.  
val numClasses = 2  
val categoricalFeaturesInfo = Map[Int, Int]()  
val numTrees = 3 // Use more in practice.  
val featureSubsetStrategy = "auto" // Let the algorithm choose.  
val impurity = "variance"  
val maxDepth = 4  
val maxBins = 32  

val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)  

// Evaluate model on test instances and compute test error  
val labelsAndPredictions = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}  
val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()  
println(s"Test Mean Squared Error = $testMSE")  
println(s"Learned regression forest model:\n ${model.toDebugString}")  

// Save and load model  
model.save(sc, "target/tmp/myRandomForestRegressionModel")  
val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")  

~~~
## Output.  
~~~
Tree 0:

If (feature 406 <= 126.5)

Predict: 0.0

Else (feature 406 > 126.5) Predict: 1.0

Tree 1:

If (feature 406 <= 9.5)

Predict: 0.0

Else (feature 406 > 9.5) Predict: 1.0

Tree 2:

If (feature 406 <= 126.5)
    
    Predict: 0.0
    
Else (feature 406 > 126.5) 
     Predict: 1.0

scala> 
~~~

## Practice #5  

[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practica%235.pdf)  
[Scala File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practice_5.scala)  

## Concept of Multilayer perceptron classifier.  
The Multilayer Perceptron Classifier (MLPC) is a classifier based on the feed-forward artificial neural network. The MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer of the network. The input layer nodes represent the input data.
All other nodes map inputs to outputs by linearly combining the inputs with the node weights w and bias b and applying an activation function.


## Examples of application.  
The multilayer perceptron (hereinafter MLP, MultiLayer Perceptron) is used to solve problems of pattern association, image segmentation, data compression, etc.


## Code.  
~~~
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame. || Carga los datos almacenados en formato LIBSVM como DataFrame.

//val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
val data = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt")

// Split the data into train and test || Divide los datos
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network: || especificar capas para la red neuronal:
// input layer of size 4 (features), two intermediate of size 5 and 4 || capa de entrada de tamano 4 (features), dos intermedias de tamano 5 y 4
// and output of size 3 (classes) || y salida de tamano 3 (classes) 
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters || Crea el trainer y establece sus parametros.
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// train the model || entrena el model
val model = trainer.fit(train)

// compute accuracy on the test set || precision de calculo en el conjunto de prueba
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
~~~
## Results.  

~~~
scala> println(s"Test set accuracy= ${evaluator.evaluate(predictionAndLabels)}")
Test set accuracy=0.9019607843137255
~~~
## Practice #7.  

[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practice7.pdf)  
[Scala File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/nb.scala)  

## Concept of Naive Bayes.  
Naive Bayes is one of the simplest and most powerful algorithms for classification based on Bayes' Theorem with an assumption of independence between predictors. Naive Bayes is easy to build and particularly useful for very large data sets.

The Naive Bayes classifier assumes that the effect of a particular feature on a class is independent of other features. For example, a loan applicant is desirable or undesirable depending on their income, prior loan and transaction history, age, and location. Even if these features are interdependent, these features are considered independently. This assumption simplifies computation, and is therefore considered naive. This assumption is called conditional class independence.

## Strengths.  
-A quick and easy way to predict classes, for binary and multiclass classification problems.  
-The algorithm performs better than other classification models, even with less training data.  
-The decoupling of the class conditional feature distributions means that each distribution can be estimated independently as if it had only one dimension.  

## weaknesses.  
-Naive Bayes algorithms are known to be poor estimators. Therefore, the odds that are obtained should not be taken very seriously.  
-The Naive assumption of independence will most likely not reflect what the data is like in the real world.  
-When the test data set has a feature that has not been observed in the training set, the model will assign it a probability of zero and predictions will be useless.  

## Code.  
~~~
//Importar las librerias necesarias

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Cargar los datos especificando la ruta del archivo

val data = spark.read.format("libsvm").load("C:/spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())

//Mostrar las primeras 20 líneas por defecto

data.show()

//Divida aleatoriamente el conjunto de datos en conjunto de entrenamiento y conjunto de prueba de acuerdo con los pesos proporcionados. También puede especificar una seed

val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)
// El resultado es el tipo de la matriz, y la matriz almacena los datos de tipo DataSet

//Incorporar al conjunto de entrenamiento (operación de ajuste) para entrenar un modelo bayesiano
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//El modelo llama a transform() para hacer predicciones y generar un nuevo DataFrame.

val predictions = naiveBayesModel.transform(testData)

//Salida de datos de resultados de predicción
predictions.show()

//Evaluación de la precisión del modelo

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// Precisión
val precision = evaluator.evaluate (predictions) 

//Imprimir la tasa de error
println ("tasa de error =" + (1-precision))
~~~



##  Output.  

![bayes](/images/bayes.PNG)  


## Evaluation practice. 

[PDF link]()  
[Scala File]()  

## Introduction.  
In this practice, we will use an algorithm of machine learning called multilayer perceptron, we will use its libraries to succesfully make the evaluation practice from unit 2 of big data course.

## Code.  
~~~
##Import neccesary libraries

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors

## Load dataframe (provided by the teacher)

val csvfile=spark.read.format("csv").option("header","true").option("inferSchema", "true").load("iris.csv")

## Clean data

val Clean =csvfile.na.drop()

##Show columns name

Clean.columns

## show schema
Clean.printSchema

##print the first 5 columns

Clean.show(5)

##Use the describe() method to learn more about the data in the DataFrame.

Clean.describe().show 

##Make the pertinent transformation for the categorical data which will be
our labels to classify.
val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(Clean)
sustituir la columna "species" con nuestra columna "indexedLabel" y la vamos a mostrar con el nombre de "label"

val indexed = labelIndexer.transform(Clean).drop("species").withColumnRenamed("indexedLabel", "label") 
indexed.describe().show() 
val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
val  features = assembler.transform(indexed)
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(indexed)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

features.show
##Build the classification model and explain its architecture

vamos a separar el dataset en 30% en datos de prueba y un 70% en datos de entrenamiento establecemos la semilla de aleatoriedad
val splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L) 
val train = splits(0) 
val test = splits(1)
val layers = Array[Int](4, 5, 4, 3)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
val model = trainer.fit(train)
val result = model.transform(test)


##Print the model results
val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels.show

val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

~~~

## Outputs.  
scala>println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")  
Test set accuracy = 0.95  

##  Conclusions.  

## Edgar Munguia:  
In this unit, we learned more about machine learning and some of the algorithms using this technology. Machine learning is the main topic in technology in the present years because it represents the future of technology and data. We, as future data scientists, we will use this kind of algorithms to make our live easier when working with this kind of data, so, in conclusion, we can say, that machine learning its a powerful tool for the treatment of data, and of course, it will bring a lot of benefits for us, as data scientists.  

## Alicia Pérez:  
This evaluative practice, in my opinion, was a little more complex since it contained the topics that were presented in class, and the practices based on them, if it was a small challenge and the truth is that it is a bit difficult for me to transmit what I recently learned, but achieved.  

## Link of the video (Youtube):  
https://www.youtube.com/watch?v=esK8jToP6E0  

## Github link:  
https://github.com/Aliciap26/DATOS-MASIVOS/tree/unit_2  



