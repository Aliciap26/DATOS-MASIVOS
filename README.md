 unit_2
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

## Practice #2
[PDF link]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Practice%20%232.pdf}
[scala File]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/PRACTICA%202.Pipeline}
## Concept
Decision trees and their ensembles are popular methods for machine learning regression and classification tasks. Decision trees are widely used because they are easy to interpret, handle categorical features, extend to multiclass classification settings, do not require feature scaling, and can capture nonlinearities and feature interactions. Ensemble-of-trees algorithms, such as random forests and boosting, are among the best for classification and regression tasks.

A decision tree has a structure similar to a flowchart where an internal node represents a feature or attribute, the branch represents a decision rule, and each node or leaf represents the result. The top node of a decision tree is known as the root node.

## Input code 
~~~
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
~~~
## Ouput.
~~~

Predict: 0.87
~~~

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
## Practice #4
[PDF link]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Practice4.docx}
[Scala File]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Practica4.Gradi}

## Concept of 
Support vector machine (SVM) es un algoritmo de aprendizaje supervisado que se utiliza en muchos problemas de clasificación y regresión, incluidas aplicaciones médicas de procesamiento de señales, procesamiento del lenguaje natural y reconocimiento de imágenes y voz.

## Code
~~~
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Train a GradientBoostedTrees model.
// The defaultParams for Classification use LogLoss by default.
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point => val prediction = model.predict(point.features) (point.label, prediction)}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println(s"Test Error = $testErr")
println(s"Learned classification GBT model:\n ${model.toDebugString}")

// Save and load model
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
val sameModel = GradientBoostedTreesModel.load(sc, "target/tmp/myGradientBoostingClassificationModel")
~~~

## Results
~~~
Learned classification GBT model:
 TreeEnsembleModel classifier with 3 trees

  Tree 0:
    If (feature 434 <= 70.5)
     If (feature 99 <= 83.0)
      Predict: -1.0
     Else (feature 99 > 83.0)
      Predict: 1.0
    Else (feature 434 > 70.5)
     Predict: 1.0
  Tree 1:
    If (feature 490 <= 43.0)
     If (feature 99 <= 83.0)
      If (feature 238 <= 235.5)
       If (feature 126 <= 35.0)
        Predict: -0.4768116880884702
       Else (feature 126 > 35.0)
        Predict: -0.4768116880884703
      Else (feature 238 > 235.5)
       Predict: -0.47681168808847035
     Else (feature 99 > 83.0)
      Predict: 0.4768116880884694
    Else (feature 490 > 43.0)
     If (feature 348 <= 33.0)
      Predict: 0.47681168808847024
     Else (feature 348 > 33.0)
      Predict: 0.4768116880884712
  Tree 2:
    If (feature 434 <= 70.5)
     If (feature 99 <= 83.0)
      Predict: -0.4381935810427206
     Else (feature 99 > 83.0)
      Predict: 0.43819358104271977
    Else (feature 434 > 70.5)
     If (feature 291 <= 5.5)
      Predict: 0.4381935810427206
     Else (feature 291 > 5.5)
      Predict: 0.43819358104272155
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
## Practice #6
[PDF link]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Pr%C3%A1ctica%206%20.pdf}
[Scala file]{https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Pr%C3%A1ctica%206%20.pdf}
## Concept of 

## Code
~~~
// Linear Support Vector Machine

// Import the "LinearSVC" library.
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training  = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

//Set the maximum number of iterations and the regularization parameter 
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)


// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
~~~
## Results
~~~
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.170630317473439E-4,-1.172288654973735E-4,-8.882754836918948E-5,8.522360710187464E-5,0.0,0.0,-1.3436361263314267E-5,3.729569801338091E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.888949552633658E-4,2.9864059761812683E-4,3.793378816193159E-4,-1.762328898254081E-4,0.0,1.5028489269747836E-6,1.8056041144946687E-6,1.8028763260398597E-6,-3.3843713506473646E-6,-4.041580184807502E-6,2.0965017727015125E-6,8.536111642989494E-5,2.2064177429604464E-4,2.1677599940575452E-4,-5.472401396558763E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.21415502407147E-4,3.1351066886882195E-4,2.481984318412822E-4,0.0,-4.147738197636148E-5,-3.6832150384497175E-5,0.0,-3.9652366184583814E-6,-5.1569169804965594E-5,-6.624697287084958E-5,-2.182148650424713E-5,1.163442969067449E-5,-1.1535211416971104E-6,3.8138960488857075E-5,1.5823711634321492E-6,-4.784013432336632E-5,-9.386493224111833E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.3174897827077767E-4,1.7055492867397665E-4,0.0,-2.7978204136148868E-5,-5.88745220385208E-5,-4.1858794529775E-5,-3.740692964881002E-5,-3.9787939304887E-5,-5.545881895011037E-5,-4.505015598421474E-5,-3.214002494749943E-6,-1.6561868808274739E-6,-4.416063987619447E-6,-7.9986183315327E-6,-4.729962112535003E-5,-2.516595625914463E-5,-3.6407809279248066E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.4719098130614967E-4,0.0,-3.270637431382939E-5,-5.5703407875748054E-5,-5.2336892125702286E-5,-7.829604482365818E-5,-7.60385448387619E-5,-8.371051301348216E-5,-1.8669558753795108E-5,0.0,1.2045309486213725E-5,-2.3374084977016397E-5,-1.0788641688879534E-5,-5.5731194431606874E-5,-7.952979033591137E-5,-1.4529196775456057E-5,8.737948348132623E-6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0012589360772978808,-1.816228630214369E-4,-1.0650711664557365E-4,-6.040355527710781E-5,-4.856392973921569E-5,-8.973895954652451E-5,-8.78131677062384E-5,-5.68487774673792E-5,-3.780926734276347E-5,1.3834897036553787E-5,7.585485129441565E-5,5.5017411816753975E-5,-1.5430755398169695E-5,-1.834928703625931E-5,-1.0354008265646844E-4,-1.3527847721351194E-4,-1.1245007647684532E-4,-2.9373916056750564E-5,-7.311217847336934E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.858228613863785E-4,-1.2998173971449976E-4,-1.478408021316135E-4,-8.203374605865772E-5,-6.556685320008032E-5,-5.6392660386580244E-5,-6.995571627330911E-5,-4.664348159856693E-5,-2.3026593698824318E-5,7.398833979172035E-5,1.4817176130099997E-4,1.0938317435545486E-4,7.940425167011364E-5,-6.743294804348106E-7,-1.2623302721464762E-4,-1.9110387355357616E-4,-1.8611622108961136E-4,-1.2776766254736952E-4,-8.935302806524433E-5,-1.239417230441996E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.829530831354112E-4,-1.3912189600461263E-4,-1.2593136464577562E-4,-5.964745187930992E-5,-5.360328152341982E-5,-1.0517880662090183E-4,-1.3856124131005022E-4,-7.181032974125911E-5,2.3249038865093483E-6,1.566964269571967E-4,2.3261206954040812E-4,1.7261638232256968E-4,1.3857530960270466E-4,-1.396299028868332E-5,-1.5765773982418597E-4,-2.0728798812007546E-4,-1.9106441272002828E-4,-1.2744834161431415E-4,-1.2755611630280015E-4,-5.1885591560478935E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.59081567023441E-4,-1.216531230287931E-4,-5.623851079809818E-5,-3.877987126382982E-5,-7.550900509956966E-5,-1.0703140005463545E-4,-1.4720428138106226E-4,-8.781423374509368E-5,7.941655609421792E-5,2.3206354986219992E-4,2.7506982343672394E-4,2.546722233188043E-4,1.810821666388498E-4,-1.3069916689929984E-5,-1.842374220886751E-4,-1.977540482445517E-4,-1.7722074063670741E-4,-1.487987014723575E-4,-1.1879021431288621E-4,-9.755283887790393E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.302740311359312E-4,-5.3683030235535024E-5,-1.7631200013656873E-5,-7.846611034608254E-5,-1.22100767283256E-4,-1.7281968533449702E-4,-1.5592346128894157E-4,-5.239579492910452E-5,1.680719343542442E-4,2.8930086786548053E-4,3.629921493231646E-4,2.958223512266975E-4,2.1770466955449064E-4,-6.40884808188951E-5,-1.9058225556007997E-4,-2.0425138564600712E-4,-1.711994903702119E-4,-1.3853486798341369E-4,-1.3018592950855062E-4,-1.1887779512760102E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-7.021411112285498E-5,-1.694500843168125E-5,-7.189722824172193E-5,-1.4560828004346436E-4,-1.4935497340563198E-4,-1.9496419340776972E-4,-1.7383743417254187E-4,-3.3438825792010694E-5,2.866538327947017E-4,2.9812321570739803E-4,3.77250607691119E-4,3.211702827486386E-4,2.577995115175486E-4,-1.6627385656703205E-4,-1.8037105851523224E-4,-2.0419356344211325E-4,-1.7962237203420184E-4,-1.3726488083579862E-4,-1.3461014473741762E-4,-1.2264216469164138E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0015239752514658556,-5.472330865993813E-5,-9.65684394936216E-5,-1.3424729853486994E-4,-1.4727467799568E-4,-1.616270978824712E-4,-1.8458259010029364E-4,-1.9699647135089726E-4,1.3085261294290817E-4,2.943178857107149E-4,3.097773692834126E-4,4.112834769312103E-4,3.4113620757035025E-4,1.6529945924367265E-4,-2.1065410862650534E-4,-1.883924081539624E-4,-1.979586414569358E-4,-1.762131187223702E-4,-1.272343622678854E-4,-1.2708161719220297E-4,-1.4812221011889967E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.001140680600536578,-1.323467421269896E-4,-1.2904607854274846E-4,-1.4104748544921958E-4,-1.5194605434027872E-4,-2.1104539389774283E-4,-1.7911827582001795E-4,-1.8952948277194435E-4,2.1767571552539842E-4,3.0201791656326465E-4,4.002863274397723E-4,4.0322806756364006E-4,4.118077382608461E-4,3.7917405252859545E-6,-1.9886290660234838E-4,-1.9547443112937263E-4,-1.9857348218680872E-4,-1.3336892200703206E-4,-1.2830129292910815E-4,-1.1855916317355505E-4,-1.765597203760205E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0010938769592297973,-1.2785475305234688E-4,-1.3424699777466666E-4,-1.505200652479287E-4,-1.9333287822872713E-4,-2.0385160086594937E-4,-1.7422470698847553E-4,4.63598443910652E-5,2.0617623087127652E-4,2.862882891134514E-4,4.074830988361515E-4,3.726357785147985E-4,3.507520190729629E-4,-1.516485494364312E-4,-1.7053751921469217E-4,-1.9638964654350848E-4,-1.9962586265806435E-4,-1.3612312664311173E-4,-1.218285533892454E-4,-1.1166712081624676E-4,-1.377283888177579E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.044386260118809E-4,-1.240836643202059E-4,-1.335317492716633E-4,-1.5783442604618277E-4,-1.9168434243384107E-4,-1.8710322733892716E-4,-1.1283989231463139E-4,1.1136504453105364E-4,1.8707244892705632E-4,2.8654279528966305E-4,4.0032117544983536E-4,3.169637536305377E-4,2.0158994278679014E-4,-1.3139392844616033E-4,-1.5181070482383948E-4,-1.825431845981843E-4,-1.602539928567571E-4,-1.3230404795396355E-4,-1.1669138691257469E-4,-1.0532154964150405E-4,-1.3709037042366007E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-4.0287410145021705E-4,-1.3563987950912995E-4,-1.3225887084018914E-4,-1.6523502389794188E-4,-2.0175074284706945E-4,-1.572459106394481E-4,2.577536501278673E-6,1.312463663419457E-4,2.0707422291927531E-4,3.9081065544314936E-4,3.3487058329898135E-4,2.5790441367156086E-4,2.6881819648016494E-5,-1.511383586714907E-4,-1.605428139328567E-4,-1.7267287462873575E-4,-1.1938943768052963E-4,-1.0505245038633314E-4,-1.1109385509034013E-4,-1.3469914274864725E-4,-2.0735223736035555E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.034374233912422E-4,-1.5961213688405883E-4,-1.274222123810994E-4,-1.582821104884909E-4,-2.1301220616286252E-4,-1.2933366375029613E-4,1.6802673102179614E-5,1.1020918082727098E-4,2.1160795272688753E-4,3.4873421050827716E-4,2.6487211944380384E-4,1.151606835026639E-4,-5.4682731396851946E-5,-1.3632001630934325E-4,-1.4340405857651405E-4,-1.248695773821634E-4,-8.462873247977974E-5,-9.580708414770257E-5,-1.0749166605399431E-4,-1.4618038459197777E-4,-3.7556446296204636E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.124342611878493E-4,-2.0369734099093433E-4,-1.3626985098328694E-4,-1.3313768183302705E-4,-1.871555537819396E-4,-1.188817315789655E-4,-1.8774817595622694E-5,5.7108412194993384E-5,1.2728161056121406E-4,1.9021458214915667E-4,1.2177397895874969E-4,-1.2461153574281128E-5,-7.553961810487739E-5,-1.0242174559410404E-4,-4.44873554195981E-5,-9.058561577961895E-5,-6.837347198855518E-5,-8.084409304255458E-5,-1.3316868299585082E-4,-2.0335916397646626E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.966510928472775E-4,-1.3738983629066386E-4,-3.7971221409699866E-5,-6.431763035574533E-5,-1.1857739882295322E-4,-9.359520863114822E-5,-5.0878371516215046E-5,-8.269367595092908E-8,0.0,1.3434539131099211E-5,-1.9601690213728576E-6,-2.8527045990494954E-5,-7.410332699310603E-5,-7.132130570080122E-5,-4.9780961185536E-5,-6.641505361384578E-5,-6.962005514093816E-5,-7.752898158331023E-5,-1.7393609499225025E-4,-0.0012529479255443958,0.0,0.0,2.0682521269893754E-4,0.0,0.0,0.0,0.0,0.0,-4.6702467383631055E-4,-1.0318036388792008E-4,1.2004408785841247E-5,0.0,-2.5158639357650687E-5,-1.2095240910793449E-5,-5.19052816902203E-6,-4.916790639558058E-6,-8.48395853563783E-6,-9.362757097074547E-6,-2.0959335712838412E-5,-4.7790091043859085E-5,-7.92797600958695E-5,-4.462687041778011E-5,-4.182992428577707E-5,-3.7547996285851254E-5,-4.52754480225615E-5,-1.8553562561513456E-5,-2.4763037962085644E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.4886180455242474E-4,-5.687523659359091E-6,7.380040279654313E-5,4.395860636703821E-5,7.145198242379862E-5,6.181248343370637E-6,0.0,-6.0855538083486296E-5,-4.8563908323274725E-5,-4.117920588930435E-5,-4.359283623112936E-5,-6.608754161500044E-5,-5.443032251266018E-5,-2.7782637880987207E-5,0.0,0.0,2.879461393464088E-4,-0.0028955529777851255,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.2312114837837392E-4,-1.9526747917254753E-5,-1.6999506829961688E-5,5.4835294148085086E-5,1.523441632762399E-5,-5.8365604525328614E-5,-1.2378194216521848E-4,-1.1750704953254656E-4,-6.19711523061306E-5,-5.042009645812091E-5,-1.4055260223565886E-4,-1.410330942465528E-4,-1.9272308238929396E-4,-4.802489964676616E-4] Intercept: 0.012911305214513969
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

[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/Practice%20unit%202.pdf)  
[Scala File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/evaluation%20practice%202.scala)  

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
~~~
scala>println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")  
Test set accuracy = 0.95  
~~~
##  Conclusions.  

## Edgar Munguia:  
In this unit, we learned more about machine learning and some of the algorithms using this technology. Machine learning is the main topic in technology in the present years because it represents the future of technology and data. We, as future data scientists, we will use this kind of algorithms to make our live easier when working with this kind of data, so, in conclusion, we can say, that machine learning its a powerful tool for the treatment of data, and of course, it will bring a lot of benefits for us, as data scientists.  

## Alicia Pérez:  
This evaluative practice, in my opinion, was a little more complex since it contained the topics that were presented in class, and the practices based on them, if it was a small challenge and the truth is that it is a bit difficult for me to transmit what I recently learned, but achieved.  

## Link of the video (Youtube):  
https://www.youtube.com/watch?v=esK8jToP6E0  

## Github link:  
https://github.com/Aliciap26/DATOS-MASIVOS/tree/unit_2  




# Big data
#branch development added
 development
