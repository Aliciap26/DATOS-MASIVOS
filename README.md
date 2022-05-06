# **Unit # 2 branch. Big Data course.**  

## Practice#1  
[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/practice%231%2C%20unit%232.pdf)  
[Scala File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/correlation.scala)  


## Input  
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

## Output  
![logo](/images/corre.png)  


## Practice #3  

[PDF link]()  
[Scala File(Clasification)](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/classification.scala)  
[Scala File(Regression)](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_2/practices/regressiont.scala)  


## Random Forest concept.  
Random Forest, is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both classification and regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and improve model performance. As the name suggests, "Random Forest is a classifier that contains a series of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on a decision tree, the random forest takes the prediction of each tree and based on the majority votes of the predictions, predicts the final result.

## Input code (Clasification).  

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


## Output.  

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


## Input code (Regression).  

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


## Output.  

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










