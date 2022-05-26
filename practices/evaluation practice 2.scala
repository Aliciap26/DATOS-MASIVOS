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




