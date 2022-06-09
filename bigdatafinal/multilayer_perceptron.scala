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