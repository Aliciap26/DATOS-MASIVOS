//1. Import a simple spark session.

import org.apache.spark.sql.SparkSession

//2. Use code lines to reduce code errors

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3. Create a spark session instance

val spark = SparkSession.builder().getOrCreate()

//4. Import Kmeans library to use the cluster algorithm

import org.apache.spark.ml.clustering.KMeans

// 5. Load  the dataset( Wholesale Customers Data.csv)

val data = spark.read.option("header", "true").option("inferSchema","true")csv("Wholesale_customers_data.csv")

// 6. Make another data set selecting the follow columns and call the dataframe "feature_data"(Fresh, Milk, Grocery, Frozen, Detergents_Paper,
Delicassen)

val feature_data = data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")

// 7 Show the data of the new dataset
feature_data.show()


// 8 Importarting Vector Assembler y Vector libraries
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// 9 Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
val assembler=(new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features"))

// 10 Use the object assembler to transform feature_data
val transform =assembler.transform(feature_data)

// 11 Show the transformed results
transform.show()

// 12 Create kmeans model with k =3

val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(transform)

//13 Evaluating the groups using within set sum and print the centroids
val WSSSE = model.computeCost(transform)
println(s"Within Set Sum of Squared Errors = $WSSSE")

//14 Printing the centroids 
println("Cluster Centers: ")
model.clusterCenters.foreach(println)


