## Evaluation practice unit 3, Big Data.  

[PDF link](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_3/evaluation%20practice/Unidad_3_datos_masivos_2022.pdf)  
[.R File](https://github.com/Aliciap26/DATOS-MASIVOS/blob/unit_3/evaluation%20practice/unit3_bigdata.scala)  

## Introduction.  
The goal of this practice is to try to clustering customers from specific regions from a wholesale distributor. 
This is based on the sales of some product categories. We will use kmeans algorithm (using it libraries) in spark to use the clustering algorithm. In summary, we used this algorithm to make small groups of data and place certain data in the group that is most similar in features, this is the main goal of kmeans.


## Code.  

~~~

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

~~~



## Output.  
## Sum of Squared Errors:  
~~~
scala> println(s"Within Set Sum of Squared Errors = $WSSSE")
Within Set Sum of Squared Errors = 8.095172370767671E10

~~~
## Centroids:  
~~~
scala> println("Cluster Centers: ")
Cluster Centers: 

scala> model.clusterCenters.foreach(println)
[7993.574780058651,4196.803519061584,5837.4926686217,2546.624633431085,2016.2873900293255,1151.4193548387098]
[9928.18918918919,21513.081081081084,30993.486486486487,2960.4324324324325,13996.594594594595,3772.3243243243246]
[35273.854838709674,5213.919354838709,5826.096774193548,6027.6612903225805,1006.9193548387096,2237.6290322580644]
~~~


## Conclusions.  



## Edgar Munguia:  
This practice was a little bit short, but the knowlege i got was big, because i learned how to use kmeans algorithm to cluster data. In this case, we worked with customer data to cluster the customers acording his features. I made this practice in data mining too(in a diferent context) so im sure i learned how to use it in all of possible cases.


## Alicia López: 
It is interesting to see how the data can be controlled with a method, in addition to this the k-means method seemed particular to me since it can classify data depending on its characteristics.
In R we could observe the graphs of the data and it is still more visual how they are accommodated.




## Video link (Youtube):  https://www.youtube.com/watch?v=xg1Qbh2A-e0


## Github repository link:  https://github.com/Aliciap26/DATOS-MASIVOS







