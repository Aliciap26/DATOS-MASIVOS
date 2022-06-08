

# Proyecto final

## Introducción
El presente proyecto consta de 4 algoritmos los cuales vamos a comparar para determinar cuál es el más eficiente y conveniente de utilizar en diversas situaciones. 
Culminamos la materia de Datos masivos la cual consta de 4 unidades, las cuales fueron muy interesantes y nos dejaron mucho conocimiento respecto al tema del big data, se aprendió bastante y lo que más destacamos, son los temas de machine learning, ya que como sabemos, ML es uno de los temas que más suenan en nuestra sociedad, entonces es necesario, estar a la vanguardia para implementar dichas tecnologías y no quedarnos atrás tecnológicamente hablando, nosotros, como futuros científicos de datos, debemos conocer mejor que nadie, este tipo de algoritmos, a continuación, vamos a presentar dichos algoritmos, los definiremos, los mostraremos, los compararemos para determinar, cuál es el mejor, en términos de rapidez y eficiencia.

# Marco teórico de los algoritmos.

## Support Vector Machine (SVM): 
Es un algoritmo de aprendizaje supervisado que se utiliza en muchos problemas de clasificación y regresión, incluidas aplicaciones médicas de procesamiento de señales, procesamiento del lenguaje natural y reconocimiento de imágenes y voz.

El objetivo del algoritmo SVM es encontrar un hiperplano que separe de la mejor forma posible dos clases diferentes de puntos de datos. “De la mejor forma posible” implica el hiperplano con el margen más amplio entre las dos clases, representado por los signos más y menos en la siguiente figura. El margen se define como la anchura máxima de la región paralela al hiperplano que no tiene puntos de datos interiores. El algoritmo solo puede encontrar este hiperplano en problemas que permiten separación lineal; en la mayoría de los problemas prácticos, el algoritmo maximiza el margen flexible permitiendo un pequeño número de clasificaciones erróneas.


Referencia: 
- MATLAB. (2015, 2 marzo). Support Vector Machine (SVM). MATLAB & Simulink. Recuperado 5 de junio de 2022, de https://es.mathworks.com/discovery/support-vector-machine.html








## Decision Tree:
Los árboles de decisión son algoritmos estadísticos o técnicas de machine learning que nos permiten la construcción de modelos predictivos de analítica de datos para el Big Data basados en su clasificación según ciertas características o propiedades, o en la regresión mediante la relación entre distintas variables para predecir el valor de otra.

En los modelos de clasificación queremos predecir el valor de una variable mediante la clasificación de la información en función de otras variables (tipo, pertenencia a un grupo…). Por ejemplo, queremos pronosticar qué personas comprarán un determinado producto, clasificando entre clientes y no clientes, o qué marcas de portátiles comprará cada persona mediante la clasificación entre las distintas marcas. Los valores a predecir son predefinidos, es decir, los resultados están definidos en un conjunto de posibles valores.

En los modelos de regresión se intenta predecir el valor de una variable en función de otras variables que son independientes entre sí. Por ejemplo, queremos predecir el precio de venta del terreno en función de variables como su localización, superficie, distancia a la playa, etc. El posible resultado no forma parte de un conjunto predefinido, sino que puede tomar cualquier posible valor.

Referencia: 
- Unir, V. (2021, 19 octubre). Árboles de decisión: en qué consisten y aplicación en Big Data. UNIR. Recuperado 5 de junio de 2022, de https://www.unir.net/ingenieria/revista/arboles-de-decision/





## Logistic Regression: 
La Regresión Logística es un método estadístico para predecir clases binarias. El resultado o variable objetivo es de naturaleza dicotómica. Dicotómica significa que solo hay dos clases posibles. Por ejemplo, se puede utilizar para problemas de detección de cáncer o calcular la probabilidad de que ocurra un evento.

La Regresión Logística es uno de los algoritmos de Machine Learning más simples y más utilizados para la clasificación de dos clases. Es fácil de implementar y se puede usar como línea de base para cualquier problema de clasificación binaria. La Regresión Logística describe y estima la relación entre una variable binaria dependiente y las variables independientes.

Referencia: 
- Gonzalez, L. (2020, 21 agosto). Regresión Logística - Teoría. 🤖 Aprende IA. Recuperado 5 de junio de 2022, de https://aprendeia.com/regresion-logistica-multiple-machine-learning-teoria/#:%7E:text=La%20Regresi%C3%B3n%20Log%C3%ADstica%20es%20uno,cualquier%20problema%20de%20clasificaci%C3%B3n%20binaria.

## Multilayer perceptron:
El perceptrón multicapa (MLP) es un complemento de la red neuronal de avance. Consta de tres tipos de capas: la capa de entrada, la capa de salida y la capa oculta. La capa de entrada recibe la señal de entrada para ser procesada. La capa de salida realiza la tarea requerida, como la predicción y la clasificación. Un número arbitrario de capas ocultas que se colocan entre la capa de entrada y la de salida son el verdadero motor computacional del MLP. De manera similar a una red de avance en un MLP, los datos fluyen en la dirección de avance desde la capa de entrada a la de salida. Las neuronas en el MLP se entrenan con el algoritmo de aprendizaje de retropropagación. Los MLP están diseñados para aproximar cualquier función continua y pueden resolver problemas que no son linealmente separables. Los principales casos de uso de MLP son la clasificación, el reconocimiento, la predicción y la aproximación de patrones.

Referencia: 
- Sciencedirect. (2014a, abril 1). Multilayer Perceptron. Recuperado 5 de junio de 2022, de https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron


## Implementación.
Para llevar a cabo la implementación de los algoritmos anteriormente mencionados, hicimos uso del lenguaje de programación spark/scala, ya que se trata de una herramienta muy poderosa para los tópicos de Big Data (Datos masivos) y además, es relativamente sencilla de utilizar, en realidad, la comparamos más o menos con python, ya que no se trata de un lenguaje tan complejo en comparación con otros. Las posibilidades que nos ofrece, son infinitas, siendo para nosotros, una de las herramientas top para trabajar, con datos masivos.

# Codigo
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
## Desicion Tree
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
## Logistic Regression
~~~

~~~
 ## Multilayer perceptron.
 ~~~
 
 ~~~
 
 
# Conclusiones.

Podemos notar como cada algoritmo tiene distintas predicciones y cual es su margen de error, lo cual es interesante ya que se trabajó con el mismo archivo de datos (csv), es interesante el cómo se puede manejar una gran cantidad de datos con, este tipo de algoritmos, nos llevamos un aprendizaje de cómo usar estas herramientas y la gran cantidad de cosas que podemos realizar. En conclusión conforme efectividad el mejor de los algoritmos es el de SVM, ya que en los promedios muestra menor rango de error.

# Referencias.

- MATLAB. (2015, 2 marzo). Support Vector Machine (SVM). MATLAB & Simulink. Recuperado 5 de junio de 2022, de https://es.mathworks.com/discovery/support-vector-machine.html

- Unir, V. (2021, 19 octubre). Árboles de decisión: en qué consisten y aplicación en Big Data. UNIR. Recuperado 5 de junio de 2022, de https://www.unir.net/ingenieria/revista/arboles-de-decision/

- Gonzalez, L. (2020, 21 agosto). Regresión Logística - Teoría. 🤖 Aprende IA. Recuperado 5 de junio de 2022, de https://aprendeia.com/regresion-logistica-multiple-machine-learning-teoria/#:%7E:text=La%20Regresi%C3%B3n%20Log%C3%ADstica%20es%20uno,cualquier%20problema%20de%20clasificaci%C3%B3n%20binaria.

- Sciencedirect. (2014a, abril 1). Multilayer Perceptron. Recuperado 5 de junio de 2022, de https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron






