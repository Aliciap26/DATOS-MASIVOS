  /*1.-Comienza una simple sesión Spark.
    */
spark-shell

 /*2.-Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos.
    */
   val dataframe=spark.read.option("header","true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

    /*3.-¿Cuáles son los nombres de las columnas?
    */
    dataframe.columns

/*4.-¿Cómo es el esquema?
    */
    dataframe.printSchema()

    
 /*5.-Imprime las primeras 5 columnas.
    */
    dataframe.head(5)

/*6.-Usa describe () para aprender sobre el DataFrame.
    */
    dataframe.describe().show
    
/*7. Crea un nuevo dataframe con una columna nueva llamada “HV Ratio” que es la relación que
existe entre el precio de la columna “High” frente a la columna “Volumen” de acciones
negociadas por un día. Hint - es una operación*/

val dataframenew = dataframe.withColumn("HV Ratio", dataframe("High")- dataframe("Volume"))
/* mostrar columna nueva*/
dataframenew.show()
/*8. ¿Qué día tuvo el pico mas alto en la columna “Open”?
*/
dataframe.select(max("Open")).show()
dataframe.orderBy($"Open".desc).show(1)
/*9. ¿Cuál es el significado de la columna Cerrar “Close” en el contexto de información financiera,
expliquelo no hay que codificar nada? R= We could find a relationship between the column Close and High,when the column High raises, the other column (Close) also raises, that's the main pattern we could find in this dataframe.    */

/* 10. ¿Cuál es el máximo y mínimo de la columna “Volumen”? */

dataframe.select(max("volume")).show()
dataframe.select(min("volume")).show()


/* Con Sintaxis Scala/Spark $ conteste los siguiente:*/
/*a. ¿Cuántos días fue la columna “Close” inferior a $ 600?*/ 
dataframe.filter($"Close"<600).count()

/* b. ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?

val P1 = dataframe.filter($"High"> 500). count()
val P2 = dataframe.filter($"High">0).count()
val P3 : Double = P1*100
P3/P2


/*c. ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?*/
dataframe.select(corr("High","Volume")).show

/* d. ¿Cuál es el máximo de la columna “High” por año?*/
val year1 = dataframe.withColumn("Year", year(dataframe("Date")))
val Year2 = year1.select($"Year", $"High").groupBy("Year").max()
Year2.select($"Year", $"max(High)").show()



/* e. ¿Cuál es el promedio de columna “Close” para cada mes del calendario?*/
val Mes = dataframe.withColumn("Month", month (dataframe("Date")))

val mespr= Mes.select($"Month",$"Close").groupBy("Month").mean()

mespr.select($"Month",$"avg(Close)").show()
