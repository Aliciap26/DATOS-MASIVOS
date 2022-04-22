
/**
    * 1. Develop an algorithm in Scala that calculates the radius of a circle.
    */
var area: Int =12
val pi: Double=3.1416
var division: Double=0
division=area/pi
var raiz: Double=0
raiz=math.sqrt(division)
println(raiz)

/**
    * 2. Develop an algorithm in Scala that shows me if a number is prime number or not.
    */
var numero : Int = 29
var primo : Boolean = true
for(i <- Range(2, numero)) {
if((numero % i) == 0) {
primo = false
  }
}
if(primo){
println("El numero es primo")
} else {
println("No es numero primo")
}

/**
    * 3. Having the variable var bird = "tweet", use string interpolation to
     print "Estoy escribiendo un tweet" in Scala.
    */
 var bird = "tweet"
 val interpolar=s"Estoy escribiendo un $bird"
 println(interpolar)

 /**
    * 4. Having the variable var message = "Hola Luke yo soy tu padre!" use slice function to extract  "Luke" from it.
    */
    var mensaje = "Hola Luke yo soy tu padre!"
    mensaje slice(5,9)

   /**
    * 5. What's the difference between value (val) and variable (var) in Scala?.
    * Answer= Both are variables, but the difference between value and variable its that the first one is a constant valor, it means it cant be changed, by the other hand, variable value can be changed.
    */

/**
    * 6. Having the tuple (2,4,5,1,2,3,3.1416,23)  return the number 3.1416 from it.
    */
    val tupla = (2,4,5,1,2,3,3.1416,23)
    tupla._7
    