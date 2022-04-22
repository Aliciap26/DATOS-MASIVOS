/**
   *Algorithm#1
   */

def algthm1(n:Int):Int={
if(n<2){
return n
}
else{
return algthm(n-1)+algthm(n-2)
}
}



/**
   *Algorithm#2
   */

   def algthm2(n:Int):Double={
   if(n<2){
   return n}
   else{ var p=((1+math.sqrt(5))/2)
         var j=((math.pow(p,n)-(math.pow((1-p),n)))/(math.sqrt(5)))
         return j
   }
   }




/**
   *Algorithm#3
   */

def algthm3(n:Int):Int={
var a=0
var b=1
var c=0
var i=0
if(i<n){
c=b+a
a=b
b=c
i=i+1} 
return a
}




/**
   *Algorithm#4
   */

def algthm4(n:Int):Int={
var a=0
var b=1
var i=0
if(i<n){
b=b+a
a=b-a
i=i+1
}
return b}







