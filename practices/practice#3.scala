scala> var lista = List("Rojo","Blanco", "Negro")
lista: List[String] = List(Rojo, Blanco, Negro)

scala> lista = "Verde" :: "Amarillo" :: "Naranja" :: "Perla" :: lista
lista: List[String] = List(Verde, Amarillo, Naranja, Perla, Rojo, Blanco, Negro)

scala> print(lista)
List(Verde, Amarillo, Naranja, Perla, Rojo, Blanco, Negro)
scala> lista slice(0,3)
res1: List[String] = List(Verde, Amarillo, Naranja)

scala> Array.range(1,1000,5)
res2: Array[Int] = Array(1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386, 391, 396, 401, 406, 411, 416, 421, 426, 431, 436, 441, 446, 451, 456, 461, 466, 471, 476, 481, 486, 491, 496, 501, 506, 511, 516, 521, 526, 531, 536, 541, 546, 551, 556, 561, 566, 571, 576, 581, 586, 591, 596, 601, 606, 611, 616, 621, 626, 631, 636, 641, 646, 651, 656, 661, 666, 671, 676, 681, 686, 691, 696, 701, 706, 711, 716, 721, 726, 731, 736, 741, 746, 751, 756, 761, 766, 771, 776, 781, 786, 791,...
scala>

scala> var Lista11 = List(1,2,3,4,5,6,7)
Lista11: List[Int] = List(1, 2, 3, 4, 5, 6, 7)

scala> val L=List(1,3,3,4,7,3,7)
L: List[Int] = List(1, 3, 3, 4, 7, 3, 7)

scala> val L=List(1,3,3,4,7,3,7).toSet
L: scala.collection.immutable.Set[Int] = Set(1, 3, 4, 7)
                         
scala> val Nombre = collection.mutable.Map(("Jose",20),("Luis",24),("Ana", 23),("Susana",27))
Nombre: scala.collection.mutable.Map[String,Int] = Map(Susana -> 27, Ana -> 23, Luis -> 24, Jose -> 20)

scala> Nombre.keys
res4: Iterable[String] = Set(Susana, Ana, Luis, Jose)

scala> Nombre +=("Miguel"->23)
res5: Nombre.type = Map(Susana -> 27, Ana -> 23, Miguel -> 23, Luis -> 24, Jose -> 20)

scala>
                 
