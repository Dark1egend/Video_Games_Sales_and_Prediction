import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

val conf = new SparkConf().setAppName("appName").setMaster("local")
val sc = new SparkContext(conf)

// Create SparkSession
val spark = SparkSession.builder()
  .appName("VideoGameSalesAnalysis")
  .master("local")
  .getOrCreate()

// Load the dataset and Preprocessing of Dataset
var data = sc.textFile("/home/sriviswa/Downloads/vgsales4.csv")

val data1 = data.map(x => x.replace("#","")).map(x => x.replace(";","")).map(x => x.replace(":","")).map(x => x.replace("-","")).map(x => x.replace("_","")).map(x => x.replace("(","")).map(x => x.replace(")","")).map(x => x.replace("/","")).map(x => x.replace("\\","")).map(x => x.replace("'",""))
var data = data1.filter(x => x.split(",")(3).matches("[0-9]+"))

//Analysis

//Top-selling Games
val topGames = data.map(line => line.split(",")).map(fields => (fields(1), fields(10).toDouble)).sortBy(_._2, ascending = false).take(10)

//regional breakdown of top 10 selling games
val topGamesByRegion = data.map(line => line.split(",")).map(fields => (fields(1), fields(6).toDouble, fields(7).toDouble, fields(8).toDouble, fields(9).toDouble)).sortBy(_._3, ascending = false).take(10)

// Top-selling Games by Regionwise(NA)
val topGamesByRegionNA= data.map(line => line.split(",")).map(fields => (fields(1), fields(6).toDouble)).sortBy(_._2, ascending = false).take(10)
//Similarly by changing the column in consideration we can calculate the Top Selling in other regions,

//Top Games by Publisher
val topGamesByPublisher = data.map(line => line.split(",")).map(fields => (fields(1), fields(5), fields(10).toDouble)).sortBy(_._3, ascending = false).take(10)

// Platform Analysis
val platformSales = data.map(line => line.split(",")).map(fields => (fields(2), 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false).collect()

//Games Publisher each Year
val gamesPublishedByYear = data.map(line => line.split(",")).map(fields => (fields(3), 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false).collect()

//Yearly Sales Trends
val yearlySalesTrends = data.map(line => line.split(",")).map(fields => (fields(3), fields(10).toDouble)).map{ case (year, sales) => (year, sales) }.reduceByKey(_ + _).sortByKey()

//Genre Distribution by Sales
val genreDistribution = data.map(line => line.split(",")).map(fields => (fields(4), fields(10).toDouble)).reduceByKey(_+_).sortBy(_._2, ascending=false).take(10)

//Genre Distribution
val gamesInGenre = data.map(line => line.split(",")).map(fields => (fields(4), 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false).collect()

//Publisher Distribution
val gamesByPublishher = data.map(line => line.split(",")).map(fields => (fields(5), 1)).reduceByKey(_ + _).sortBy(_._2, ascending = false).collect()

//Region Sales over Time(NA)
val salesInNAEachYear=data.map(line=>line.split(",")).map(fields=>(fields(3),fields(6).toDouble)).reduceByKey(_+_).sortByKey().collect()
//Similarly by changing the fields in consideration we can calculate the Region Sales over time in other regions

//Genre Sales over Time
val salesInGenresEachYear = data.map(line => line.split(",")).map(fields => ((fields(3), fields(4)), fields(10).toDouble)).reduceByKey(_ + _).map { case ((year, genre), sales) => (year, (genre, sales)) }.groupByKey().sortByKey()

salesInGenresEachYear.foreach { case (year, genreSales) =>
  println("Year: " + year)
  genreSales.foreach { case (genre, sales) =>
    println("Genre: " + genre + ", Sales: " + sales)
  }
  println()
}

//Sales by Top 10 Publishers over time
val topPublishers = data.map(line=>line.split(",")).map(fields=>(fields(5),fields(3),fields(10).toDouble)).groupBy(_._1).mapValues(_.map(entry => (entry._2,entry._3))).sortByKey().takeOrdered(10)(Ordering[Double].reverse.on(_._2.map(_._2).sum))

//Total market sales for each genre(region wise)(NA)
val genreSalesNA = data.map(line => {
  val fields =line.split(",")
  (fields(4),fields(6).toDouble)
}).groupBy(_._1).mapValues(_.map(_._2).sum).sortBy(_._1).collect()
//Similarly changing the fields will give us total market sales for each genre in other regions

//Rank of each game based on its genre
val gameRankByGenre = data.map(line => line.split(",")).map(fields => ((fields(4), fields(1)), fields(10).toDouble)).reduceByKey(_ + _).groupBy(_._1._1).flatMap { case (genre, genreSales) =>
val sortedGenreSales = genreSales.toList.sortBy(-_._2)
val rankedGenreSales = sortedGenreSales.zipWithIndex.map { case (((_, game), sales), rank) =>
  (game, genre, rank + 1, sales)
}
rankedGenreSales
}

gameRankByGenre.foreach { case (game, genre, rank, sales) =>
  println(s"Game: $game, Genre: $genre, Rank: $rank, Sales: $sales")
}

//Highest Selling Game by Each Genre
val highestSellingGameByGenre = data.map(line => line.split(",")).map(fields => (fields(4), (fields(1), fields(10).toDouble))).reduceByKey { case (game1, game2) =>
     if (game1._2 > game2._2) game1 else game2
     }.mapValues(_._1).collect()

// Total Global Sales
val totalGlobalSales = data.map(line => line.split(",")).map(fields => fields(10).toDouble).sum()

//Most Played Game each year
val mostPlayedGameByYear = data.map(line => line.split(",")).map(fields => ((fields(3), fields(1)), fields(10).toDouble)).reduceByKey(Math.max).map { case ((year, game), sales) => (year, game, sales) }.sortBy(_._1)
mostPlayedGameByYear.collect().foreach { case (year, game, sales) => println(s"Year: $year, Most Played Game (Global Sales): $game, Sales: $sales") }












