import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression


// Create a Spark session
val spark = SparkSession.builder()
  .appName("SalesPredictionText")
  .master("local")
  .getOrCreate()

// Load the dataset into a DataFrame
val dataset = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/home/sriviswa/Downloads/vgsales.csv")

val columnsToDrop = Array("Rank", "NA_Sales","EU_Sales","JP_Sales","Other_Sales")
val data = dataset.drop(columnsToDrop: _*)

// Select the relevant text columns and target variable
val textColumns =Array("Name","Platform","Year","Genre","Publisher")
val targetColumn = "Global_Sales"

//Data Transformation

// Tokenizing each feature column and output Tokenized words
val tokenizerStages: Array[Tokenizer] = textColumns.map { column =>
  new Tokenizer()
    .setInputCol(column)
    .setOutputCol(s"${column}_words")
}


//Converts the tokenized column into a fized sized feature vector
val hashingTFStages: Array[HashingTF] = textColumns.map { column =>
  new HashingTF()
    .setInputCol(s"${column}_words")
    .setOutputCol(s"${column}_rawFeatures")
    .setNumFeatures(25000)
}

//The IDF is a feature transformer that scales down the impact of more frequent words
// and boosts the importance of rarer words in the feature vectors
val idfStages: Array[IDF] = textColumns.map { column =>
  new IDF()
    .setInputCol(s"${column}_rawFeatures")
    .setOutputCol(s"${column}_idfFeatures")
}

// Create the model
val rf = new LinearRegression()
  .setLabelCol(targetColumn)
  .setFeaturesCol("features")

val assembler = new VectorAssembler()
  .setInputCols(textColumns.map(c => s"${c}_idfFeatures"))
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(tokenizerStages ++ hashingTFStages ++ idfStages :+ assembler :+ rf)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

val model = pipeline.fit(trainingData)

//model.save("/home/sriviswa/Downloads/lr_model")

val predictions = model.transform(testData)

predictions.select("prediction", targetColumn).show()

// Evaluate the model
val evaluator = new RegressionEvaluator()
  .setLabelCol(targetColumn)
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE): $rmse")

// Load the trained model
var modell = PipelineModel.load("/home/sriviswa/Downloads/lr_model")

import spark.implicits._

val inputData = Seq(
  ("Hello","2010", "PS4", "Action", "EA")
).toDF("Name","Year", "Platform", "Genre", "Publisher")

val transformedData = modell.transform(inputData)

val predictedSales = transformedData.select("prediction").head().getDouble(0)

println(s"Predicted Sales: $predictedSales")

spark.stop()