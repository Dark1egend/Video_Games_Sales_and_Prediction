import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GBTRegressor


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

val dropcolumn = Array("Rank", "NA_Sales","EU_Sales","JP_Sales","Other_Sales")
val data = dataset.drop(dropcolumn: _*)


val featureColumns =Array("Name","Platform","Year","Genre","Publisher")
val targetColumn = "Global_Sales"


val tokenizerStages: Array[Tokenizer] = featureColumns.map { column =>
  new Tokenizer()
    .setInputCol(column)
    .setOutputCol(s"${column}_words")
}

val hashingTFStages: Array[HashingTF] = featureColumns.map { column =>
  new HashingTF()
    .setInputCol(s"${column}_words")
    .setOutputCol(s"${column}_rawFeatures")
    .setNumFeatures(25000)
}

val idfStages: Array[IDF] = featureColumns.map { column =>
  new IDF()
    .setInputCol(s"${column}_rawFeatures")
    .setOutputCol(s"${column}_idfFeatures")
}

val rf = new GBTRegressor()
  .setLabelCol(targetColumn)
  .setFeaturesCol("features")

val assembler = new VectorAssembler()
  .setInputCols(featureColumns.map(c => s"${c}_idfFeatures"))
  .setOutputCol("features")


val pipeline = new Pipeline()
  .setStages(tokenizerStages ++ hashingTFStages ++ idfStages :+ assembler :+ rf)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

val model = pipeline.fit(trainingData)

val predictions = model.transform(testData)

predictions.select("prediction", targetColumn).show()

model.save("/home/sriviswa/Downloads/GBR_model")

// Evaluate the model
val evaluator = new RegressionEvaluator()
  .setLabelCol(targetColumn)
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE): $rmse")



import spark.implicits._

// Load the trained model
var modell = PipelineModel.load("/home/sriviswa/Downloads/GBR_model")

val inputData = Seq(
  ("2010", "PS4", "Action", "EA")
).toDF("Year", "Platform", "Genre", "Publisher")

val transformedData = modell.transform(inputData)

val predictedSales = transformedData.select("prediction").head().getDouble(0)

println(s"Predicted Sales: $predictedSales")

spark.stop()