import ServiceProtocol._
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.io.http.{HTTPRequestData, HTTPSchema, HTTPTransformer}
import com.microsoft.ml.spark.stages.FixedMiniBatchTransformer
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.spark.sql.functions.{col, udf}
import spray.json.DefaultJsonProtocol.{jsonFormat3, jsonFormat5, _}
import spray.json.{RootJsonFormat, _}

case class Node(Name: String,
                Id: String,
                Content: String)
case class Normalization(NormalizationName: String,
                         InputNodeId: String,
                         OutputNodeId: String)
case class InputData(inputText: String,
                     nodes: Seq[Node],
                     normalizations: Seq[Normalization],
                     trackingguid: String,
                     servicename: String)
object ServiceProtocol {
  implicit val NodeEnc: RootJsonFormat[Node] = jsonFormat3(Node.apply)
  implicit val NormEnc: RootJsonFormat[Normalization] = jsonFormat3(Normalization.apply)
  implicit val InputEnc: RootJsonFormat[InputData] = jsonFormat5(InputData.apply)
}
class QuickTest extends TestBase {
  import session.implicits._

  test("test"){

    val input = List(
      ("foo", "SpellNormalization"),
      ("foo2", "SpellNormalization"),
      ("foo3", "SpellNormalization"),
      ("foo4", "SpellNormalization"),
      ("foo5", "SpellNormalization"),
      ("foo6", "SpellNormalization"),
      ("foo7", "SpellNormalization"),
      ("foo8", "SpellNormalization")
    ).toDF("Content", "NormalizationName")

    val toRow = HTTPRequestData.makeToRowConverter
    val makeRequestUDF = udf({ (contents: Seq[String], normNames: Seq[String]) =>
      val p = new HttpPost("http://adqualityalgoframework1.azurewebsites.net/api/algoframework/Post")
      p.setEntity(new StringEntity(
        InputData(
          "test",
          contents.zipWithIndex.map { case (c, i) => Node("query", i.toString, c) },
          normNames.zipWithIndex.map { case (nn, i) => Normalization(nn, i.toString, i.toString) },
          "", "foo").toJson.compactPrint))
      p.setHeader("Content-Type", "application/json")
      toRow(new HTTPRequestData(p))
    }, HTTPRequestData.schema)
    val batched = new FixedMiniBatchTransformer()
      .setBatchSize(5)
      .transform(input)
      .withColumn("requests", makeRequestUDF(col("Content"), col("NormalizationName")))
    val resultDF = new HTTPTransformer()
      .setConcurrency(5)
      .setInputCol("requests")
      .setOutputCol("responses")
      .transform(batched)
      .withColumn("parsed", HTTPSchema.entity_to_string(col("responses").getItem("entity")))
    val results = resultDF.collect
    println(results)
  }
}
