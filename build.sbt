import java.io.{File, PrintWriter}
import java.net.URL

import org.apache.commons.io.FileUtils

import scala.sys.process.Process

val condaEnvName = "mmlspark"
name := "mmlspark"
organization := "com.microsoft.ml.spark"
scalaVersion := "2.11.12"

val sparkVersion = "2.4.0"
val sprayVersion = "1.3.4"
val cntkVersion = "2.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "compile",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "compile",
  "org.scalactic" %% "scalactic" % "3.0.5",
  "org.scalatest" %% "scalatest" % "3.0.5",
  "io.spray" %% "spray-json" % "1.3.2",
  "com.microsoft.cntk" % "cntk" % cntkVersion,
  "org.openpnp" % "opencv" % "3.2.0-1",
  "com.jcraft" % "jsch" % "0.1.54",
  "com.jcraft" % "jsch" % "0.1.54",
  "org.apache.httpcomponents" % "httpclient" % "4.5.6",
  "com.microsoft.ml.lightgbm" % "lightgbmlib" % "2.2.350"
)

lazy val IntegrationTest2 = config("it").extend(Test)

def join(folders: String*): File = {
  folders.tail.foldLeft(new File(folders.head)) { case (f, s) => new File(f, s) }
}

val createCondaEnvTask = TaskKey[Unit]("createCondaEnv", "create conda env")
createCondaEnvTask := {
  val s = streams.value
  val hasEnv = Process("conda env list").lineStream.toList
    .map(_.split("\\s+").head).contains(condaEnvName)
  if (!hasEnv) {
    Process(
      "conda env create -f environment.yaml",
      new File(".")) ! s.log
  } else {
    println("Found conda env " + condaEnvName)
  }
}

val cleanCondaEnvTask = TaskKey[Unit]("cleanCondaEnv", "create conda env")
cleanCondaEnvTask := {
  val s = streams.value
  Process(
    s"conda env remove --name $condaEnvName -y",
    new File(".")) ! s.log
}

def osPrefix: Seq[String] = {
  if (sys.props("os.name").toLowerCase.contains("windows")) {
    Seq("cmd", "/C")
  } else {
    Seq()
  }
}

def activateCondaEnv: Seq[String] = {
  if (sys.props("os.name").toLowerCase.contains("windows")) {
    osPrefix ++ Seq("activate", condaEnvName, "&&")
  } else {
    Seq()
    //TODO figure out why this doesent work
    //Seq("/bin/bash", "-l", "-c", "source activate " + condaEnvName, "&&")
  }
}

val packagePythonTask = TaskKey[Unit]("packagePython", "Package python sdk")
val genDir = join("target", "scala-2.11", "generated")
val pythonSrcDir = join(genDir.toString, "src", "python")
val pythonPackageDir = join(genDir.toString, "package", "python")
val pythonTestDir = join(genDir.toString, "test", "python")

def pythonizeVersion(v: String): String = {
  if (v.contains("+")){
    v.split("+".head).head + ".dev1"
  }else{
    v
  }
}

packagePythonTask := {
  val s = streams.value
  (run in IntegrationTest2).toTask("").value
  createCondaEnvTask.value
  val destPyDir = join(baseDirectory.value.getAbsolutePath, "target", "scala-2.11", "classes", "mmlspark")
  if (destPyDir.exists()) FileUtils.forceDelete(destPyDir)
  FileUtils.copyDirectory(join(pythonSrcDir.getAbsolutePath, "mmlspark"), destPyDir)
  
  Process(
    activateCondaEnv ++
      Seq(s"python", "setup.py", "bdist_wheel", "--universal", "-d", s"${pythonPackageDir.absolutePath}"),
    pythonSrcDir,
    "MML_PY_VERSION" -> pythonizeVersion(version.value)) ! s.log
}

val installPipPackageTask = TaskKey[Unit]("installPipPackage", "install python sdk")

installPipPackageTask := {
  val s = streams.value
  publishLocal.value
  packagePythonTask.value
  Process(
    activateCondaEnv ++ Seq("pip", "install",
      s"mmlspark-${pythonizeVersion(version.value)}-py2.py3-none-any.whl"),
    pythonPackageDir) ! s.log
}

val testPythonTask = TaskKey[Unit]("testPython", "test python sdk")

testPythonTask := {
  val s = streams.value
  installPipPackageTask.value
  Process(
    activateCondaEnv ++ Seq("python", "tools2/run_all_tests.py"),
    new File("."),
    "MML_VERSION" -> version.value
  ) ! s.log
}

val getDatasetsTask = TaskKey[Unit]("getDatasets", "download datasets used for testing")
val datasetName = "datasets-2019-05-02.tgz"
val datasetUrl = new URL(s"https://mmlspark.blob.core.windows.net/installers/$datasetName")
val datasetDir = settingKey[File]("The directory that holds the dataset")
datasetDir := {
  join(target.value.toString, "scala-2.11", "datasets", datasetName.split(".".toCharArray.head).head)
}

getDatasetsTask := {
  val d = datasetDir.value.getParentFile
  val f = new File(d, datasetName)
  if (!d.exists()) d.mkdirs()
  if (!f.exists()) {
    FileUtils.copyURLToFile(datasetUrl, f)
    UnzipUtils.unzip(f, d)
  }
}

val setupTask = TaskKey[Unit]("setup", "set up library for intellij")
setupTask := {
  (Compile / compile).toTask.value
  (Test / compile).toTask.value
  (IntegrationTest2 / compile).toTask.value
  getDatasetsTask.value
}

publishM2 := (publishM2 dependsOn packagePythonTask).value
publishLocal := (publishLocal dependsOn packagePythonTask).value

val publishBlob = TaskKey[Unit]("publishBlob", "publish the library to mmlspark blob")
publishBlob := {
  val s = streams.value
  publishM2.value
  val scalaVersionSuffix = scalaVersion.value.split(".".toCharArray.head).dropRight(1).mkString(".")
  val nameAndScalaVersion = s"${name.value}_$scalaVersionSuffix"
  
  val localPackageFolder = join(
    Seq(new File(new URI(Resolver.mavenLocal.root)).getAbsolutePath)
      ++ organization.value.split(".".toCharArray.head)
      ++ Seq(nameAndScalaVersion, version.value): _*).toString

  val blobMavenFolder = organization.value.replace(".", "/") +
    s"/$nameAndScalaVersion/${version.value}"
  val command = Seq("az", "storage", "blob", "upload-batch",
    "--source", localPackageFolder,
    "--destination", "maven",
    "--destination-path", blobMavenFolder,
    "--account-name", "mmlspark",
    "--account-key", Secrets.storageKey)
  println(command.mkString(" "))
  Process(osPrefix ++ command) ! s.log
}

val settings = Seq(
  (scalastyleConfig in Test) := baseDirectory.value / "scalastyle-test-config.xml",
  logBuffered in Test := false,
  buildInfoKeys := Seq[BuildInfoKey](
    name, version, scalaVersion, sbtVersion,
    baseDirectory, datasetDir),
  parallelExecution in Test := false,
  buildInfoPackage := "com.microsoft.ml.spark.build") ++
  inConfig(IntegrationTest2)(Defaults.testSettings)

lazy val mmlspark = (project in file("."))
  .configs(IntegrationTest2)
  .enablePlugins(BuildInfoPlugin)
  .enablePlugins(ScalaUnidocPlugin)
  .settings(settings: _*)

homepage := Some(url("https://github.com/Azure/mmlspark"))
scmInfo := Some(ScmInfo(url("https://github.com/Azure/mmlspark"), "git@github.com:Azure/mmlspark.git"))
developers := List(
  Developer("mhamilton723", "Mark Hamilton",
    "mmlspark-support@microsoft.com", url("https://github.com/mhamilton723")),
  Developer("imatiach-msft", "Ilya Matiach",
    "mmlspark-support@microsoft.com", url("https://github.com/imatiach-msft")),
  Developer("drdarshan", "Sudarshan Raghunathan",
    "mmlspark-support@microsoft.com", url("https://github.com/drdarshan"))
)

credentials += Credentials("Sonatype Nexus Repository Manager",
  "oss.sonatype.org",
  Secrets.nexusUsername,
  Secrets.nexusPassword)

pgpPassphrase := Some(Secrets.pgpPassword.toCharArray)
pgpSecretRing := {
  val temp = File.createTempFile("secret", ".asc")
  new PrintWriter(temp) {
    write(Secrets.pgpPrivate); close()
  }
  temp
}
pgpPublicRing := {
  val temp = File.createTempFile("public", ".asc")
  new PrintWriter(temp) {
    write(Secrets.pgpPublic); close()
  }
  temp
}

licenses += ("MIT", url("https://github.com/Azure/mmlspark/blob/master/LICENSE"))
publishMavenStyle := true
publishTo := Some(
  if (isSnapshot.value) {
    Opts.resolver.sonatypeSnapshots
  } else {
    Opts.resolver.sonatypeStaging
  }
)
