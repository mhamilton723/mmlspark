// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

val fullDependency = "compile->compile;test->test"

lazy val core = project
  .settings(
    Extras.defaultSettings,
    libraryDependencies ++= Seq(
      "com.typesafe" % "config" % "1.3.1",
      "org.apache.logging.log4j" %  "log4j-api"       % "2.8.1" % "provided",
      "org.apache.logging.log4j" %  "log4j-core"      % "2.8.1" % "provided",
      "org.apache.logging.log4j" %% "log4j-api-scala" % "2.8.1" % "provided"
    )
  )

lazy val codegen = project
  .settings(
      Extras.defaultSettings,
      Extras.noJar,
      // Running this project will load all jars, which will fail if they're
      // all "provided".  This magic makes it as if the "provided" is not
      // there for the run task.  See https://github.com/sbt/sbt-assembly and
      // http://stackoverflow.com/questions/18838944/
      run in Compile := Defaults.runTask(
        fullClasspath in Compile,
        mainClass in (Compile, run),
        runner in (Compile, run)).evaluated)
  .dependsOn(core % fullDependency)

lazy val lib = project
  .settings(Extras.defaultSettings)
  .dependsOn(core % fullDependency)

lazy val io = project
  .settings(Extras.defaultSettings)
  .dependsOn(core % fullDependency, lib % fullDependency)

lazy val opencv = project
  .settings(Extras.defaultSettings)
  .dependsOn(core % fullDependency, io % fullDependency)

lazy val cntk = project
  .settings(Extras.defaultSettings)
  .dependsOn(core % fullDependency, io % fullDependency,
      lib % fullDependency, opencv % fullDependency)

lazy val lightgbm = project
  .settings(Extras.defaultSettings)
  .dependsOn(core % fullDependency, lib % fullDependency)

val aggregationDeps = "compile->compile;optional"

lazy val MMLSpark = (project in file("."))
  .settings(
    Extras.defaultSettings,
    Extras.rootSettings,
    name := "mmlspark",
    version in ThisBuild := Extras.mmlVer)
  .aggregate(cntk, codegen, core, opencv, io, lib, lightgbm)
  .enablePlugins(ScalaUnidocPlugin)
  .dependsOn(
    cntk % aggregationDeps,
    codegen % aggregationDeps,
    core % aggregationDeps,
    opencv % aggregationDeps,
    io % aggregationDeps,
    lib % aggregationDeps,
    lightgbm % aggregationDeps)

