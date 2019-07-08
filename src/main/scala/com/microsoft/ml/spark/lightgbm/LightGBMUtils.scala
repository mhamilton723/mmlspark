// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.lightgbm

import java.io._
import java.net.{InetAddress, ServerSocket, Socket}
import java.util.concurrent.Executors

import com.microsoft.ml.lightgbm._
import com.microsoft.ml.spark.core.env.NativeLoader
import com.microsoft.ml.spark.featurize.{Featurize, FeaturizeUtilities}
import org.apache.http.conn.util.InetAddressUtils
import org.apache.spark.lightgbm.BlockManagerUtils
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.slf4j.Logger

import scala.collection.immutable.HashSet
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration.{Duration, SECONDS}
import scala.concurrent.{ExecutionContext, Future}

/** Helper utilities for LightGBM learners */
object LightGBMUtils {
  def validate(result: Int, component: String): Unit = {
    if (result == -1) {
      throw new Exception(component + " call failed in LightGBM with error: "
                            + lightgbmlib.LGBM_GetLastError())
    }
  }

  /** Loads the native shared object binaries lib_lightgbm.so and lib_lightgbm_swig.so
    */
  def initializeNativeLibrary(): Unit = {
    val osPrefix = NativeLoader.getOSPrefix()
    new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm")
    new NativeLoader("/com/microsoft/ml/lightgbm").loadLibraryByName(osPrefix + "_lightgbm_swig")
  }

  def featurizeData(dataset: Dataset[_], labelColumn: String, featuresColumn: String,
                    weightColumn: Option[String] = None, groupColumn: Option[String] = None): PipelineModel = {
    // Create pipeline model to featurize the dataset
    val oneHotEncodeCategoricals = true
    val featuresToHashTo = FeaturizeUtilities.numFeaturesTreeOrNNBased
    val featureColumns = dataset.columns.filter(col => col != labelColumn &&
      weightColumn.forall(_ != col) && groupColumn.forall(_ != col)).toSeq
    val featurizer = new Featurize()
      .setFeatureColumns(Map(featuresColumn -> featureColumns))
      .setOneHotEncodeCategoricals(oneHotEncodeCategoricals)
      .setNumberOfFeatures(featuresToHashTo)
    featurizer.fit(dataset)
  }

  def getBoosterPtrFromModelString(lgbModelString: String): SWIGTYPE_p_void = {
    val boosterOutPtr = lightgbmlib.voidpp_handle()
    val numItersOut = lightgbmlib.new_intp()
    LightGBMUtils.validate(
      lightgbmlib.LGBM_BoosterLoadModelFromString(lgbModelString, numItersOut, boosterOutPtr),
      "Booster LoadFromString")
    lightgbmlib.voidpp_value(boosterOutPtr)
  }

  def getCategoricalIndexes(df: DataFrame,
                            featuresCol: String,
                            categoricalColumnIndexes: Array[Int],
                            categoricalColumnSlotNames: Array[String]): Array[Int]  = {
    val categoricalSlotNamesSet = HashSet(categoricalColumnSlotNames: _*)
    val featuresSchema = df.schema(featuresCol)
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    val categoricalIndexes =
      if (metadata.attributes.isEmpty) Array[Int]()
      else {
        metadata.attributes.get.zipWithIndex.flatMap {
          case (null, _) => Iterator()
          case (attr, idx) =>
            if (attr.name.isDefined && categoricalSlotNamesSet.contains(attr.name.get)) {
              Iterator(idx)
            } else {
              attr match {
                case _: NumericAttribute | UnresolvedAttribute => Iterator()
                case binAttr: BinaryAttribute => Iterator(idx)
                case nomAttr: NominalAttribute => Iterator(idx)
              }
            }
        }
      }
    categoricalColumnIndexes.union(categoricalIndexes).distinct
  }

  /**
    * Opens a socket communications channel on the driver, starts a thread that
    * waits for the host:port from the executors, and then sends back the
    * information to the executors.
    * @param numWorkers The total number of training workers to wait for.
    * @return The address and port of the driver socket.
    */
  def createDriverNodesThread(numWorkers: Int, df: DataFrame,
                              log: Logger, timeout: Double,
                              barrierExecutionMode: Boolean): (String, Int, Future[Unit]) = {
    // Start a thread and open port to listen on
    implicit val context = ExecutionContext.fromExecutor(Executors.newSingleThreadExecutor())
    val driverServerSocket = new ServerSocket(0)
    // Set timeout on socket
    val duration = Duration(timeout, SECONDS)
    if (duration.isFinite()) {
      driverServerSocket.setSoTimeout(duration.toMillis.toInt)
    }
    val f = Future {
      var emptyWorkerCounter = 0
      val hostAndPorts = ListBuffer[(Socket, String)]()
      if (barrierExecutionMode) {
        log.info(s"driver using barrier execution mode")
        var finished = false
        while (!finished) {
          log.info("driver accepting a new connection...")
          val driverSocket = driverServerSocket.accept()
          val reader = new BufferedReader(new InputStreamReader(driverSocket.getInputStream))
          val comm = reader.readLine()
          if (comm == LightGBMConstants.finishedStatus) {
            log.info("driver received all workers from barrier stage")
            finished = true
          } else if (comm == LightGBMConstants.ignoreStatus) {
            log.info("driver received ignore status from worker")
          } else {
            log.info(s"driver received socket from worker: $comm")
            val socketAndComm = (driverSocket, comm)
            hostAndPorts += socketAndComm
          }
        }
      } else {
        log.info(s"driver expecting $numWorkers connections...")
        while (hostAndPorts.size + emptyWorkerCounter < numWorkers) {
          log.info("driver accepting a new connection...")
          val driverSocket = driverServerSocket.accept()
          val reader = new BufferedReader(new InputStreamReader(driverSocket.getInputStream))
          val comm = reader.readLine()
          if (comm == LightGBMConstants.ignoreStatus) {
            log.info("driver received ignore status from worker")
            emptyWorkerCounter += 1
          } else {
            log.info(s"driver received socket from worker: $comm")
            val socketAndComm = (driverSocket, comm)
            hostAndPorts += socketAndComm
          }
        }
      }
      // Concatenate with commas, eg: host1:port1,host2:port2, ... etc
      val allConnections = hostAndPorts.map(_._2).mkString(",")
      log.info(s"driver writing back to all connections: $allConnections")
      // Send data back to all threads on executors
      hostAndPorts.foreach(hostAndPort => {
        val writer = new BufferedWriter(new OutputStreamWriter(hostAndPort._1.getOutputStream))
        writer.write(allConnections + "\n")
        writer.flush()
      })
      log.info("driver closing all sockets and server socket")
      hostAndPorts.foreach(_._1.close())
      driverServerSocket.close()
    }
    val host = getDriverHost(df)
    val port = driverServerSocket.getLocalPort
    log.info(s"driver waiting for connections on host: ${host} and port: $port")
    (host, port, f)
  }

  /** Returns an integer ID for the current node.
    * @return In cluster, returns the executor id.  In local case, returns the worker id.
    */
  def getId(): Int = {
    val executorId = SparkEnv.get.executorId
    val ctx = TaskContext.get
    val partId = ctx.partitionId
    // If driver, this is only in test scenario, make each partition a separate worker
    val id = if (executorId == "driver") partId else executorId
    val idAsInt = id.toString.toInt
    idAsInt
  }

  def getHostToIP(hostname: String): String = {
    if (InetAddressUtils.isIPv4Address(hostname) || InetAddressUtils.isIPv6Address(hostname))
      hostname
    else
      InetAddress.getByName(hostname).getHostAddress
  }

  /** Get number of cores from dummy dataset for 1 executor.
    * Note: all executors have same number of cores,
    * and this is more reliable than getting value from conf.
    * @param dataset The dataset containing the current spark session.
    * @return The number of cores per executor.
    */
  def getNumCoresPerExecutor(dataset: Dataset[_], log: Logger): Int = {
    val spark = dataset.sparkSession
    val confTaskCpus =
      try {
        spark.sparkContext.getConf.get("spark.task.cpus", "1").toInt
      } catch {
        case _: NoSuchElementException => 1
      }
    try {
      val confCores = spark.sparkContext.getConf
        .get("spark.executor.cores").toInt
      val coresPerExec = confCores / confTaskCpus
      log.info(s"LightGBM calculated num cores per executor as $coresPerExec from $confCores " +
        s"cores and $confTaskCpus task CPUs")
      coresPerExec
    } catch {
      case _: NoSuchElementException =>
        // If spark.executor.cores is not defined, get the cores per JVM
        import spark.implicits._
        val numMachineCores = spark.range(0, 1)
          .map(_ => java.lang.Runtime.getRuntime.availableProcessors).collect.head
        val coresPerExec = numMachineCores / confTaskCpus
        log.info(s"LightGBM calculated num cores per executor as $coresPerExec from " +
          s"$numMachineCores machine cores from JVM and $confTaskCpus task CPUs")
        coresPerExec
    }
  }

  private def getDriverHost(dataset: Dataset[_]): String = {
    val blockManager = BlockManagerUtils.getBlockManager(dataset)
    blockManager.master.getMemoryStatus.toList.flatMap({ case (blockManagerId, _) =>
      if (blockManagerId.executorId == "driver") Some(getHostToIP(blockManagerId.host))
      else None
    }).head
  }

  /** Returns a list of executor id and host.
    * @param dataset The dataset containing the current spark session.
    * @return List of executors as an array of (id,host).
    */
  def getExecutors(dataset: Dataset[_]): Array[(Int, String)] = {
    val blockManager = BlockManagerUtils.getBlockManager(dataset)
    blockManager.master.getMemoryStatus.toList.flatMap({ case (blockManagerId, _) =>
      if (blockManagerId.executorId == "driver") None
      else Some((blockManagerId.executorId.toInt, getHostToIP(blockManagerId.host)))
    }).toArray
  }

  /** Returns the number of executors * number of cores.
    * @param dataset The dataset containing the current spark session.
    * @param numCoresPerExec The number of cores per executor.
    * @return The number of executors * number of cores.
    */
  def getNumExecutorCores(dataset: Dataset[_], numCoresPerExec: Int): Int = {
    val executors = getExecutors(dataset)
    if (!executors.isEmpty) {
      executors.length * numCoresPerExec
    } else {
      val master = dataset.sparkSession.sparkContext.master
      val rx = "local(?:\\[(\\*|\\d+)(?:,\\d+)?\\])?".r
      master match {
        case rx(null)  => 1
        case rx("*")   => Runtime.getRuntime.availableProcessors()
        case rx(cores) => cores.toInt
        case _         => BlockManagerUtils.getBlockManager(dataset)
                            .master.getMemoryStatus.size
      }
    }
  }

  /** Converts a host,id pair to the lightGBM host:port format.
    * @param hostAndId The host,id.
    * @param defaultListenPort The default listen port.
    * @return The string lightGBM representation of host:port.
    */
  def toAddr(hostAndId: (String, Int), defaultListenPort: Int): String =
    hostAndId._1 + ":" + (defaultListenPort + hostAndId._2)

  /** Returns the nodes from mapPartitions.
    * Only run in case when num partitions < num executors.
    * @param processedData The input data.
    * @param defaultListenPort The default listening port.
    * @param executorToHost Map from executor id to host name.
    * @return The list of nodes in host:port format.
    */
  def getNodesFromPartitions(processedData: DataFrame, defaultListenPort: Int,
                             executorToHost: Map[Int, String]): Array[(Int, String)] = {
    import processedData.sparkSession.implicits._
    val nodes =
      processedData.mapPartitions((_: Iterator[Row]) => {
        val id = getId()
        Array((id, executorToHost(id))).toIterator
      }).collect()
    nodes
  }

  /** Returns the nodes from mapPartitions.
    * Only run in local[*] case.
    * @param processedData The input data.
    * @param defaultListenPort The default listening port.
    * @return The list of nodes in host:port format.
    */
  def getNodesFromPartitionsLocal(processedData: DataFrame,
                                  defaultListenPort: Int): Array[(Int, String)] = {
    import processedData.sparkSession.implicits._
    val blockManager = BlockManagerUtils.getBlockManager(processedData)
    val host = blockManager.master.getMemoryStatus.flatMap({ case (blockManagerId, _) =>
      Some(getHostToIP(blockManagerId.host))
    }).head
    val nodes =
      processedData.mapPartitions((_: Iterator[Row]) => {
        // The logic below is to get it to run in local[*] spark context
        val id = getId()
        Array((id, host)).toIterator
      }).collect()
    nodes
  }

  def intToPtr(value: Int): SWIGTYPE_p_int64_t = {
    val longPtr = lightgbmlib.new_longp()
    lightgbmlib.longp_assign(longPtr, value)
    lightgbmlib.long_to_int64_t_ptr(longPtr)
  }

  def generateData(numRows: Int, rowsAsDoubleArray: Array[Array[Double]]):
      (SWIGTYPE_p_void, SWIGTYPE_p_double) = {
    val numCols = rowsAsDoubleArray.head.length
    val data = lightgbmlib.new_doubleArray(numCols * numRows)
    rowsAsDoubleArray.zipWithIndex.foreach(ri =>
      ri._1.zipWithIndex.foreach(value =>
        lightgbmlib.doubleArray_setitem(data, value._2 + (ri._2 * numCols), value._1)))
    (lightgbmlib.double_to_voidp_ptr(data), data)
  }

  def generateDenseDataset(numRows: Int, rowsAsDoubleArray: Array[Array[Double]],
                           referenceDataset: Option[LightGBMDataset],
                           featureNamesOpt: Option[Array[String]]): LightGBMDataset = {
    val numRowsIntPtr = lightgbmlib.new_intp()
    lightgbmlib.intp_assign(numRowsIntPtr, numRows)
    val numRows_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numRowsIntPtr)
    val numCols = rowsAsDoubleArray.head.length
    val isRowMajor = 1
    val numColsIntPtr = lightgbmlib.new_intp()
    lightgbmlib.intp_assign(numColsIntPtr, numCols)
    val numCols_int32_tPtr = lightgbmlib.int_to_int32_t_ptr(numColsIntPtr)
    val datasetOutPtr = lightgbmlib.voidpp_handle()
    val datasetParams = "max_bin=255 is_pre_partition=True"
    val data64bitType = lightgbmlibConstants.C_API_DTYPE_FLOAT64
    var data: Option[(SWIGTYPE_p_void, SWIGTYPE_p_double)] = None
    try {
      data = Some(generateData(numRows, rowsAsDoubleArray))
      // Generate the dataset for features
      LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromMat(
                               data.get._1, data64bitType,
                               numRows_int32_tPtr, numCols_int32_tPtr,
                               isRowMajor, datasetParams, referenceDataset.map(_.dataset).orNull, datasetOutPtr),
                             "Dataset create")
    } finally {
      if (data.isDefined) lightgbmlib.delete_doubleArray(data.get._2)
    }
    val dataset = new LightGBMDataset(lightgbmlib.voidpp_value(datasetOutPtr))
    dataset.setFeatureNames(featureNamesOpt, numCols)
    dataset
  }

  /** Generates a sparse dataset in CSR format.
    * @param sparseRows The rows of sparse vector.
    * @return
    */
  def generateSparseDataset(sparseRows: Array[SparseVector],
                            referenceDataset: Option[LightGBMDataset],
                            featureNamesOpt: Option[Array[String]]): LightGBMDataset = {
    val numCols = sparseRows(0).size

    val datasetOutPtr = lightgbmlib.voidpp_handle()
    val datasetParams = "max_bin=255 is_pre_partition=True"

    // Generate the dataset for features
    LightGBMUtils.validate(lightgbmlib.LGBM_DatasetCreateFromCSRSpark(
                             sparseRows.asInstanceOf[Array[Object]],
                             sparseRows.length,
                             intToPtr(numCols), datasetParams, referenceDataset.map(_.dataset).orNull,
                             datasetOutPtr),
                           "Dataset create")
    val dataset = new LightGBMDataset(lightgbmlib.voidpp_value(datasetOutPtr))
    dataset.setFeatureNames(featureNamesOpt, numCols)
    dataset
  }
}
