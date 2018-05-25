# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.notebooknode import NotebookNode
from textwrap import dedent
import re, os, subprocess
from nbformat import read as read_nb, NO_CONVERT


class LocalNotebookTester(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ExecutePreprocessor(timeout=600, enabled=True, allow_errors=False)

    def _check_for_pyspark(self):
        """
        _in_pyspark: Returns true if this test is run in a context that has access to PySpark
        """
        try:
            from pyspark.sql import SparkSession
            return True
        except ImportError:
            return False

    def edit_notebook(self, nb):
        """
        Inject the code needed to setup and shutdown spark and sc magic variables.
        """
        mml_version     = os.environ.get("MML_VERSION", subprocess.run(
            ["../../../tools/runme/show-version"], stdout=subprocess.PIPE)
                                         .stdout.decode('utf-8').rstrip())
        scala_version   = os.environ["SCALA_VERSION"]
        package = "com.microsoft.ml.spark:mmlspark_{}:{}".format(scala_version, mml_version)
        repo = "file:" + os.path.abspath("../../packages/m2")
        preamble_node = NotebookNode(cell_type="code", source=dedent("""
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \\
                .master("local[*]") \\
                .appName("NotebookTestSuite") \\
                .config("spark.jars.repositories", "{}") \\
                .config("spark.jars.packages", "{}") \\
                .getOrCreate()
            globals()["spark"] = spark
            globals()["sc"] = spark.sparkContext
            """.format(repo, package)))
        epilogue_node = NotebookNode(cell_type="code", source=dedent("""
            try:
                spark.stop()
            except:
                pass
            """))
        nb.cells.insert(0, preamble_node)
        nb.cells.append(epilogue_node)
        return nb

    @classmethod
    def initialize_tests(cls, nbfiles):
        for nbfile in nbfiles:
            test_name = "test_" + re.sub("\\W+", "_", nbfile.split("/")[-1])
            def make_test(nbfile):
                return lambda instance: instance.verify_notebook(nbfile)
            setattr(cls, test_name, make_test(nbfile))

    def verify_notebook(self, nbfile):
        """
        verify_notebook: Runs a notebook and ensures that all cells execute without errors.
        """
        try:
            # First newline avoids the confusing "F"/"." output of unittest
            print("\nTesting " + nbfile)
            nb = read_nb(nbfile, NO_CONVERT)
            assert  self._check_for_pyspark()
            nb = self.edit_notebook(nb)
            self.preprocessor.preprocess(nb, {})
        except Exception as err:
            self.fail(err)
