# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.notebooknode import NotebookNode
from textwrap import dedent
import re, os, subprocess, shutil, sys, time
from nbformat import read as read_nb, NO_CONVERT
from pyspark.sql import SparkSession
import nose

if __name__ == "__main__":
    start= os.getcwd()
    nb_directory = os.path.join(*([".."] * 7 + ["BuildArtifacts", "notebooks", "local"]))
    os.chdir(nb_directory)

#mml_version = os.environ.get("MML_VERSION", subprocess.run(
#    ["../../../tools/runme/show-version"], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip())
# Clear MMLSPark from the ivycache
#if mml_version == "0.0":
#    print("Clearning mmlspark from the ivycache")
#    ivy_dirs = [f[0] for f in os.walk(os.path.join(os.environ["HOME"], ".ivy2")) if "mmlspark" in f[0]]
#    for dir in ivy_dirs:
#        if os.path.exists(dir):
#            shutil.rmtree(dir)

preprocessor = ExecutePreprocessor(timeout=600, enabled=True, allow_errors=False)

class test_notebooks(object):

    _multiprocess_shared_ = True

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
                .getOrCreate()
            globals()["spark"] = spark
            globals()["sc"] = spark.sparkContext
            """))
        epilogue_node = NotebookNode(cell_type="code", source=dedent("""
            try:
                pass
                #spark.stop()
            except:
                pass
            """))
        nb.cells.insert(0, preamble_node)
        nb.cells.append(epilogue_node)
        return nb


    def verify_notebook(self, nbfile):
        """
        verify_notebook: Runs a notebook and ensures that all cells execute without errors.
        """
        # First newline avoids the confusing "F"/"." output of unittest
        print("\nTesting " + nbfile)
        nb = read_nb(nbfile, NO_CONVERT)
        assert(self._check_for_pyspark())
        nb = self.edit_notebook(nb)
        preprocessor.preprocess(nb, {})

    @classmethod
    def add_nb(cls, nbfile):
        def func(self):
            self.verify_notebook(nbfile)

        title = "test_" + re.sub("\\W+", "_", nbfile.split("/")[-1])
        func.__name__ = title
        func.__doc__ = title
        setattr(cls, func.__name__, func )

nbfiles = [test for test in os.listdir(".") if test.endswith(".ipynb")]
for nbfile in nbfiles[0:2]:
    test_notebooks.add_nb(nbfile)

if __name__ == "__main__":
    print(os.getcwd())
    nose.main(defaultTest=__name__, env={"NOSE_PROCESSES":2, 'NOSE_PROCESS_TIMEOUT': 600})