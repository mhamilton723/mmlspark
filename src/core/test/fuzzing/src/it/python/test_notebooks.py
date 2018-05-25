# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import unittest
from LocalNotebookTester import LocalNotebookTester
import os, xmlrunner
import shutil
import subprocess

nb_directory = os.path.join(*([".."] * 7 + ["BuildArtifacts", "notebooks", "local"]))
os.chdir(nb_directory)
nbfiles = [test for test in os.listdir(".") if test.endswith(".ipynb")]

mml_version = os.environ.get("MML_VERSION", subprocess.run(
    ["../../../tools/runme/show-version"], stdout=subprocess.PIPE).stdout.decode('utf-8').rstrip())
# Clear MMLSPark from the ivycache
if mml_version =="0.0":
    print("Clearning mmlspark from the ivycache")
    ivy_dirs = [f[0] for f in os.walk(os.path.join(os.environ["HOME"],".ivy2")) if "mmlspark" in f[0]]
    for dir in ivy_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)

LocalNotebookTester.initialize_tests(nbfiles)

if __name__ == "__main__":
    result = unittest.main(testRunner=xmlrunner.XMLTestRunner(output=os.getenv("TEST_RESULTS","TestResults"),
                                                              outsuffix=None),
                           failfast=True, buffer=False, catchbreak=False)
