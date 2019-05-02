# unjumbler

**Environment:**

- Anaconda Python

- python 3.7

- pyspark 2.4.2

**Setup:**

```
pip install pyspark
set JAVA_HOME=/path/to/java/jdk/
set HADOOP_HOME=/path/to/local/hadoop/
set PYSPARK_PYTHON=/path/to/python/
```

**When running on windows:**

Download [winutils](https://github.com/steveloughran/winutils) from its github repository and extract the version 
corresponding to pyspark 2.4.2 or your previously installed version and extract it to %HADOOP_HOME%\bin

**Execution:**

The unjumbler can be run like any other python program from your favorite IDE or from the shell or command line:

`$> python unjumble.py`