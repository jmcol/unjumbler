import json
from itertools import permutations

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql.functions import col
from pyspark.sql.functions import max
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import concat_ws
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import ArrayType
from pyspark.sql.types import ShortType
from pyspark.sql.types import StructType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


def load_lookup():
    f_dictionary = open('freq_dict.json')
    dictionary_data = json.load(f_dictionary)
    lookup = [(w, ''.join(sorted(w)), r) for w, r in dictionary_data.items()]

    lookup_schema = StructType([StructField("word", StringType(), True),
                                StructField('sorted_word', StringType(), True),
                                StructField("rank", IntegerType(), True)])

    return spark.createDataFrame(lookup, lookup_schema), lookup


def enrich_puzzle_data(puzzles):
    f_enrichment = open('puzzle_meta.json')
    enrichment_data = json.load(f_enrichment)

    enrichment_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                    StructField('final_jumble_groupings', ArrayType(ShortType()), False)])

    enriched_puzzles = spark.createDataFrame(enrichment_data['puzzles'], enrichment_schema)
    return puzzles.join(other=enriched_puzzles, how='left', on='puzzle_no')


def load_puzzles():
    f_puzzle = open('puzzles.json')
    puzzle_data = json.load(f_puzzle)
    sort_word = udf(lambda j: ''.join(sorted(j)))

    puzzle_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                StructField('scrambled_word', StringType(), False),
                                StructField('key_indices', ArrayType(ShortType()), False)])

    puzzles = spark.createDataFrame(puzzle_data['jumbles'], puzzle_schema) \
        .withColumn('sorted_word', sort_word(col('scrambled_word')))

    return enrich_puzzle_data(puzzles)


def unscramble_words(dict_df):
    pick_chars = udf(lambda j, y: ''.join([j[i] for i in y]))
    puzzle_df = load_puzzles()

    solution_df = puzzle_df\
        .join(other=dict_df, how='left', on='sorted_word')\
        .withColumn('max_rank', max('rank').over(Window.partitionBy('sorted_word')))

    return solution_df\
        .where(solution_df.rank == solution_df.max_rank)\
        .withColumn('special_chars', pick_chars(col('word'), col('key_indices')))


def solve_jumbler():
    dict_df, lookup = load_lookup()
    solution_df = unscramble_words(dict_df)
    
    aggregated_jumbles = solution_df\
        .groupBy('puzzle_no', 'final_jumble_groupings')\
        .agg(concat_ws('', collect_list(col('special_chars'))).alias('agg_chars'))

    # TODO: Add final solving algorithm.

    # get permutations of final collection of letters
    letters = aggregated_jumbles\
        .select('agg_chars', 'final_jumble_groupings')\
        .rdd.map(lambda j: list(j))\
        .collect()

    for grouping, frames in letters:
        perms = [list(p) for p in permutations(grouping)]
        parsed_phrases = []
        for p in perms:
            word_list = []
            i = 0
            for f in frames:
                w_split = p[i:i + f]
                word_list.append(w_split)
                i += f
            parsed_phrases.append(word_list)

    return parsed_phrases

    # frame each into work lengths

    # filter out all that contain words not in dictionary

    # aggregate & rank


if __name__ == '__main__':
    df = solve_jumbler()
