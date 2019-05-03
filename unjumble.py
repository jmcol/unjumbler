import json
from itertools import permutations
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import concat_ws
from pyspark.sql.functions import max
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import ShortType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


def load_lookup() -> Tuple[DataFrame, Dict[str, int]]:
    """Load lookup dictionary.

    :return: PySpark dataframe of lookup dictionary.
    """
    f_dictionary = open('freq_dict.json')
    dictionary_data = json.load(f_dictionary)
    lookup = [(w, ''.join(sorted(w)), r) for w, r in dictionary_data.items()]

    lookup_schema = StructType([StructField("word", StringType(), True),
                                StructField('sorted_word', StringType(), True),
                                StructField("rank", IntegerType(), True)])

    return spark.createDataFrame(lookup, lookup_schema), dictionary_data


def enrich_puzzle_data(puzzles: DataFrame) -> DataFrame:
    """Merge puzzle data with corresponding puzzles.

    :param puzzles: PySpark DataFrame containing un-enriched puzzle data.
    :return: PySpark DataFrame containing metadata-enriched puzzles.
    """
    f_enrichment = open('puzzle_meta.json')
    enrichment_data = json.load(f_enrichment)

    enrichment_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                    StructField('final_jumble_groupings', ArrayType(ShortType()), False)])

    enriched_puzzles = spark.createDataFrame(enrichment_data['puzzles'], enrichment_schema)
    return puzzles.join(other=enriched_puzzles, how='left', on='puzzle_no')


def load_puzzles() -> DataFrame:
    """Automated puzzle loading.
    :return: PySpark DataFrame of puzzles and information about them.
    """
    f_puzzle = open('puzzles.json')
    puzzle_data = json.load(f_puzzle)
    sort_word = udf(lambda j: ''.join(sorted(j)))

    puzzle_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                StructField('scrambled_word', StringType(), False),
                                StructField('key_indices', ArrayType(ShortType()), False)])

    puzzles = spark.createDataFrame(puzzle_data['jumbles'], puzzle_schema) \
        .withColumn('sorted_word', sort_word(col('scrambled_word')))

    return enrich_puzzle_data(puzzles)


def unscramble_words(dict_df: DataFrame) -> DataFrame:
    """Perform algorithm to find most likely answer to anagram puzzles.

    :param: PySpark DataFrame of lookup dictionary.
    :return: PySpark DataFrame of solved anagrams.
    """
    pick_chars = udf(lambda j, y: ''.join([j[i] for i in y]))
    puzzle_df = load_puzzles()

    solution_df = puzzle_df\
        .join(other=dict_df, how='left', on='sorted_word')\
        .withColumn('max_rank', max('rank').over(Window.partitionBy('sorted_word')))

    return solution_df\
        .where(solution_df.rank == solution_df.max_rank)\
        .withColumn('special_chars', pick_chars(col('word'), col('key_indices')))


def rank_potential_solutions(potential_solns: List[Any],
                             lookup: Dict[str, int]) -> List[Any]:
    """Function to rank list of potential solutions formed by permutations.

    :param potential_solns: List of words which are potentials solutions to the puzzle.
    :param lookup: Dictionary of words ranked by usage
    :return: List of ranked solutions.
    """
    # TODO: iterate through list of potential solutions
    # Perform summation of word list based on ranking in lookup dictionary
    # Account for zero ranking value
    # Sort by ranking value
    return potential_solns


def solve_jumbler():
    """Driver function to solve jumble puzzle."""
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

    solutions = []
    for grouping, frames in letters:
        potential_solns = []
        for frame in frames:
            if not potential_solns:
                potential_solns = [[list(p)] for p in permutations(grouping, frame) if ''.join(p) in lookup]
            else:
                folded_words = []
                for word_list in potential_solns:
                    used = [item for sublist in word_list for item in sublist]
                    available = [item for item in grouping if item not in used]
                    sub_perms = permutations(available, frame)
                    for sp in sub_perms:
                        if ''.join(sp) in lookup:
                            folded_words.append([*word_list, list(sp)])
                potential_solns = folded_words

        solutions.append(rank_potential_solutions(potential_solns, lookup))

    return solutions

    # TODO: Rank outputs in potential solutions

    # TODO: Output or persist solutions for puzzles

    #           ##### Abandoned methodology #####
    #     perms = [list(p) for p in permutations(grouping)]
    #     parsed_phrases = []
    #     for p in perms:
    #         word_list = []
    #         i = 0
    #         for f in frames:
    #             w_split = p[i:i + f]
    #             word_list.append(w_split)
    #             i += f
    #         parsed_phrases.append(word_list)


def save_output(answers: List[Any]):
    """Saves output of solver to json file.

    :param answers: List of most likely answers.
    """
    with open('answers.json', 'wb') as outfile:
        json.dump(answers, outfile)


if __name__ == '__main__':
    df = solve_jumbler()
