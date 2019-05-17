import json
from itertools import permutations
from pathlib import Path
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
    data_folder = Path('resources')
    freq = data_folder / 'freq_dict.json'
    f_dictionary = open(freq)
    dictionary_data = json.load(f_dictionary)
    lookup_list = [(w, ''.join(sorted(w)), r) for w, r in dictionary_data.items()]

    lookup_schema = StructType([StructField("word", StringType(), True),
                                StructField('sorted_word', StringType(), True),
                                StructField("rank", IntegerType(), True)])

    return spark.createDataFrame(lookup_list, lookup_schema), dictionary_data


def enrich_puzzle_data(puzzles: DataFrame) -> DataFrame:
    """Merge puzzle data with corresponding puzzles.

    :param puzzles: PySpark DataFrame containing un-enriched puzzle data.
    :return: PySpark DataFrame containing metadata-enriched puzzles.
    """
    data_folder = Path('resources')
    meta = data_folder / 'puzzle_meta.json'
    f_enrichment = open(meta)
    enrichment_data = json.load(f_enrichment)

    enrichment_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                    StructField('final_jumble_groupings', ArrayType(ShortType()), False)])

    enriched_puzzles = spark.createDataFrame(enrichment_data['puzzles'], enrichment_schema)
    return puzzles.join(other=enriched_puzzles, how='left', on='puzzle_no')


def load_puzzles() -> DataFrame:
    """Automated puzzle loading.
    :return: PySpark DataFrame of puzzles and information about them.
    """
    data_folder = Path('resources')
    puzzle_file = data_folder / 'puzzles.json'
    f_puzzle = open(puzzle_file)
    puzzle_data = json.load(f_puzzle)
    sort_word = udf(lambda j: ''.join(sorted(j)))

    puzzle_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                StructField('scrambled_word', StringType(), False),
                                StructField('key_indices', ArrayType(ShortType()), False)])

    puzzles = spark.createDataFrame(puzzle_data['jumbles'], puzzle_schema) \
        .withColumn('sorted_word', sort_word(col('scrambled_word')))

    return enrich_puzzle_data(puzzles)


def unscramble_words(dictionary: DataFrame) -> DataFrame:
    """Find most likely answer to anagram puzzles.

    :param: PySpark DataFrame of lookup dictionary.
    :return: PySpark DataFrame of solved anagrams.
    """
    pick_chars = udf(lambda j, y: ''.join([j[i] for i in y]))
    puzzle_df = load_puzzles()

    solved = puzzle_df\
        .join(other=dictionary, how='left', on='sorted_word')\
        .withColumn('max_rank', max('rank').over(Window.partitionBy('sorted_word')))

    return solved\
        .where(solved.rank == solved.max_rank)\
        .withColumn('special_chars', pick_chars(col('word'), col('key_indices')))


def rank_potential_solutions(potential_solns: List[Any],
                             lookup_dict: Dict[str, int]) -> List[Any]:
    """Function to rank list of potential solutions formed by permutations.

    :param potential_solns: List of words which are potentials solutions to the puzzle.
    :param lookup_dict: Dictionary of words ranked by usage
    :return: List of ranked solutions.
    """
    # TODO: iterate through list of potential solutions
    # Perform summation of word list based on ranking in lookup dictionary
    for soln_list in potential_solns:
        for possible in soln_list:
            running_score = 0
            for w in possible:
                word = ''.join(w)
                score = lookup_dict[word]
                # Account for zero ranking value
                adjusted_score = 9889 if score == 0 else score
                running_score += adjusted_score

            possible.append(running_score)

    return potential_solns


def top_solutions(scored_potential_solns: List[Any]) -> List[Any]:
    """Select top scored potential solutions for each puzzle.

    :param scored_potential_solns: List of potential solutions with score.
    :return: Top scored potential solution.
    """
    likely = []
    for soln_list in scored_potential_solns:
        sorted_list = sorted(soln_list, key=lambda x: x[-1])
        top_solution_words = sorted_list[0][0:-1]
        joined_solution = list(map(lambda x: ''.join(x), top_solution_words))
        likely.append(joined_solution)

    return likely


def gather_word_permutations(letter_list: List[Any], lookup_dict: Dict[str, int]) -> List[Any]:
    """Function to gather possible solutions by aggregating word permutations.

    :param letter_list: List of letters in word jumble.
    :param lookup_dict: Lookup dictionary of English words.
    :return: List of permutations for a puzzle.
    """
    potentials = []
    for grouping, frames in letter_list:
        potential_solns = []
        for frame in frames:
            if not potential_solns:
                potential_solns = [[list(p)] for p in permutations(grouping, frame) if ''.join(p) in lookup_dict]
            else:
                folded_words = []
                for word_list in potential_solns:
                    used = [item for sublist in word_list for item in sublist]
                    available = [item for item in grouping if item not in used]
                    sub_perms = permutations(available, frame)
                    for sp in sub_perms:
                        if ''.join(sp) in lookup_dict:
                            folded_words.append([*word_list, list(sp)])
                potential_solns = folded_words

        potentials.append(potential_solns)

    return potentials


def parse_letters(anagram_df: DataFrame) -> List[Any]:
    """Parse out usable list from anagram DataFrame.

    :param anagram_df: Spark DataFrame containing solved anagram puzzles.
    :return: List of words from anagram DataFrame.
    """
    aggregated_jumbles = anagram_df \
        .groupBy('puzzle_no', 'final_jumble_groupings') \
        .agg(concat_ws('', collect_list(col('special_chars'))).alias('agg_chars'))

    words_list = aggregated_jumbles \
        .select('agg_chars', 'final_jumble_groupings') \
        .rdd.map(lambda j: list(j)) \
        .collect()

    return words_list


def save_output(answers: List[Any], persist: bool = False):
    """Saves output of solver to json file.

    :param answers: List of most likely answers.
    :param persist: Flag to write output as parquet.
    """
    answer_schema = StructType([StructField('puzzle_no', ShortType(), False),
                                StructField('answer', ArrayType(StringType()), False)])

    formatted_answers = [(i, a) for i, a in enumerate(answers)]
    final_answer = spark.createDataFrame(formatted_answers, answer_schema)

    if persist:
        final_answer.write.parquet('output/answers.parquet')

    return final_answer


if __name__ == '__main__':
    dict_df, lookup = load_lookup()
    solution_df = unscramble_words(dict_df)
    letters = parse_letters(solution_df)

    solutions = gather_word_permutations(letters, lookup)
    ranked_solutions = rank_potential_solutions(solutions, lookup)
    best_solutions = top_solutions(ranked_solutions)

    answers_df = save_output(best_solutions)
    answers_df.show(truncate=False)
