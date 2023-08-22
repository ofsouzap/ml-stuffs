from typing import Tuple, Set, DefaultDict, Optional, List
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from progress import Progress
from bij_map import BijMap
from text.text_source import ITextSource
from check import check


def learn_word_rel_pos(text_source: ITextSource,
                       max_look_dist: int,
                       signed: bool = False,
                       word_count_max: Optional[int] = None,
                       progress: Optional[Progress] = None) -> Tuple[npt.NDArray[np.float32], BijMap[str, int]]:
    """Takes a text source and returns a large matrix of the average distances between any two words.

Parameters:

    text_source - the text source to read from

    max_look_dist - a positive integer describing the maximum distance to search from one word to look for nearby words in either direction

    signed - if true, will count words before a word negatively instead of just looking at absolute distances

    word_count_max (optional) - if provided, this positive integer will determine the maximum number of words to include in the output. \
The words kept will be the most common words in the texts

    progress (optional) - optional progress tracker. Will be configured so that it starts at 0 and, when the function completes, it will be finished. \
To acheive this, it may be set to have, for example, half of it dedicated to the first half of the function and half to the other, even if one half is much \
faster than the other.

Returns:

    matrix - an NxN matrix describing the average distance from one word to another. \
The value `matrix[i,j]` gives you the average distance from an instance of word i to an instance of word j

    words - a vector of N strings describing which words correspond to which indexes in the matrix
"""

    if (word_count_max) and (word_count_max < 1):
        raise ValueError(word_count_max)

    if max_look_dist <= 0:
        raise ValueError(max_look_dist)

    discovered_words: Set[str] = set()
    """All the words discovered so far"""

    tot_dists: DefaultDict[Tuple[str, str], int] = defaultdict(lambda: 0)
    """Dictionary mapping pairs of words to the total distance found from the first word of the pair to the second so far"""

    word_occurence_counts: DefaultDict[str, int] = defaultdict(lambda: 0)
    """Dictionary mapping words to the number of occurences of the word"""

    pair_occurence_counts: DefaultDict[Tuple[str, str], int] = defaultdict(lambda: 0)
    """Dictionary mapping pairs of words to the number of times that they've occured near each other"""

    _sections = text_source.read_all_sections()

    if progress:
        progress.max = len(_sections)*2

    for section in _sections:

        for main_index in range(0, len(section)):

            curr_word: str = section[main_index].word
            if curr_word not in discovered_words:
                discovered_words.add(curr_word)
            word_occurence_counts[curr_word] += 1

            for search_index in range(
                max(0, main_index-max_look_dist),
                min(main_index+max_look_dist+1, len(section))
            ):

                if search_index == main_index:
                    continue

                search_word: str = section[search_index].word
                if search_word not in discovered_words:
                    discovered_words.add(search_word)

                pair: Tuple[str, str] = (curr_word, search_word)
                pair_occurence_counts[pair] += 1
                tot_dists[pair] += (search_index-main_index) if signed else abs(search_index-main_index)

        if progress:
            progress.next()

    # Checks for development

    check(all([(a in discovered_words) and (b in discovered_words) for (a,b) in tot_dists.keys()]))

    check(all([(a in discovered_words) and (b in discovered_words) for (a,b) in pair_occurence_counts.keys()]))

    check(all([pair_occurence_counts[(a,b)] == pair_occurence_counts[(b,a)] for (a,b) in pair_occurence_counts.keys()]))

    check(all([pair_occurence_counts[(a,b)] == pair_occurence_counts[(b,a)] for (a,b) in pair_occurence_counts.keys()]))

    # Filter words used by occurence count (if requested)

    considered_words: Set[str]

    if (word_count_max) and (word_count_max < len(discovered_words)):

        considered_words = set(sorted(discovered_words, key=lambda w: word_occurence_counts[w], reverse=True)[:word_count_max])

    else:

        considered_words = discovered_words

    # Construct output arrays

    N = len(considered_words)

    matrix = np.ones(shape=(N,N), dtype=np.float32) * np.inf
    word_indexes = BijMap[str, int]()
    word_indexes_index: int = 0

    if progress:
        progress.max = len(tot_dists)*2
        progress.start()  # Reset
        progress.next(len(tot_dists))  # Move to halfway

    for pair in tot_dists:

        word_1, word_2 = pair

        # Only use words meant to be included
        if (word_1 not in considered_words) or (word_2 not in considered_words):
            continue

        # Add word if they weren't there before

        if word_indexes.to_contains(word_1):
            word_index_1 = word_indexes.get_to(word_1)
        else:
            word_index_1 = word_indexes_index
            word_indexes_index += 1
            word_indexes.set_to(word_1, word_index_1)

        if word_indexes.to_contains(word_2):
            word_index_2 = word_indexes.get_to(word_2)
        else:
            word_index_2 = word_indexes_index
            word_indexes_index += 1
            word_indexes.set_to(word_2, word_index_2)

        # Insert the pair's value

        tot = tot_dists[pair]
        num = pair_occurence_counts[pair]

        avg_dist = tot / num

        matrix[word_index_1,word_index_2] = avg_dist

        # Update progress

        if progress:
            progress.next()

    if progress:
        progress.finish()

    # Return outputs

    return matrix, word_indexes
