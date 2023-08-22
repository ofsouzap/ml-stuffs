from typing import Dict, List, Tuple, Iterable
import numpy as np
import pytest
from text.learn import learn_word_rel_pos
from text.text_source import RawTextSource


_CASES: List[Tuple[str, Iterable[Tuple[Tuple[str, str], Tuple[float, float]]], int, int]] = [
    (
        "word more more more",
        [
            (("word", "word"), (np.inf, np.inf)),
            (("word", "more"), (2, 2)),
            (("more", "word"), (2, -2)),
            (("more", "more"), (4/3, 0)),
        ],
        5,
        2
    ),
    (
        "word more more more",
        [
            (("word", "word"), (np.inf, np.inf)),
            (("word", "more"), (1, 1)),
            (("more", "word"), (1, -1)),
            (("more", "more"), (1, 0)),
        ],
        1,
        2
    ),
    (
        "word more more more",
        [
            (("more", "more"), (4/3, 0)),
        ],
        5,
        1
    ),
    (
        "a b c d e f",
        [
            (("a","a"), (np.inf,np.inf)),
            (("a","b"), (1,1)),
            (("b","a"), (1,-1)),
            (("a","e"), (4,4)),
            (("e","a"), (4,-4)),
            (("b","d"), (2,2)),
            (("d","b"), (2,-2)),
        ],
        5,
        6
    ),
    (
        # From Flanders and Swann, Ill Wind
        "i once had a whim and i had to obey it to buy a french horn in a second hand shop",
        [
            (("i","i"), (6,0)),
            (("i","a"), (54/6,48/6)),
            (("french","whim"), (10,-10)),
        ],
        25,
        50
    ),
    (
        # From Flanders and Swann, Ill Wind
        "i once had a whim and i had to obey it to buy a french horn in a second hand shop",
        [
            (("i","i"), (6,0)),
            (("i","a"), (54/6,48/6)),
        ],
        25,
        4
    ),
    # TODO - more test cases
]
"""Each value has the input text and a dictionary. The dictonary values are `(unsigned_expected_value, signed_expected_value)`"""


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_unsigned_symmetry(in_text: str, tests: Dict[Tuple[str, str], float], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=False, word_count_max=word_count_max)

    for i in range(mat.shape[0]):
        for j in range(i+1):
            if i == j:
                assert np.isclose(mat[i,j], mat[j,i])


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_signed_antisymmetry(in_text: str, tests: Dict[Tuple[str, str], float], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=True, word_count_max=word_count_max)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                assert np.isclose(mat[i,j], -mat[j,i])


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_unsigned_output_shape(in_text: str, tests: Dict[Tuple[str, str], float], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=False, word_count_max=word_count_max)

    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] <= word_count_max


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_signed_output_shape(in_text: str, tests: Dict[Tuple[str, str], float], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=True, word_count_max=word_count_max)

    assert mat.shape[0] == mat.shape[1]
    assert mat.shape[0] <= word_count_max


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_values_unsigned(in_text: str, tests: Dict[Tuple[str, str], Tuple[float, float]], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, word_indexes = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=False, word_count_max=word_count_max)

    for pair, exp_vals in tests:

        i1 = word_indexes.get_to(pair[0])
        i2 = word_indexes.get_to(pair[1])

        assert np.isclose(mat[i1,i2], exp_vals[0]), f"Failed with {pair[0]}-{pair[1]}. Expected {exp_vals[0]}, got {mat[i1,i2]}"


@pytest.mark.parametrize(("in_text", "tests", "max_look_dist", "word_count_max"), _CASES)
def test_values_signed(in_text: str, tests: Dict[Tuple[str, str], Tuple[float, float]], max_look_dist: int, word_count_max: int):

    source = RawTextSource(in_text)

    mat, word_indexes = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=True, word_count_max=word_count_max)

    for pair, exp_vals in tests:

        i1 = word_indexes.get_to(pair[0])
        i2 = word_indexes.get_to(pair[1])

        assert np.isclose(mat[i1,i2], exp_vals[1]), f"Failed with {pair[0]}-{pair[1]}. Expected {exp_vals[1]}, got {mat[i1,i2]}"
