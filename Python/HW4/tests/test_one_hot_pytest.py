import pytest
from one_hot_encoder import fit_transform

@pytest.mark.parametrize(
    "input_value, expected",
    [
        (['a', 'b', 'a'], [('a', [0, 1]), ('b', [1, 0]), ('a', [0, 1])]),
        (['one'], [('one', [1])]),
        (('x', 'y', 'x'), [('x', [0, 1]), ('y', [1, 0]), ('x', [0, 1])]),
        (['Moscow', 'New York', 'Moscow', 'London'],
         [('Moscow', [0, 0, 1]), ('New York', [0, 1, 0]), ('Moscow', [0, 0, 1]), ('London', [1, 0, 0])]),
    ],
)
def test_fit_transform_various(input_value, expected):
    result = fit_transform(input_value) if not isinstance(input_value, tuple) else fit_transform(*input_value)
    assert result == expected

def test_fit_transform_no_args_raises():
    with pytest.raises(TypeError):
        fit_transform()

def test_single_string_arg_is_one_category():
    assert fit_transform("ab") == [("ab", [1])]

def test_iterable_of_chars_is_treated_as_sequence():
    assert fit_transform(list("ab")) == [("a", [0, 1]), ("b", [1, 0])]

def test_vector_length_equals_number_of_unique_categories():
    cats = ["x", "y", "z", "x"]
    res = fit_transform(cats)
    uniq = len(set(cats))
    assert all(len(vec) == uniq for _, vec in res)
    