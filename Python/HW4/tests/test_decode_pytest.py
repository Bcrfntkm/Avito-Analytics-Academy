import pytest
from morse import decode


@pytest.mark.parametrize(
    "morse, expected",
    [
        ("... --- ...", "SOS"),
        (".- -... -.-.", "ABC"),
        (".... . .-.. .-.. ---", "HELLO"),
    ],
)
def test_decode_parametrized(morse, expected):
    assert decode(morse) == expected
