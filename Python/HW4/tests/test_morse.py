import runpy
from morse import encode, decode

def test_roundtrip_sos_and_dot():
    assert encode("SOS") == "... --- ..."
    assert decode("... --- ...") == "SOS"
    assert encode(".") == ".-.-.-"
    assert decode(".-.-.-") == "."

def test_encode_decode_roundtrip_various():
    msgs = ["A", "PYTHON", "123"]
    for m in msgs:
        assert decode(encode(m)) == m
