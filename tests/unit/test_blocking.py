from src.data.build_blocks import create_block_key


def test_create_block_key_comma_format():
    assert create_block_key("Frenkel, Josif") == "j.frenkel"


def test_create_block_key_plain_format():
    assert create_block_key("Josif Frenkel") == "j.frenkel"


def test_create_block_key_empty():
    assert create_block_key("") == "unknown"
