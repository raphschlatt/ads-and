from author_name_disambiguation.data.build_blocks import create_block_key


def test_create_block_key_comma_format():
    assert create_block_key("Frenkel, Josif") == "j.frenkel"


def test_create_block_key_plain_format():
    assert create_block_key("Josif Frenkel") == "j.frenkel"


def test_create_block_key_empty():
    assert create_block_key("") == "unknown"


def test_create_block_key_diacritics_are_normalized():
    assert create_block_key("Allègre, C. J.") == "c.allegre"
    assert create_block_key("Müller, A.") == "a.muller"


def test_create_block_key_handles_particles():
    assert create_block_key("van der Waals J") == "j.waals"
    assert create_block_key("von Neumann, John") == "j.neumann"
