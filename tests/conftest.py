import pytest

from providus_parse.textlayer import Char


@pytest.fixture
def make_chars():
    """Build one Char per letter on one line. A space moves the next letter right and gives no Char."""
    def build(text, bold=False, size=12.0, y=100.0):
        out, x = [], 72.0
        for ch in text:
            if not ch.isspace():
                out.append(Char(ch, x, y - size, x + size / 2, y, y, size, bold, False, False))
            x += size / 2
        return out
    return build
