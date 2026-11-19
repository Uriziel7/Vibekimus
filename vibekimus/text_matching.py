"""Finding typed words in the text read from pictures.

Both sides are normalised first: lower case, no diacritics ("Żółw" and "zolw" are
the same), punctuation and runs of spaces as one space. A quoted phrase must appear
as it is (in that order); loose words count one by one."""
import re
import unicodedata

import numpy as np

# Letters Unicode does not split into a base letter and a mark.
SEPARATE_LETTERS = str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ß": "ss", "æ": "ae", "œ": "oe"})
NOT_WORD = re.compile(r"[^\w]+")
MIN_WORD_CHARS = 3   # shorter loose words ("w", "na") match almost everywhere and are ignored


def normalize(text):
    """Lower case, no diacritics, words separated by single spaces, padded with a
    space on both sides (so a phrase is matched as whole words)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold().translate(SEPARATE_LETTERS))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " " + " ".join(NOT_WORD.sub(" ", plain).split()) + " "


def words_of(text):
    """The normalised words of a text worth matching one by one."""
    return [word for word in normalize(text).split() if len(word) >= MIN_WORD_CHARS]


class TextMatch:
    """Typed text against the text read from pictures: phrases that must be there,
    and loose words that raise a file's score."""

    def __init__(self, texts, phrases, words):
        self._texts = texts                                   # TextIndex
        self._phrases = [normalize(phrase) for phrase in phrases if normalize(phrase).strip()]
        self._words = list(dict.fromkeys(word for text in words for word in words_of(text)))

    @property
    def requires_text(self):
        return bool(self._phrases)

    def required_rows(self, index):
        """Rows of an index whose file shows every phrase (None when nothing is required).
        Video frames have no text read, so they never qualify."""
        if not self._phrases:
            return None
        file_of, texts = index.format.file_of, self._texts.normalized_texts()
        return np.fromiter((all(phrase in texts.get(file_of(key), "") for phrase in self._phrases)
                            for key in index.keys), dtype=bool, count=len(index.keys))

    def word_share(self, index):
        """Per row, the share of the loose words found on the picture (0–1), or None
        when there are no loose words."""
        if not self._words:
            return None
        file_of, texts = index.format.file_of, self._texts.normalized_texts()
        found = (sum(f" {word} " in texts.get(file_of(key), "") for word in self._words)
                 for key in index.keys)
        return np.fromiter(found, dtype=np.float32, count=len(index.keys)) / len(self._words)
