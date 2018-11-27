import string
import epitran
from epitran import backoff

DATA_FOLDER = 'data'
SRC_LANGS = ['en']
TGT_LANGS = ['hi', 'es', 'de', 'nl']
TRANSLATE_FNAMES = ['train', 'dev', 'test']
ALIGN_HEURISTICS = ['fast-align', 'affix-match']
ALL_PARTIAL_SUFFIXES = ['partial', 'all']

DOC_BEGIN_STR = '-DOCSTART-'

INDEX_TOKENS = 0
INDEX_POS_TAGS = 1
INDEX_CHUNKS = 2
INDEX_NER_TAGS = 3

SEED = 0

NER_TAGS = ['LOC', 'MISC', 'ORG', 'PER']
TAG_PREFIXES = ['B', 'I']
OUTSIDE_TAG = 'O'

PROCESSED_FILE_SUFFIX = '_processed'

# TGT_LANG_PUNCTUATIONS = string.punctuation + '।' + '&quot;'
# TGT_LANG_PUNCTUATIONS = string.punctuation
TGT_LANG_PUNCTUATIONS = ['।']
PKL_EXT = '.pkl'
TSV_EXT = '.tsv'
QUOTE = '&quot;'

FAST_ALIGN_SEP = ' ||| '
FAST_ALIGN_PATH = '/Users/alankar/Documents/cmu/code/fall_2018/fast_align/build'
FAST_ALIGN_OUTPUT_FNAME = 'forward.align'

BATCH_SIZE = 128

LEXICON_FILE_NAMES = ["lexicon_ground-truth", "lexicon_idc"]
MATCHING_ALGOS = ["token_match", "ipa_match"]


char_to_numeric = {'O': 0, 'LOC': 1, 'MISC': 2, 'ORG': 3, 'PER': 4}

gold_test_dict = dict()
gold_test_dict["en"] = dict()
gold_test_dict["en"] = {
    "documents": 946,
    "sentences": 14041,
    "tokens": 203621,
    "LOC": 7140,
    "MISC": 3438,
    "ORG": 6321,
    "PER": 6600
}

stop_word_dict = {"es": "spanish", "de": "german", "nl": "dutch"}
epi_dict = {"es": "spa-Latn", "de": "deu-Latn", "nl": "nld-Latn"}


def get_allowed_tags():
    allowed_tags = []
    for tp in TAG_PREFIXES:
        for nt in NER_TAGS:
            allowed_tags.append(tp + '-' + nt)
    return allowed_tags


allowed_tags = get_allowed_tags() + [OUTSIDE_TAG]


def get_epi(lang):
    return epitran.Epitran(epi_dict[lang])
