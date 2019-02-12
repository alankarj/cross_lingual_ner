import logging
import random
import os
import pickle
import math
import time
import nltk
import html
import copy
import itertools
import string
import epitran
from nltk.parse.corenlp import CoreNLPParser
import copy
from googleapiclient.discovery import build
from src import config
from src.util import data_processing
from src.util.annotation_evaluation import post_process_annotations
from datetime import datetime
from collections import Counter
from nltk.corpus import stopwords
import re

random.seed(config.SEED)

MATCHING_SCORE_THRESHOLD = 0.66
MAX_SET_SIZE = 1000
MOST_COMMON = 4
DISTRIBUTION_OCCURRENCE_THRESHOLD = 2
ABLATION_STRING = "original"

DATE_TODAY = datetime.today().strftime("%d-%m-%Y")
# DATE_TODAY = "28-11-2018"
print("Today's date: ", DATE_TODAY)

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.discovery').setLevel(logging.ERROR)
logging.getLogger('cloudpickle').setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class Translation:
    def __init__(self, annotated_list, args):
        self.src_annotated_list = annotated_list
        self.args = args
        self.base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
        self.sentence_ids = None
        self.tgt_sentence_list = None
        self.tgt_phrase_list = None
        self.src_phrase_list = None
        self.tgt_annotated_list = list()
        self.tgt_phrase_instance_list = list()
        self.suffix = None
        self.drop_list = None
        self.tgt_lang = args.tgt_lang
        self.epi = config.get_epi(self.tgt_lang)
        # self.src_epi = epitran.Epitran("eng-Latn")
        if self.tgt_lang == "zh" or self.tgt_lang == "ar":
            self.use_corenlp = True
        else:
            self.use_corenlp = False

        if self.tgt_lang in config.stop_word_dict:
            self.stop_word_list = set(stopwords.words(config.stop_word_dict[self.tgt_lang]))
        elif self.tgt_lang == "hi":
            self.stop_word_list = config.stop_word_list_hi
        elif self.tgt_lang == "ta":
            self.stop_word_list = config.stop_word_list_ta
        elif self.tgt_lang == "zh":
            self.stop_word_list = config.stop_word_list_zh
        else:
            raise ValueError("Language %s stop word list not available." % self.tgt_lang)

        handler = logging.FileHandler("translation_" + "en-" + self.tgt_lang + "_"
                                      + datetime.today().strftime("%d-%m-%Y") + "_" +
                                      str(MATCHING_SCORE_THRESHOLD) + "_" + str(MOST_COMMON) +
                                      ABLATION_STRING + ".log")
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    def translate_data(self):
        num_sentences = len(self.src_annotated_list)

        if self.args.translate_all == 0:
            self.suffix = str(self.args.num_sample)
        else:
            self.suffix = "all"

        if self.args.translate_all == 0:
            if self.args.num_sample > num_sentences:
                raise ValueError("Cannot sample more than the number of sentences.")
            sentence_ids_file_name = self.args.sentence_ids_file + "_" + self.suffix + ".pkl"
            sentence_ids_path = os.path.join(self.base_path, sentence_ids_file_name)
            if os.path.exists(sentence_ids_path):
                sentence_ids = pickle.load(open(sentence_ids_path, 'rb'))
            else:
                sentence_ids = random.sample(list(range(num_sentences)), self.args.num_sample)
                with open(sentence_ids_path, 'wb') as f:
                    pickle.dump(sentence_ids, f)
        else:
            sentence_ids = list(range(num_sentences))

        src_sentence_list = [' '.join(self.src_annotated_list[sid].tokens) for sid in sentence_ids]

        if self.args.verbosity == 2:
            print("Source sentence list loaded. Length: %d" % len(src_sentence_list))

        # File names to either read from (if trans_sent is 0) or write to (if trans_sent is 1)
        tgt_sentence_path = os.path.join(self.base_path, self.args.translate_fname +
                                         '_' + self.suffix + config.PKL_EXT)
        tgt_phrase_path = os.path.join(self.base_path, self.args.translate_fname +
                                       '_tgt_phrases_' + self.suffix + config.PKL_EXT)
        src_phrase_path = os.path.join(self.base_path, self.args.translate_fname +
                                       '_src_phrases_' + self.suffix + config.PKL_EXT)

        if self.args.trans_sent == 2:
            # Get sentence list in target language
            batch_size = config.BATCH_SIZE
            tgt_sentence_list = _get_saved_list(tgt_sentence_path)
            print(len(tgt_sentence_list))

            max_iter = int(math.ceil(num_sentences/batch_size))
            for i in range(max_iter):

                # if i < 107:
                #     continue

                beg = i * batch_size
                end = (i + 1) * batch_size

                if self.args.verbosity == 2:
                    print("Batch ID: %d" % i)
                    print("From %d to %d..." % (beg, end))

                if self.args.tgt_lang == "zh":
                    tgt_lang = "zh-CN"
                else:
                    tgt_lang = self.args.tgt_lang

                src_sentences = src_sentence_list[beg:end]
                tgt_sentence_list.extend(get_google_translations(src_sentences,
                                                                 self.args.src_lang,
                                                                 tgt_lang,
                                                                 self.args.api_key))
                with open(tgt_sentence_path, 'wb') as f:
                    pickle.dump(tgt_sentence_list, f)
                time.sleep(10)

        if self.args.trans_sent >= 1:
            tgt_sentence_list = pickle.load(open(tgt_sentence_path, 'rb'))


            # src_phrase_list = pickle.load(open(src_phrase_path, 'rb'))
            # tgt_phrase_list = pickle.load(open(tgt_phrase_path, 'rb'))
            # print("Length of source phrase list: ", len(src_phrase_list))
            # print("Length of target phrase list: ", len(tgt_phrase_list))

            src_phrase_list = list()
            print("Length of source phrase list: ", len(src_phrase_list))
            tgt_phrase_list = list()
            print("Length of target phrase list: ", len(tgt_phrase_list))

            for i, sid in enumerate(sentence_ids):
                # if i < 11743:
                #     continue

                if self.args.verbosity >= 1:
                    print("######################################################################")
                    print("Sentence-%d" % i)

                a = self.src_annotated_list[sid]
                span_list = a.span_list

                tgt_tokens = get_clean_tokens(tgt_sentence_list[i], self.use_corenlp)
                if self.args.verbosity == 2:
                    print("Source tokens: ", a.tokens)
                    print("Tgt tokens: ", tgt_tokens)

                src_phrase_list.append(list())
                for span in span_list:
                    phrase = ' '.join(a.tokens[span.beg:span.end])
                    src_phrase_list[i].append(phrase)
                    if self.args.verbosity == 2:
                        print("Beginning: %d, End: %d, Tag: %s" % (span.beg, span.end,
                                                                   span.tag_type), end='')
                        print(", Phrase: ", phrase)

                if self.args.verbosity == 2:
                    print("Source phrase list: ", src_phrase_list[i])

                if src_phrase_list[i] != list():
                    tgt_phrase_list.append(get_google_translations(src_phrase_list[i], self.args.src_lang, self.args.tgt_lang, self.args.api_key))
                else:
                    tgt_phrase_list.append(list())

                with open(tgt_phrase_path, 'wb') as f:
                    pickle.dump(tgt_phrase_list, f)
                with open(src_phrase_path, 'wb') as f:
                    pickle.dump(src_phrase_list, f)

                    if self.args.verbosity == 2:
                        print("Target phrase list: ", tgt_phrase_list[i])

            if self.args.verbosity >= 1:
                print("######################################################################")
                print("Source entities (phrases) translated.")

            with open(tgt_phrase_path, 'wb') as f:
                pickle.dump(tgt_phrase_list, f)
            with open(src_phrase_path, 'wb') as f:
                pickle.dump(src_phrase_list, f)

        else:
            tgt_sentence_list = pickle.load(open(tgt_sentence_path, 'rb'))
            tgt_phrase_list = pickle.load(open(tgt_phrase_path, 'rb'))
            src_phrase_list = pickle.load(open(src_phrase_path, 'rb'))

        self.sentence_ids = sentence_ids
        self.tgt_sentence_list = tgt_sentence_list
        self.tgt_phrase_list = tgt_phrase_list
        self.src_phrase_list = src_phrase_list

        for i, src_phrase in enumerate(self.src_phrase_list):
            assert len(self.src_phrase_list[i]) == len(self.tgt_phrase_list[i])


        # for i, src_a in enumerate(self.src_annotated_list):
        #     if src_a.tokens[0] == "Abu":
        #         print(i)
        #         print(src_a.tokens)

        if self.args.verbosity >= 1:
            print("######################################################################")
            print("Target sentence list loaded. Length: %d" % len(tgt_sentence_list))
            print("Source entity (phrase) list loaded. Length: %d" % len(src_phrase_list))
            print("Target entity (phrase) list loaded. Length: %d" % len(tgt_phrase_list))

    def prepare_mega_tgt_phrase_list(self, calc_count_found=False):
        lexicons = list()
        tgt_phrase_instance_list = list()

        tgt_phrase_instance_list_path = os.path.join(self.base_path, self.args.translate_fname
        + "_tgt_phrase_instance_list.pkl")
        if os.path.exists(tgt_phrase_instance_list_path):
            self.tgt_phrase_instance_list = pickle.load(open(tgt_phrase_instance_list_path, 'rb'))

        else:
            for lexicon_file_name in config.LEXICON_FILE_NAMES:
                lexicon = pickle.load(open(os.path.join(self.base_path, lexicon_file_name + ".pkl"), "rb"))
                lexicons.append(lexicon)

            for i, src_phrase_sent in enumerate(self.src_phrase_list):
                tgt_phrase_instance_list.append(list())
                for j, src_phrase in enumerate(src_phrase_sent):
                    tgt_phrase_instance = TgtPhrases(src_phrase)
                    tgt_phrase_instance.add_tgt_phrase([self.tgt_phrase_list[i][j].replace("&#39;", "\'")])
                    tgt_phrase_instance.add_tgt_phrase([src_phrase])
                    tgt_phrase_instance_list[i].append(tgt_phrase_instance)

                for k, lexicon in enumerate(lexicons):
                    for l, src_phrase in enumerate(src_phrase_sent):
                        temp = dict()
                        tgt_phrase = list()

                        src_phrase = src_phrase.split(" ")
                        for m, src_token in enumerate(src_phrase):
                            temp[m] = list()
                            if src_token.lower() in lexicon:
                                for tgt_token in lexicon[src_token.lower()]:
                                    temp[m].append(tgt_token.capitalize())
                            else:
                                temp[m].append(src_token)

                        for key in temp:
                            if tgt_phrase == list():
                                tgt_phrase = temp[key]
                            else:
                                tgt_phrase = [phrase_1 + " " + phrase_2 for phrase_1 in tgt_phrase
                                              for phrase_2 in temp[key]]

                        tgt_phrase_instance_list[i][l].add_tgt_phrase(tgt_phrase)

            with open(tgt_phrase_instance_list_path, "wb") as f:
                pickle.dump(tgt_phrase_instance_list, f)

            self.tgt_phrase_instance_list = tgt_phrase_instance_list

            if calc_count_found:
                total_tokens = list()
                translations_found = list()
                translation_tokens = list()
                all_tokens = list()

                for _ in config.LEXICON_FILE_NAMES:
                    total_tokens.append(0)
                    translations_found.append(0)
                    translation_tokens.append(list())

                for i, src_phrase_sent in enumerate(self.src_phrase_list):
                    for k, lexicon in enumerate(lexicons):
                        for l, src_phrase in enumerate(src_phrase_sent):
                            src_phrase = src_phrase.split(" ")
                            for m, src_token in enumerate(src_phrase):
                                total_tokens[k] += 1
                                all_tokens.append(src_token)
                                if src_token.lower() in lexicon:
                                    translation_tokens[k].append(src_token)
                                    translations_found[k] += 1

                for i, _ in enumerate(lexicons):
                    print("Unique tokens in IDC: ", len(set(translation_tokens[1])))
                    print("Unique tokens in ground truth: ", len(set(translation_tokens[0])))
                    print("Unique tokens in source entities: ", len(set(all_tokens)))
                    print("In IDC but not in ground truth: ", len(set(translation_tokens[1]).difference(set(translation_tokens[0]))))
                    print("In ground truth but not in IDC: ", len(set(translation_tokens[0]).difference(set(translation_tokens[1]))))
                    print("Total tokens: ", total_tokens[i])
                    print("Translations found: ", translations_found[i])
                    print("Fraction of translations found: %.4f" % float(translations_found[i] / total_tokens[i]))

        self._test_tgt_phrase_instance_list()

    def _test_tgt_phrase_instance_list(self):
        for tgt_phrase_instances in self.tgt_phrase_instance_list:
            for tpinstance in tgt_phrase_instances:
                assert len(tpinstance.tgt_phrase_list) == len(config.LEXICON_FILE_NAMES) + 2

    def get_tgt_annotations_new(self):
        # Target language tokens (obtained through Google Translate + cleaned)
        path = os.path.join(self.base_path, self.args.translate_fname + "_tgt_annotated_list_period_" + self.suffix + ".pkl")
        if os.path.exists(path):
            self.tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            for i, sentence in enumerate(self.tgt_sentence_list):
                print("######################################################################")
                print("Sentence-%d: %s" % (i, sentence))
                sentence = sentence.replace("&#39;", "\'")
                sentence = sentence.replace(" &amp; ", "&")
                tgt_annotation = data_processing.Annotation()
                tgt_annotation.tokens = get_clean_tokens(sentence, self.use_corenlp)

                if self.tgt_lang == "hi":
                    for j, token in enumerate(tgt_annotation.tokens):
                        if token.endswith("।"):
                            tgt_annotation.tokens[j] = token.rstrip("।")
                            if tgt_annotation.tokens[j] == "":
                                tgt_annotation.tokens[j] = "."
                            else:
                                tgt_annotation.tokens.insert(j + 1, ".")

                print("Tokens: ", tgt_annotation.tokens)
                self.tgt_annotated_list.append(tgt_annotation)
            with open(path, "wb") as f:
                pickle.dump(self.tgt_annotated_list, f)

        # if ABLATION_STRING not in ["_no_idc_", "_no_gold_", "_no_copy_",
        #                            "_no_phonetic_", "_no_google_", "_no_google_v2_"]:
        #     temp_suffix = "1_period_"
        # else:

        temp_suffix = ABLATION_STRING

        path = os.path.join(self.base_path, self.args.translate_fname +
                            "_annotated_list_" + str(MATCHING_SCORE_THRESHOLD) + temp_suffix + self.suffix + DATE_TODAY + "_partial" + ".pkl")

        if os.path.exists(path):
            self.tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            candidates_list = self.get_candidates_list()
            potential_matches = self.get_potential_matches(candidates_list)
            self.get_partial_tags(potential_matches)
            with open(path, 'wb') as f:
                pickle.dump(self.tgt_annotated_list, f)

        temp_suffix = ABLATION_STRING
        path = os.path.join(self.base_path, self.args.translate_fname +
                            "_annotated_list_" + str(MATCHING_SCORE_THRESHOLD) + temp_suffix + self.suffix + DATE_TODAY + ".pkl")

        if os.path.exists(path):
            self.tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            print("Final tags unavailable. Getting them...")
            self.get_final_tags()
            with open(path, 'wb') as f:
                pickle.dump(self.tgt_annotated_list, f)

        # path = os.path.join(self.base_path, "problematic_entities_" + DATE_TODAY + ".pkl")
        # problematic_entities = pickle.load(open(path, "rb"))
        #
        # path = os.path.join(self.base_path, "problem_children_" + DATE_TODAY + ".pkl")
        # problem_children = pickle.load(open(path, "rb"))
        #
        # path = os.path.join(self.base_path, "problematic_sentences_" + DATE_TODAY + ".pkl")
        # problematic_sentences = pickle.load(open(path, "rb"))
        #
        # final_candidate_dict = self.get_refined_candidates(problem_children, problematic_entities)
        # checklist = [[False for _ in problematic_sentences[sent]] for sent in problematic_sentences]
        # num_annotation = 0
        # for index in range(MOST_COMMON):
        #     num_annotation += self.tag_problematic_sentences(problematic_sentences, final_candidate_dict,
        #                                    problematic_entities, index, checklist)
        # logging.info("Total number of annotations: " + str(num_annotation))
        #
        # for i, sid in enumerate(self.sentence_ids):
        #     tgt_a = self.tgt_annotated_list[sid]
        #     for ind, tag in enumerate(tgt_a.ner_tags):
        #         if tag not in config.allowed_tags:
        #             tgt_a.ner_tags[ind] = config.OUTSIDE_TAG
        #
        # path = os.path.join(self.base_path, self.args.translate_fname +
        #                     "_annotated_list_" + str(
        #     MATCHING_SCORE_THRESHOLD) + "2_period_" + self.suffix + DATE_TODAY + ".pkl")
        # with open(path, 'wb') as f:
        #     pickle.dump(self.tgt_annotated_list, f)

    def get_candidates_list(self):
        candidates_list = list()
        # self.sentence_ids = [153]
        # self.sentence_ids = list(range(100))

        print("######################################################################")
        print("######################################################################")
        print("Step-1: Getting candidate tags.")
        print("######################################################################")
        print("######################################################################")
        for i, sid in enumerate(self.sentence_ids):
            src_a = self.src_annotated_list[sid]
            src_phrases = self.src_phrase_list[sid]

            print("######################################################################")
            print("Sentence ID: ", sid)
            print("######################################################################")

            candidates_list.append(list())
            if src_phrases != list():
                for j, src_phrase in enumerate(src_phrases):
                    print("######################################################################")
                    print("Source phrase: ", src_phrase)
                    print("######################################################################")

                    candidates_list[i].append(list())

                    all_tgt_phrases = self.tgt_phrase_instance_list[sid][j].tgt_phrase_list
                    if ABLATION_STRING == "_no_idc_":
                        all_tgt_phrases = all_tgt_phrases[:-1]
                    elif ABLATION_STRING == "_no_gold_":
                        all_tgt_phrases = all_tgt_phrases[:-2]
                    elif ABLATION_STRING == "_no_copy_" or ABLATION_STRING == "_no_phonetic_":
                        all_tgt_phrases = all_tgt_phrases[:-3]
                    elif ABLATION_STRING == "_no_google_" or ABLATION_STRING == "_no_google_v2_":
                        all_tgt_phrases = all_tgt_phrases[1:]

                    for k, tgt_phrases in enumerate(all_tgt_phrases):
                        candidates_list[i][j].append(list())
                        for l, tgt_phrase in enumerate(tgt_phrases):
                            if l == 100:
                                break
                            print(src_a.tokens)

                            for span in src_a.span_list:
                                print(span.beg, span.end)

                            print(len(src_a.span_list))
                            print("Index: ", j)
                            tag_type = src_a.span_list[j].tag_type
                            tgt_tokens = tgt_phrase.split(" ")

                            candidates_list[i][j][k].append(list())
                            for m, tgt_token in enumerate(tgt_tokens):
                                if m == 0:
                                    ner_tag = "B-" + tag_type
                                else:
                                    ner_tag = "I-" + tag_type
                                print("Target token: ", tgt_token)
                                candidates_list[i][j][k][l].append((tgt_token, ner_tag))
        return candidates_list

    def get_potential_matches(self, candidates_list):
        print("######################################################################")
        print("######################################################################")
        print("Step-2: Getting all possible potential matches.")
        print("######################################################################")
        print("######################################################################")
        potential_matches = list()
        for i, sid in enumerate(self.sentence_ids):
            tgt_matches = dict()
            sent_candidates = candidates_list[i]
            if sent_candidates == list():
                print("%d: " % i, sent_candidates)
            else:
                for j, phrase_candidates in enumerate(sent_candidates):
                    for k, diff_phrase_candidates in enumerate(phrase_candidates):
                        for l, all_candidates in enumerate(diff_phrase_candidates):
                            for m, candidate in enumerate(all_candidates):
                                print("%d.%d.%d.%d.%d Candidate token: %s" % (i, j, k, l, m, candidate[0]), end='')
                                print(", Candidate tag: ", candidate[1])

                                tgt_a = self.tgt_annotated_list[sid]

                                temp_matches = dict()
                                for o, matching_algo in enumerate(config.MATCHING_ALGOS):
                                    temp_matches[matching_algo] = list()

                                for n, reference_token in enumerate(tgt_a.tokens):
                                    if n not in tgt_matches:
                                        tgt_matches[n] = set()
                                    for o, matching_algo in enumerate(config.MATCHING_ALGOS):

                                        if o == 0:
                                            transliterate = False
                                        else:
                                            transliterate = True

                                        if ABLATION_STRING == "_no_phonetic_" and o == 1:
                                            transliterate = False

                                        # if k == 1:
                                        #     epi = self.src_epi
                                        # else:
                                        #     epi = self.epi

                                        epi = self.epi

                                        x = _find_match(reference_token, candidate[0], epi,
                                                        self.stop_word_list, transliterate=transliterate)
                                        temp_matches[matching_algo].append(x)

                                for o, matching_algo in enumerate(config.MATCHING_ALGOS):
                                    temp_match = temp_matches[matching_algo]
                                    temp_match = list(map(list, zip(*temp_match)))
                                    if True in temp_match[1]:
                                        max_score = max(temp_match[0])
                                        max_indices = [ind for ind, score in enumerate(temp_match[0])
                                                       if score == max_score]
                                        for index in max_indices:
                                            if max_score >= MATCHING_SCORE_THRESHOLD:
                                                tgt_matches[index].add((j, k, o, max_score, candidate[1]))

            potential_matches.append(tgt_matches)
        return potential_matches

    def get_partial_tags(self, potential_matches):
        print("######################################################################")
        print("######################################################################")
        print("Step-3: Getting a minimal set of potential matches.")
        print("######################################################################")
        print("######################################################################")
        for i, sid in enumerate(self.sentence_ids):
            potential_match_sent = potential_matches[i]
            tgt_a = self.tgt_annotated_list[sid]
            print("######################################################################")
            print("Sentence ID: ", sid)
            print("######################################################################")

            if potential_match_sent == dict():
                tgt_a.ner_tags = [config.OUTSIDE_TAG for _ in tgt_a.tokens]

            for key, val in potential_match_sent.items():
                print("Key: ", tgt_a.tokens[key], " Value: ", val)
                val = list(val)
                all_vals = list(map(list, zip(*val)))
                if all_vals == list():
                    tgt_a.ner_tags.append(config.OUTSIDE_TAG)
                else:
                    val.sort(key=lambda a: (-a[3], a[1], a[2]))

                    phrases = set(all_vals[0])
                    tags = set(all_vals[-1])

                    new_vals = list()
                    phrase_added = list()
                    tag_added = list()
                    ind_added = list()

                    for t, v in enumerate(val):
                        for phrase in set(phrases):
                            if (phrase not in phrase_added) and (v[0] == phrase):
                                new_vals.append(v)
                                phrase_added.append(phrase)
                                ind_added.append(t)

                        for tag in set(tags):
                            if (tag not in tag_added) and (t not in ind_added) and (v[-1] == tag):
                                new_vals.append(v)
                                tag_added.append(tag)
                                ind_added.append(t)

                    tgt_a.ner_tags.append(new_vals)
                    print("Final tag: ", new_vals)

    def get_final_tags(self):
        print("######################################################################")
        print("######################################################################")
        print("Step-4: Getting final annotations.")
        print("######################################################################")
        print("######################################################################")
        problematic_entities = Counter()
        src_entities = 0
        occur_problem_entities = 0
        problematic_sentences = dict()
        problem_children = dict()
        idf = dict()

        for i, sid in enumerate(self.sentence_ids):
            print("######################################################################")
            print("Sentence ID: ", sid)
            print("######################################################################")
            tgt_a = self.tgt_annotated_list[sid]
            src_a = self.src_annotated_list[sid]
            src_phrases = self.src_phrase_list[sid]
            ner_tag_list = tgt_a.ner_tags

            dummy_list = [-1 for _ in ner_tag_list]

            entity_candidates = {j: copy.deepcopy(dummy_list) for j in range(len(src_phrases))}
            find_entity_spans(entity_candidates, ner_tag_list)

            score_match = dict()
            for j in entity_candidates:
                src_entities += 1
                span_list = get_spans(j, entity_candidates[j])

                tok_set = set([tok.lower() for tok in tgt_a.tokens])
                for tok in tok_set:
                    if tok not in idf:
                        idf[tok] = 0
                    idf[tok] += 1

                if span_list == list():
                    occur_problem_entities += 1
                    if i not in problematic_sentences:
                        problematic_sentences[i] = list()
                    problematic_sentences[i].append((j, src_phrases[j]))

                    problematic_entities.update({src_phrases[j]: 1})
                    logging.info("######################################################################")
                    logging.info("No correspondence found in sentence: " + str(sid))
                    logging.info("Source tokens: " + str(src_a.tokens))
                    logging.info("Target tokens: " + str(tgt_a.tokens))
                    logging.info("Source phrase for which no match found: " + str(src_phrases[j]))

                    if src_phrases[j] not in problem_children:
                        problem_children[src_phrases[j]] = Counter()

                    for tok in tgt_a.tokens:
                        problem_children[src_phrases[j]].update({tok: 1})

                    logging.info("######################################################################")

                else:
                    print("######################################################################")
                    print("Source phrase: ", src_phrases[j])
                    print("######################################################################")
                    temp_tgt_phrases = set()
                    all_tgt_phrases = self.tgt_phrase_instance_list[sid][j].tgt_phrase_list
                    for k, candidate_phrases in enumerate(all_tgt_phrases):
                        temp_tgt_phrases.update(candidate_phrases)

                    final_tgt_phrases = copy.deepcopy(temp_tgt_phrases)

                    if ABLATION_STRING != "_no_google_v2_":
                        for tgt_phrase in temp_tgt_phrases:
                            if len(final_tgt_phrases) > MAX_SET_SIZE:
                                break
                            if tgt_phrase != all_tgt_phrases[0][0]:
                                permutations = itertools.permutations(tgt_phrase.split(" "))
                                final_tgt_phrases.update([" ".join(x) for x in permutations])

                        if len(final_tgt_phrases) > 2 * MAX_SET_SIZE:
                            final_tgt_phrases = copy.deepcopy(temp_tgt_phrases)

                    print("All source candidate phrases: ", len(final_tgt_phrases))

                    new_span_list = list()
                    for l, span in enumerate(span_list):
                        start, end = span
                        orig_start = start
                        orig_end = end

                        start_list = [orig_start]
                        while tgt_a.tokens[start] in self.stop_word_list:
                            start += 1
                            if start == orig_end:
                                break
                            start_list.append(start)

                        end_list = [orig_end]
                        while tgt_a.tokens[end-1] in self.stop_word_list:
                            end -= 1
                            if end == orig_start:
                                break
                            end_list.append(end)

                        for s in start_list:
                            for e in end_list:
                                new_span_list.append((s, e))

                    best_score = float("inf")

                    for l, span in enumerate(new_span_list):
                        tgt_phrase = " ".join(tgt_a.tokens[span[0]:span[1]])

                        print("Target candidate phrase: ", tgt_phrase)

                        score = float("inf")
                        best_possible_translation = ""

                        if len(final_tgt_phrases) < 100:
                            print("Final target phrases: ", final_tgt_phrases)
                        else:
                            print("Size of target phrase set more than 100.")

                        for possible_translation in final_tgt_phrases:
                            edit_dist = nltk.edit_distance(tgt_phrase.lower(), possible_translation.lower())
                            score = min(score, edit_dist)
                            if score == edit_dist:
                                best_possible_translation = possible_translation

                        print("Score: ", score)

                        best_score = min(best_score, score)
                        if best_score == score:
                            if span not in score_match:
                                score_match[span] = dict()
                            score_match[span][j] = (best_score, best_possible_translation)

                    for span in score_match:
                        if j in score_match[span]:
                            if score_match[span][j][0] > best_score:
                                score_match[span].pop(j)

            print("Final score match: ", score_match)
            for span in score_match:
                if score_match[span] != dict():
                    # max_score = max([v[0] for v in score_match[span].values()])
                    max_score = min([v[0] for v in score_match[span].values()])
                    max_indices = [i for i in score_match[span].keys()
                                   if score_match[span][i][0] == max_score]

                    # Pick the first one.
                    tag_type = src_a.span_list[max_indices[0]].tag_type
                    start, end = span
                    best_span = list(range(start, end))
                    for ind, sp in enumerate(best_span):
                        if ind == 0:
                            prefix = "B"
                        else:
                            prefix = "I"
                        tgt_a.ner_tags[sp] = prefix + "-" + tag_type

            print("######################################################################")
            print("Target tokens: ", tgt_a.tokens)
            for ind, tag in enumerate(tgt_a.ner_tags):
                if tag not in config.allowed_tags:
                    tgt_a.ner_tags[ind] = config.OUTSIDE_TAG
            print("Target NER tags: ", tgt_a.ner_tags)

        logging.info("######################################################################")
        logging.info("Entities with no match: " + str(problematic_entities))
        logging.info("Number of unique entities with no match: " + str(len(problematic_entities)))
        logging.info("Number of occurrences of entities with no match: " + str(occur_problem_entities))
        logging.info("Number of all source entities: " + str(src_entities))

        if ABLATION_STRING == "original" or ABLATION_STRING == "_no_google_" or ABLATION_STRING == "_no_google_v2_":
            N = len(self.sentence_ids)
            logging.info("Total documents: " + str(N))
            logging.info("Potential candidates for problematic entities:")
            count_pc = 0
            for pc in problem_children:
                if problematic_entities[pc] > DISTRIBUTION_OCCURRENCE_THRESHOLD:
                    count_pc += problematic_entities[pc]
                if problematic_entities[pc] > 0:
                    for k, v in problem_children[pc].items():
                        assert k.lower() in idf
                        problem_children[pc][k] = v * math.log(float(N / idf[k.lower()]))
                    logging.info("######################################################################")
                    logging.info(pc + ": " + str(problem_children[pc].most_common(100)))

            sentences_to_drop = list()
            for k, val in problematic_sentences.items():
                to_be_dropped = False
                for v in val:
                    if problematic_entities[v[1]] > 0:
                        to_be_dropped = True
                if to_be_dropped:
                    sentences_to_drop.append(k)

            logging.info("Number of problematic entities %d" % len(problematic_entities))
            logging.info("Number of sentences with at least one problematic entity: %d" %
                         len(sentences_to_drop))

            path = os.path.join(self.base_path, "problematic_entities_" + DATE_TODAY + ".pkl")
            with open(path, 'wb') as f:
                pickle.dump(problematic_entities, f)

            path = os.path.join(self.base_path, "problem_children_" + DATE_TODAY + ".pkl")
            with open(path, 'wb') as f:
                pickle.dump(problem_children, f)

            path = os.path.join(self.base_path, "problematic_sentences_" + DATE_TODAY + ".pkl")
            with open(path, 'wb') as f:
                pickle.dump(problematic_sentences, f)

            final_candidate_dict = self.get_refined_candidates(problem_children, problematic_entities)
            checklist = [[False for _ in problematic_sentences[sent]] for sent in problematic_sentences]
            num_annotation = 0
            num_annotation += self.tag_problematic_sentences(problematic_sentences, final_candidate_dict,
                                           problematic_entities, checklist)
            logging.info("Number of occurrences of problematic entities with count > %d (PC-%d): %d" %
                         (DISTRIBUTION_OCCURRENCE_THRESHOLD, DISTRIBUTION_OCCURRENCE_THRESHOLD,
                          count_pc))
            logging.info("Total number of annotations: " + str(num_annotation))

    def get_refined_candidates(self, problem_children, problematic_entities):
        final_candidate_dict = dict()
        for token in problem_children:
            if problematic_entities[token] > DISTRIBUTION_OCCURRENCE_THRESHOLD:
                most_common_candidates = problem_children[token].most_common(10 * MOST_COMMON)

                new_most_common_candidates = list()

                for i, candidates in enumerate(most_common_candidates):
                    if candidates[0] not in self.stop_word_list and candidates[0] not in string.punctuation\
                            and len(re.findall("\d+", candidates[0])) == 0:
                        new_most_common_candidates.append(candidates)

                final_candidates = new_most_common_candidates[:MOST_COMMON]

                final_candidate_dict[token] = final_candidates
                logging.info("######################################################################")
                logging.info("Token: " + str(token) + " Final candidates: " + str(final_candidates))
        logging.info("######################################################################")
        logging.info("Final candidates dict: " + str(final_candidate_dict))
        return final_candidate_dict

    def tag_problematic_sentences(self, problematic_sentences, final_candidate_dict, problematic_entities, checklist):
        total_entities = 0
        total_annotations_done = 0

        logging.info("Tagging problematic sentences.")
        for i_sent, sent in enumerate(problematic_sentences):
            logging.info("######################################################################")
            logging.info("Sentence id: " + str(sent))
            src_a = self.src_annotated_list[sent]
            tgt_a = self.tgt_annotated_list[sent]
            logging.info("Source tokens: " + str(src_a.tokens))
            logging.info("Target tokens: " + str(tgt_a.tokens))

            for i_ent, entities in enumerate(problematic_sentences[sent]):
                if not checklist[i_sent][i_ent]:
                    if problematic_entities[entities[1]] > DISTRIBUTION_OCCURRENCE_THRESHOLD:
                        total_entities += 1
                        tag_type = src_a.span_list[entities[0]].tag_type
                        entity_str = entities[1]
                        candidate_list = final_candidate_dict[entity_str]

                        token_list = {k: v for k, v in candidate_list}
                        temp_ner_list = [(-1, 0) for _ in tgt_a.tokens]

                        for i, tgt_token in enumerate(tgt_a.tokens):
                            if tgt_token in token_list:
                                temp_ner_list[i] = (tag_type, token_list[tgt_token])

                        logging.info("######################################################################")
                        logging.info("Entity: " + str(entity_str))
                        ordered_span_list = get_ordered_spans(tag_type, temp_ner_list)
                        logging.info("Ordered span list: " + str(ordered_span_list))

                        for span in ordered_span_list:
                            if self.can_tag(span[0], span[1], tgt_a):
                                checklist[i_sent][i_ent] = True
                                logging.info("Tagging possible!")
                                logging.info("Target phrase: " + " ".join(tgt_a.tokens[span[0]:span[1]]))
                                logging.info("Tag: " + tag_type)
                                total_annotations_done += 1
                                best_span = list(range(span[0], span[1]))
                                for ind, sp in enumerate(best_span):
                                    if ind == 0:
                                        prefix = "B"
                                    else:
                                        prefix = "I"
                                    tgt_a.ner_tags[sp] = prefix + "-" + tag_type
                                break

        logging.info("######################################################################")
        logging.info("Number of remaining problematic entities: " + str(total_entities))
        logging.info("Number of annotated entities: " + str(total_annotations_done))
        return total_annotations_done

    def can_tag(self, span_beg, span_end, tgt_a):
        for i in range(span_beg, span_end):
            if tgt_a.ner_tags[i] != "O":
                if type(tgt_a.ner_tags[i]) != list:
                    return False
        return True

    def prepare_train_file(self):
        if self.tgt_lang in ["es", "nl"]:
            capitalize = True
        else:
            capitalize = False
        self.tgt_annotated_list = post_process_annotations(self.tgt_annotated_list, self.stop_word_list,
                                                           capitalize=capitalize)
        file_path = os.path.join(self.base_path, self.args.translate_fname + "_affix-match_blah_" +
                                 str(MATCHING_SCORE_THRESHOLD) + ABLATION_STRING + datetime.today().strftime("%d-%m-%Y"))
        if self.tgt_lang in ["hi", "ta"]:
            remove_misc = True
        else:
            remove_misc = False
        data_processing.prepare_train_file(self.tgt_annotated_list, self.drop_list, file_path,
                                           remove_misc=remove_misc)


class TgtPhrases:
    def __init__(self, src_phrase):
        self.src_phrase = src_phrase
        self.tgt_phrase_list = list()

    def add_tgt_phrase(self, tgt_phrase):
        self.tgt_phrase_list.append(tgt_phrase)


class Match:
    def __init__(self, token, ner_tag):
        self.token = token
        self.ner_tag = ner_tag
        self.match_list = list()

    def add_match(self, ref_id, score):
        self.match_list.append((ref_id, score))


def get_google_translations(src_sentence_list, src_lang, tgt_lang, api_key):
    service = build('translate', 'v2', developerKey=api_key)
    tgt_dict = service.translations().list(source=src_lang, target=tgt_lang, q=src_sentence_list).execute()
    tgt_sentence_list = [t['translatedText'] for t in tgt_dict['translations']]
    return tgt_sentence_list

def tokenize_using_corenlp(text):
    corenlp_parser = CoreNLPParser('http://localhost:9001', encoding='utf8')
    result = corenlp_parser.api_call(text, {'annotators': 'tokenize,ssplit'})
    tokens = [token['originalText'] or token['word'] for sentence in result['sentences'] for token in
              sentence['tokens']]
    return tokens

def get_clean_tokens(sentence, use_corenlp=True):
    for entity in html.entities.html5:
        sentence = sentence.replace("&" + entity + ";", html.entities.html5[entity])

    if use_corenlp:
        tokens = tokenize_using_corenlp(sentence)
    else:
        tokens = nltk.word_tokenize(sentence)

    final_tokens = list()
    ampersand_found = False
    prior_period_index = -float("inf")

    for i, token in enumerate(tokens):
        if ampersand_found:
            ampersand_found = False
            continue

        elif token == "&" and i != 0 and i != len(tokens)-1:
            final_tokens.pop()
            final_tokens.append(tokens[i-1] + "&" + tokens[i+1])
            ampersand_found = True

        elif token == "." and i != 0 and i != len(tokens) - 1:
            if prior_period_index == -float("inf") or i - prior_period_index > 2:
                prior_period_index = i
                final_tokens.pop()
                final_tokens.append(tokens[i - 1] + ".")
            else:
                final_tokens.pop()
                final_tokens.pop()
                final_tokens.append(tokens[i - 3] + "." + tokens[i - 1] + ".")

        else:
            if token == "``" or token == "''":
                final_tokens.append("\"")
            else:
                final_tokens.append(token)
    return final_tokens


def _find_match(reference, hypothesis, epi, stop_word_list, transliterate=False):
    reference = copy.deepcopy(reference).lower()
    hypothesis = copy.deepcopy(hypothesis).lower()

    reference_set = set(reference.split("-"))
    reference_set.add(reference)
    hypothesis_set = set(hypothesis.split("-"))
    hypothesis_set.add(hypothesis)

    if len(reference_set) > 1 or len(hypothesis_set) > 1:
        list_of_all_scores = list()
        for r in reference_set:
            for h in hypothesis_set:
                list_of_all_scores.append(_find_match_helper(r, h, epi, stop_word_list,
                                                             transliterate=transliterate))
        list_of_all_scores = sorted(list_of_all_scores, key=lambda x: x[0], reverse=True)
        return list_of_all_scores[0]

    else:
        return _find_match_helper(reference, hypothesis, epi, stop_word_list, transliterate=transliterate)


def _find_match_helper(reference, hypothesis, epi, stop_word_list, transliterate=False):
    is_stop_word = False
    if (hypothesis in stop_word_list) or (reference in stop_word_list):
        is_stop_word = True

    if transliterate:
        hypothesis = epi.transliterate(hypothesis)
        reference = epi.transliterate(reference)

    L = len(hypothesis)
    score = L
    ret_val = False
    if L == 0 or reference == '':
        return 0, False
    if reference == hypothesis:
        ret_val = True
    else:
        for j in range(L, 0, -1):
            sub_str = hypothesis[:j]
            if reference.startswith(sub_str):
                if not is_stop_word:
                    ret_val = True
                    break
            elif reference.endswith(sub_str):
                if not is_stop_word:
                    ret_val = True
                    break
            score -= 1

    return min(float(score / len(reference)), float(score / L)), ret_val


def generate_fast_align_data(src_sent_list, tgt_sent_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, src_sent in enumerate(src_sent_list):
            f.writelines(src_sent + config.FAST_ALIGN_SEP + tgt_sent_list[i] + '\n')


def get_readable_annotations(tokens, annotations):
    # Annotation is a tuple containing score, index, tag
    flat = list(map(list, zip(*annotations)))
    if flat == []:
        full_annotations = [config.OUTSIDE_TAG for _ in tokens]

    else:
        full_annotations = []
        for i, token in enumerate(tokens):
            if i in flat[1]:
                full_annotations.append(flat[2][flat[1].index(i)])
            else:
                full_annotations.append(config.OUTSIDE_TAG)
    return full_annotations


def read_lexicon(src_file_path, tgt_file_path):
    with open(src_file_path, "r", encoding="utf-8") as f:
        rows = f.readlines()
        lexicon = dict()
        for row in rows:
            row = row.rstrip("\n").split(" ")
            print(row)
            src_word = row[0]
            tgt_word = row[1]
            if src_word not in lexicon:
                lexicon[src_word] = list()
            lexicon[src_word].append(tgt_word)

    with open(tgt_file_path, 'wb') as f:
        pickle.dump(lexicon, f)


def _get_saved_list(path):
    if os.path.exists(path):
        saved_list = pickle.load(open(path, 'rb'))
    else:
        saved_list = list()
    return saved_list


def _get_tag_type(tag):
    if tag == config.OUTSIDE_TAG:
        tag_type = None
    else:
        tag_type = tag.split("-")[1]
    return tag_type


def find_entity_spans(entity_candidates, ner_tag_list):
    for i, ner_tags in enumerate(ner_tag_list):
        all_ids = list(map(list, zip(*ner_tags)))

        if all_ids[0] != ["O"]:
            for j in all_ids[0]:
                entity_candidates[j][i] = j


def get_spans(tag_id, tag_list):
    prev_tag = -1
    span_list = list()
    tag_list.append(-1)

    beg = 0
    for i, curr_tag in enumerate(tag_list):
        if (prev_tag == -1) and (curr_tag == tag_id):
            beg = i
        elif (prev_tag == tag_id) and (curr_tag == -1):
            span_list.append((beg, i))
        prev_tag = curr_tag

    return span_list


def get_ordered_spans(tag_id, tag_score_list):
    prev_tag = (-1, 0.0)
    span_list = list()
    tag_score_list.append((-1, 0.0))

    beg = 0
    score = 0
    for i, curr_tag in enumerate(tag_score_list):
        if (prev_tag[0] == -1) and (curr_tag[0] == tag_id):
            beg = i
        elif (prev_tag[0] != -1) and (curr_tag[0] == -1):
            span_list.append((beg, i, score))
            score = 0
        score += curr_tag[1]
        prev_tag = curr_tag

    span_list = sorted(span_list, key=lambda x: x[2], reverse=True)
    return span_list

