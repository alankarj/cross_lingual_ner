import logging
import random
import os
import pickle
import math
import time
import nltk
import html
import copy

from googleapiclient.discovery import build
from src import config
from src.util import data_processing
from src.util.annotation_evaluation import post_process_annotations

import itertools

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.discovery').setLevel(logging.ERROR)
logging.getLogger('cloudpickle').setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG, filename="translation.log")

logger = logging.getLogger()

handler = logging.FileHandler("translation.log")
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

random.seed(config.SEED)
LAST_SUCCESSFUL_BATCH = -1
LAST_SUCCESSFUL_SENT = 12895
THRESHOLD = 0.66
MIN_LEN = 4
MAX_LEN_DIFF_FRAC = float(1/2)

MAX_SET_SIZE = 1000
MAX_BORDER_LEN = 3


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

                beg = i * batch_size
                end = (i + 1) * batch_size

                if self.args.verbosity == 2:
                    print("Batch ID: %d" % i)
                    print("From %d to %d..." % (beg, end))

                src_sentences = src_sentence_list[beg:end]
                tgt_sentence_list.extend(get_google_translations(src_sentences,
                                                                 self.args.src_lang,
                                                                 self.args.tgt_lang,
                                                                 self.args.api_key))
                with open(tgt_sentence_path, 'wb') as f:
                    pickle.dump(tgt_sentence_list, f)
                time.sleep(10)

        if self.args.trans_sent >= 1:
            tgt_sentence_list = pickle.load(open(tgt_sentence_path, 'rb'))

            src_phrase_list = list()
            print("Length of source phrase list: ", len(src_phrase_list))
            tgt_phrase_list = list()
            print("Length of target phrase list: ", len(tgt_phrase_list))

            for i, sid in enumerate(sentence_ids):
                if self.args.verbosity >= 1:
                    print("######################################################################")
                    print("Sentence-%d" % i)

                a = self.src_annotated_list[sid]
                span_list = a.span_list

                tgt_tokens = get_clean_tokens(tgt_sentence_list[i])
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
            assert len(src_phrase) == len(self.tgt_phrase_list[i])

        if self.args.verbosity >= 1:
            print("######################################################################")
            print("Target sentence list loaded. Length: %d" % len(tgt_sentence_list))
            print("Source entity (phrase) list loaded. Length: %d" % len(src_phrase_list))
            print("Target entity (phrase) list loaded. Length: %d" % len(tgt_phrase_list))

    def prepare_mega_tgt_phrase_list(self, calc_count_found=False):
        lexicons = list()
        tgt_phrase_instance_list = list()

        tgt_phrase_instance_list_path = os.path.join(self.base_path, "tgt_phrase_instance_list.pkl")
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
        tgt_sentence_list_path = os.path.join(self.base_path, self.args.translate_fname + "_tgt_annotated_list_" + self.suffix + ".pkl")
        if os.path.exists(tgt_sentence_list_path):
            self.tgt_annotated_list = pickle.load(open(tgt_sentence_list_path, "rb"))

        else:
            for i, sentence in enumerate(self.tgt_sentence_list):
                sentence = sentence.replace("&#39;", "\'")
                sentence = sentence.replace(" &amp; ", "&")
                tgt_annotation = data_processing.Annotation()
                tgt_annotation.tokens = get_clean_tokens(sentence)
                self.tgt_annotated_list.append(tgt_annotation)
            with open(tgt_sentence_list_path, "wb") as f:
                pickle.dump(self.tgt_annotated_list, f)

        path = os.path.join(self.base_path, self.args.translate_fname +
                            "_annotated_list_" + self.suffix + "_partial" + ".pkl")

        if os.path.exists(path):
            self.tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            candidates_list = list()
            # self.sentence_ids = [6594]
            # self.sentence_ids = list(range(150))

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

                        for k, tgt_phrases in enumerate(all_tgt_phrases):
                            candidates_list[i][j].append(list())
                            for l, tgt_phrase in enumerate(tgt_phrases):
                                if l == 100:
                                    break
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
                                            x = _find_match(reference_token, candidate[0],
                                                            transliterate=transliterate)
                                            temp_matches[matching_algo].append(x)

                                    for o, matching_algo in enumerate(config.MATCHING_ALGOS):
                                        temp_match = temp_matches[matching_algo]
                                        temp_match = list(map(list, zip(*temp_match)))
                                        if True in temp_match[1]:
                                            max_score = max(temp_match[0])
                                            if max_score >= THRESHOLD:
                                                max_indices = [ind for ind, score in enumerate(temp_match[0])
                                                               if score == max_score]
                                                for index in max_indices:
                                                    tgt_matches[index].add((j, k, o, max_score, candidate[1]))
                potential_matches.append(tgt_matches)

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

            path = os.path.join(self.base_path, self.args.translate_fname +
                                "_annotated_list_" + self.suffix + "_partial" + ".pkl")

            with open(path, 'wb') as f:
                pickle.dump(self.tgt_annotated_list, f)

        path = os.path.join(self.base_path, self.args.translate_fname +
                            "_annotated_list_" + self.suffix + "_20-NOV-2018" + ".pkl")

        if os.path.exists(path):
            self.tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            print("######################################################################")
            print("######################################################################")
            print("Step-4: Getting final annotations.")
            print("######################################################################")
            print("######################################################################")
            problematic_entities = 0
            src_entities = 0
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

                for j in entity_candidates:
                    src_entities += 1
                    span_list = get_spans(j, entity_candidates[j])

                    if span_list == list():
                        problematic_entities += 1
                        logging.info("######################################################################")
                        logging.info("No correspondence found in sentence: " + str(sid))
                        logging.info("Source tokens: " + str(src_a.tokens))
                        logging.info("Target tokens: " + str(tgt_a.tokens))
                        logging.info("Source phrase for which no match found: " + str(src_phrases[j]))
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
                        for tgt_phrase in temp_tgt_phrases:
                            if len(final_tgt_phrases) > MAX_SET_SIZE:
                                break
                            if tgt_phrase != all_tgt_phrases[0][0]:
                                permutations = itertools.permutations(tgt_phrase.split(" "))
                                final_tgt_phrases.update([" ".join(x) for x in permutations])

                        if len(final_tgt_phrases) > 2 * MAX_SET_SIZE:
                            final_tgt_phrases = copy.deepcopy(temp_tgt_phrases)

                        print("All source candidate phrases: ", len(final_tgt_phrases))

                        best_score = float("inf")
                        best_match = -1
                        global_best_possible_translation = ""
                        for l, span in enumerate(span_list):
                            tgt_phrase = " ".join(tgt_a.tokens[span[0]:span[1]])

                            print("Target candidate phrase: ", tgt_phrase)

                            score = float("inf")
                            best_possible_translation = ""
                            for possible_translation in final_tgt_phrases:
                                edit_dist = nltk.edit_distance(tgt_phrase.lower(), possible_translation.lower())
                                score = min(score, edit_dist)
                                if score == edit_dist:
                                    best_possible_translation = possible_translation

                            print("Score: ", score)

                            best_score = min(best_score, score)
                            if best_score == score:
                                best_match = l
                                global_best_possible_translation = best_possible_translation

                        start, end = span_list[best_match]
                        print("Best match: ", tgt_a.tokens[start:end])
                        print("Best score: ", best_score)
                        print("Best candidate phrase: ", global_best_possible_translation)

                        while len(tgt_a.tokens[start]) <= MAX_BORDER_LEN:
                            altered_phrase = " ".join(tgt_a.tokens[start+1:end])
                            new_edit_distance = nltk.edit_distance(altered_phrase.lower(), global_best_possible_translation.lower())
                            if new_edit_distance < best_score:
                                print(altered_phrase)
                                print("New score: ", new_edit_distance)
                                print("Start dropped.")
                                start += 1
                            else:
                                break

                        while len(tgt_a.tokens[end-1]) <= MAX_BORDER_LEN:
                            altered_phrase = " ".join(tgt_a.tokens[start:end-1])
                            new_edit_distance = nltk.edit_distance(altered_phrase.lower(), global_best_possible_translation.lower())
                            if new_edit_distance < best_score:
                                print(altered_phrase)
                                print("New score: ", new_edit_distance)
                                print("End dropped.")
                                end -= 1
                            else:
                                break

                        tag_type = src_a.span_list[j].tag_type

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
            logging.info("Problematic entities: " + str(problematic_entities))
            logging.info("All source entities: " + str(src_entities))

            path = os.path.join(self.base_path, self.args.translate_fname +
                                "_annotated_list_" + self.suffix + "_20-NOV-2018" + ".pkl")

            with open(path, 'wb') as f:
                pickle.dump(self.tgt_annotated_list, f)

    def prepare_train_file(self):
        self.tgt_annotated_list = post_process_annotations(self.tgt_annotated_list)
        file_path = os.path.join(self.base_path, self.args.translate_fname + "_affix-match_23-OCT-2018")
        data_processing.prepare_train_file(self.tgt_annotated_list, file_path)


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


def get_clean_tokens(sentence, print_info=False):
    for entity in html.entities.html5:
        sentence = sentence.replace("&" + entity + ";", html.entities.html5[entity])

    tokens = nltk.word_tokenize(sentence)
    # if print_info:
    #     print("Sentence: ", sentence)
    #     print("Tokens: ", tokens)

    final_tokens = list()
    ampersand_found = False
    for i, token in enumerate(tokens):
        if ampersand_found:
            ampersand_found = False
            continue

        if token == "&" and i != 0 and i != len(tokens)-1:
            final_tokens.pop()
            final_tokens.append(tokens[i-1] + "&" + tokens[i+1])
            ampersand_found = True

        else:
            if token == "``" or token == "''":
                final_tokens.append("\"")
            else:
                final_tokens.append(token)
    return final_tokens


def _find_match(reference, hypothesis, transliterate=False):
    reference = copy.deepcopy(reference).lower()
    hypothesis = copy.deepcopy(hypothesis).lower()

    # print("Reference: ", reference, "Hypothesis: ", hypothesis)

    if transliterate:
        hypothesis = config.epi.transliterate(hypothesis)
        reference = config.epi.transliterate(reference)

    L = len(hypothesis)
    score = L
    ret_val = False
    if L == 0 or reference == '':
        return 0, False
    if reference == hypothesis:
        # print("Exact match found!")
        # print("Reference: ", reference, "Hypothesis: ", hypothesis)
        ret_val = True
    else:
        for j in range(L, 0, -1):
            sub_str = hypothesis[:j]
            if reference.startswith(sub_str):
                if (L > MIN_LEN) and (len(reference) > MIN_LEN):
                    ret_val = True
                    break
            elif reference.endswith(sub_str):
                if (L > MIN_LEN) and (len(reference) > MIN_LEN):
                    ret_val = True
                    break
            score -= 1
    return float(score / len(reference)), ret_val


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
