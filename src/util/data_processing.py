from src import config
import copy
import csv
import os


class AnnotatedData:
    """
    Class for processing raw input data (tokens separated by tags on separate
    lines) into a structured annotated form.
    """
    def __init__(self, file_path, lang=None, verbosity=0):
        self.file_path = file_path
        # Language of the input file.
        self.lang = lang
        self.verbosity = verbosity
        # List of all annotated sentences.
        self.annotated_list = None
        # Documents marked by "-DOCSTART-". This value is calculated and used
        # only for testing, if the lang is set to a non-None value.
        self.num_docs = 0

    def process_data(self):
        with open(self.file_path, mode="r", encoding="utf-8") as f:
            all_rows = f.readlines()
            # List of all annotated sentences in the input corpus.
            annotated_list = list()
            # Each sentence is stored as a list of tuples (that contain token
            # as their first element and the tag as the last).
            sentence = list()

            for i, row in enumerate(all_rows):
                row = row.rstrip("\n").split(" ")
                if row[0].startswith(config.DOC_BEGIN_STR):
                    self.num_docs += 1
                    # No need to parse this line.
                    continue
                if row == [""]:
                    # This indicates the beginning of a new sentence. Hence,
                    # annotate the sentence found so far.
                    a = self._create_annotation(sentence)
                    if a is not None:
                        annotated_list.append(a)
                    sentence = list()
                    continue
                sentence.append(tuple(row))

            a = self._create_annotation(sentence)
            if a is not None:
                annotated_list.append(a)

            if self.verbosity >= 1:
                if self.lang is not None:
                    print("Number of documents: ", self.num_docs)
                print("Number of annotated sentences: ", len(annotated_list))
            self.annotated_list = annotated_list
            if self.lang is not None:
                self._processing_test()
            return annotated_list

    def _create_annotation(self, sentence):
        """
        :param sentence: List of tuples (that contain token as their first
        element and the tag as the last).
        :return: Nothing if the sentence isn't an empty list. Else, return an
        Annotation object corresponding to the sentence.
        """
        sentence = list(map(list, zip(*sentence)))
        if sentence:
            a = Annotation()
            a.tokens = sentence[0]
            a.ner_tags = sentence[-1]
            # Get Span objects for all entities in the sentence.
            a.span_list = _get_entity_spans(a.ner_tags)
            if self.verbosity == 2:
                print("Sentences: ", a.tokens)
                print("NER Tags: ", a.ner_tags)
                tuple_span_list = [(spans.beg, spans.end, spans.tag_type)
                                   for spans in a.span_list]
                print("Span list: ", tuple_span_list)
            return a
        return None

    def _processing_test(self):
        """
        Perform a series of checks to test whether the input file has been
        correctly parsed. Works only for languages for which gold counts exist
        in config.gold_test_dict exists.
        :return: Nothing.
        """
        assert self.lang in config.gold_test_dict
        if self.num_docs > 0:
            assert self.num_docs == \
                   config.gold_test_dict[self.lang]["documents"]
        test_dict = dict()
        test_dict["sentences"] = len(self.annotated_list)
        test_dict["tokens"] = 0
        for annotation in self.annotated_list:
            test_dict["tokens"] += len(annotation.tokens)
            for spans in annotation.span_list:
                if spans.tag_type not in test_dict:
                    test_dict[spans.tag_type] = 0
                test_dict[spans.tag_type] += 1

        for key in test_dict:
            if self.verbosity == 2:
                print("Key: ", key, end='')
                print(", Test count: ", test_dict[key], end='')
                print(", Gold count: ", config.gold_test_dict[self.lang][key])
            assert test_dict[key] == config.gold_test_dict[self.lang][key]


class Annotation:
    """
    Class for an annotated sentence.
    """
    def __init__(self):
        # List of tokens in the sentence.
        self.tokens = list()
        # List of NER tags in the sentence, one corresponding to each token.
        self.ner_tags = list()
        # A list of spans of all the entity phrases in the sentence.
        self.span_list = list()


class Span:
    """
    Class for spans of entity phrases in a sentence.
    """
    def __init__(self, beg, end, tag_type):
        # Begin token index of the entity span
        self.beg = beg
        # End token index of the entity span
        self.end = end
        # Type of tag of the entity span
        self.tag_type = tag_type


def _get_entity_spans(ner_tags):
    """
    :param ner_tags: List of NER Tags in BIO format.
    :return: List of span objects corresponding to distinct entity phrases.
    """
    ner_tags = copy.deepcopy(ner_tags)
    ner_tags.append("O")
    span_list = list()
    beg = 0
    prev_tag_type = None
    for i, curr_tag in enumerate(ner_tags):
        curr_tag_type = curr_tag.split("-")[1] if curr_tag != "O" else None
        if not curr_tag.startswith("I"):
            # This means that the previous span has ended, so we add it.
            span_list = _add_span(span_list, beg, i, prev_tag_type)
            if curr_tag.startswith("B"):
                beg = i
        prev_tag_type = curr_tag_type
    return span_list


def _add_span(span_list, beg, end, tag_type):
    if tag_type is not None:
        span_list.append(Span(beg, end, tag_type))
    return span_list


def write_to_file(sentence_ids, list_annotated_list, file_path):
    prev_length = None
    # All annotated_lists must be of the same length (same number of sentences)
    for annotated_list in list_annotated_list:
        if prev_length is not None:
            assert len(annotated_list) == prev_length
        prev_length = len(annotated_list)

    with open(file_path, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        for i in range(prev_length):
            for annotated_list in list_annotated_list:
                annotation = annotated_list[i]
                write_str = [str(sentence_ids[i])]
                for j, token in enumerate(annotation.tokens):
                    write_str.append(str(token) + " {" + str(annotation.ner_tags[j]) + "}")
                csv_writer.writerow(write_str)


def prepare_train_file(annotated_list, drop_list, file_path, remove_misc=False):
    if drop_list == None:
        drop_list = []

    with open(file_path, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotated_list):
            if i not in drop_list:
                if len(annotation.tokens) >= 0:
                    for j, token in enumerate(annotation.tokens):
                        if remove_misc:
                            if annotation.ner_tags[j].endswith("MISC"):
                                write_str = token + " " + "O" + "\n"
                            else:
                                write_str = token + " " + annotation.ner_tags[j] + "\n"
                        else:
                            write_str = token + " " + annotation.ner_tags[j] + "\n"
                        if len(write_str.split(" ")) < 2:
                            write_str = '-' + write_str
                        assert len(write_str.split()) == 2
                        f.writelines(write_str)
                    f.writelines("\n")


def prepare_train_file_new_format(annotated_list, data_path, key):
    file_path = os.path.join(data_path, key + ".words.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotated_list):
            if len(annotation.tokens) >= 0:
                write_str = get_write_str(annotation.tokens)
                f.writelines(write_str)

    file_path = os.path.join(data_path, key + ".tags.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotated_list):
            if len(annotation.tokens) >= 0:
                write_str = get_write_str(annotation.ner_tags)
                f.writelines(write_str)


def get_write_str(token_list):
    write_str = ""
    for j, token in enumerate(token_list):
        write_str += token + " "
    return write_str.rstrip(" ") + "\n"
