from src import config
import copy
import csv
import os


class AnnotatedData:

    def __init__(self, file_path, lang=None, verbosity=0):
        self.file_path = file_path
        self.lang = lang
        self.verbosity = verbosity
        self.annotated_list = None
        self.num_docs = 0

    def process_data(self):
        with open(self.file_path, mode='r', encoding='utf-8') as f:
            all_rows = f.readlines()
            annotated_list = list()
            sentence = list()

            for i, row in enumerate(all_rows):
                row = row.rstrip("\n").split(" ")
                if row[0].startswith(config.DOC_BEGIN_STR):
                    self.num_docs += 1
                    continue
                if row == ['']:
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
                print("Number of documents: ", self.num_docs)
                print("Number of annotated sentences: ", len(annotated_list))
            self.annotated_list = annotated_list
            if self.lang is not None:
                self._processing_test()
            return annotated_list

    def _create_annotation(self, sentence):
        sentence = list(map(list, zip(*sentence)))
        if sentence != list():
            a = Annotation()
            a.tokens = sentence[0]
            a.ner_tags = sentence[-1]
            a.span_list = get_entity_spans(a.ner_tags)
            if self.verbosity == 2:
                print("Sentences: ", a.tokens)
                print("NER Tags: ", a.ner_tags)
                tuple_span_list = [(spans.beg, spans.end, spans.tag_type)
                                   for spans in a.span_list]
                print("Span list: ", tuple_span_list)
            return a
        return None

    def _processing_test(self):
        if self.num_docs > 0:
            assert self.num_docs == config.gold_test_dict[self.lang]["documents"]
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
                print(", Test number: ", test_dict[key], end='')
                print(", Gold number: ", config.gold_test_dict[self.lang][key])
            assert test_dict[key] == config.gold_test_dict[self.lang][key]


class Annotation:
    def __init__(self):
        self.tokens = list()
        self.ner_tags = list()
        self.span_list = list()


class Span:
    def __init__(self, beg, end, tag_type):
        self.beg = beg
        self.end = end
        self.tag_type = tag_type


def get_entity_spans(ner_tags):
    ner_tags = copy.deepcopy(ner_tags)
    ner_tags.append("O")
    span_list = list()
    beg = 0
    prev_tag_type = None
    for i, curr_tag in enumerate(ner_tags):
        curr_tag_type = curr_tag.split("-")[1] if curr_tag != "O" else None
        if not curr_tag.startswith("I"):
            span_list = add_span(span_list, beg, i, prev_tag_type)
            if curr_tag.startswith("B"):
                beg = i
        prev_tag_type = curr_tag_type
    return span_list


def add_span(span_list, beg, end, tag_type):
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


def prepare_train_file(annotated_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotated_list):
            if len(annotation.tokens) >= 0:
                for j, token in enumerate(annotation.tokens):
                    write_str = token + " " + annotation.ner_tags[j] + "\n"
                    if len(write_str.split(" ")) < 2:
                        write_str = '-' + write_str
                    assert len(write_str.split()) == 2
                    f.writelines(write_str)
                f.writelines("\n")
