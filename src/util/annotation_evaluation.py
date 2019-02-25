from src import config
from src.util import data_processing
import numpy as np
import copy


def convert_to_numeric(tags):
    numeric_tags = []
    for tag in tags:
        numeric_tags.append(config.char_to_numeric[tag])
    return numeric_tags


def get_gold_annotations(gold_file_path):
    gold_list = []
    with open(gold_file_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for i, row in enumerate(rows):
            if i % 4 == 2 or i % 4 == 3:
                continue
            if i % 4 == 0:
                annotation = data_processing.Annotation()
                annotation.tokens = row.rstrip().split('\t')
            else:
                annotation.ner_tags = row.rstrip().split('\t')
                gold_list.append(annotation)
    return gold_list


def get_fast_align_annotations(fast_align_path, ner_tag_list, tgt_token_list, annotated_list, use_misc=True):
    predicted_tags = [[config.OUTSIDE_TAG for _ in tgt_token] for tgt_token in tgt_token_list]
    match_list = []
    with open(fast_align_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for i, row in enumerate(rows):
            matches = row.rstrip('\n').split(' ')
            # print("Matches: ", matches)
            ner_tags = ner_tag_list[i]
            # print("NER Tags: ", ner_tags)

            match_dict = dict()

            for match in matches:
                match = match.split('-')
                tgt_index = int(match[1])
                src_index = int(match[0])

                if src_index not in match_dict:
                    match_dict[src_index] = list()
                match_dict[src_index].append(tgt_index)

                tag = ner_tags[src_index]
                tag_split = tag.split("-")

                if len(tag_split) > 1:
                    if not use_misc and tag_split[1] == "MISC":
                        predicted_tags[i][tgt_index] = "O"
                else:
                    predicted_tags[i][tgt_index] = tag

            match_list.append(match_dict)
            # print("Tokens: ", tgt_token_list[i])
            # print("Predicted tags: ", predicted_tags[i])

    count_miss = 0
    for i, annotation in enumerate(annotated_list):
        # print("Tokens: ", tgt_token_list[i])
        # print("Old predicted tags: ", predicted_tags[i])
        for span in annotation.span_list:
            tag_type = span.tag_type
            entity_list = []
            for j in range(span.beg, span.end):
                if j not in match_list[i]:
                    count_miss += 1
                else:
                    entity_list += match_list[i][j]
            entity_list = sorted(entity_list)
            # print(entity_list)
            if entity_list != []:
                span_beg, span_end, _ = identify_spans(entity_list)
                for index, j in enumerate(range(span_beg, span_end)):
                    if not use_misc and tag_type == "MISC":
                        predicted_tags[i][j] = "O"
                    else:
                        if index == 0:
                            predicted_tags[i][j] = "B-" + tag_type
                        else:
                            predicted_tags[i][j] = "I-" + tag_type
        # print("New predicted tags: ", predicted_tags[i])
    print("################################################################################")
    print("Count miss: ", count_miss)
    print("################################################################################")
    return predicted_tags


def calculate_f1_score(gold_tag_list, predicted_tag_list):
    den_recall = 0
    den_precision = 0

    num_recall = 0
    num_precision = 0

    for i, gold_tags in enumerate(gold_tag_list):
        predicted_tags = predicted_tag_list[i]
        gold_span_list = data_processing.get_entity_spans(gold_tags, split=False)
        predicted_span_list = data_processing.get_entity_spans(predicted_tags, split=False)
        numeric_gold_tags = convert_to_numeric(gold_tags)
        numeric_predicted_tags = convert_to_numeric(predicted_tags)

        print("Gold tags: ", numeric_gold_tags)
        print("Predicted tags: ", numeric_predicted_tags)

        total_update, predicted_update = count_entities(gold_span_list, numeric_gold_tags,
                                                        numeric_predicted_tags)
        num_recall += predicted_update
        den_recall += total_update

        print("Recall counts: %d, %d" % (total_update, predicted_update))

        total_update, predicted_update = count_entities(predicted_span_list, numeric_gold_tags,
                                                        numeric_predicted_tags)
        num_precision += predicted_update
        den_precision += total_update

        print("Precision counts: %d, %d" % (total_update, predicted_update))

    precision = float(num_precision/den_precision)
    recall = float(num_recall / den_recall)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)


def count_entities(span_list, numeric_gold_tags, numeric_predicted_tags):
    total_entities = 0
    predicted_entities = 0
    if len(span_list) > 0:
        for span in span_list:
            total_entities += 1
            gold_vector = np.array(numeric_gold_tags[span.beg:span.end])
            predicted_vector = np.array(numeric_predicted_tags[span.beg:span.end])
            if np.count_nonzero(gold_vector - predicted_vector) == 0:
                predicted_entities += 1
    return total_entities, predicted_entities


def process_annotations(tag_list):
    new_tag_list = copy.deepcopy(tag_list)
    for i, tags in enumerate(tag_list):
        prev_type = None
        for j, curr_tag in enumerate(tags):
            if curr_tag.startswith("I"):
                curr_type = curr_tag.split("-")[1]
                if curr_type != prev_type:
                    new_tag_list[i][j] = "B-" + curr_type
            elif curr_tag.startswith("B"):
                curr_type = curr_tag.split("-")[1]
            else:
                curr_type = None
            prev_type = curr_type
    return new_tag_list


def post_process_annotations(tgt_annotated_list, stop_word_list, capitalize=False):
    for i, tgt_a in enumerate(tgt_annotated_list):
        prev_type = None
        for j, curr_tag in enumerate(tgt_a.ner_tags):
            if curr_tag.startswith("I"):
                curr_type = curr_tag.split("-")[1]
                if capitalize:
                    if curr_type == "PER":
                        is_per = True
                    else:
                        is_per = False
                    tgt_a.tokens[j] = capitalize_conditionally(tgt_a.tokens[j], stop_word_list, is_per)
                if curr_type != prev_type:
                    tgt_a.ner_tags[j] = "B-" + curr_type
            elif curr_tag.startswith("B"):
                curr_type = curr_tag.split("-")[1]
                if capitalize:
                    if curr_type == "PER":
                        is_per = True
                    else:
                        is_per = False
                    tgt_a.tokens[j] = capitalize_conditionally(tgt_a.tokens[j], stop_word_list, is_per)
            else:
                curr_type = None
            prev_type = curr_type
    return tgt_annotated_list


def capitalize_conditionally(token, stop_word_list, is_per=False):
    if is_per:
        if not token[0].isupper():
            token = token.capitalize()
    elif token not in stop_word_list:
        if not token[0].isupper():
            token = token.capitalize()
    return token


def identify_spans(entity_list):
    span_list = []

    entity_list.append(-1)
    beg = -1
    prev = -1
    for i, x in enumerate(entity_list):
        if x - prev != 1 and beg > -1:
            length = entity_list[i - 1] - beg + 1
            span_list.append((beg, entity_list[i - 1] + 1, length))
            beg = x
        elif beg == -1:
            beg = x
        prev = x

    ordered_span_list = sorted(span_list, key=lambda y: y[2], reverse=True)
    # print(ordered_span_list)
    return ordered_span_list[0]
