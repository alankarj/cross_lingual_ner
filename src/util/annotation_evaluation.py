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


def get_fast_align_annotations(fast_align_path, ner_tag_list, tgt_token_list):
    predicted_tags = [[config.OUTSIDE_TAG for _ in tgt_token] for tgt_token in tgt_token_list]
    with open(fast_align_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for i, row in enumerate(rows):
            matches = row.rstrip('\n').split(' ')
            print("Matches: ", matches)
            ner_tags = ner_tag_list[i][:-1]
            print("NER Tags: ", ner_tags)
            for match in matches:
                match = match.split('-')
                tgt_index = int(match[1])
                src_index = int(match[0])
                tag = ner_tags[src_index]
                # if tag != config.OUTSIDE_TAG:
                #     tag = tag.split('-')[1]
                predicted_tags[i][tgt_index] = tag
            print("Tokens: ", tgt_token_list[i])
            print("Predicted tags: ", predicted_tags[i])
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


def post_process_annotations(tgt_annotated_list):
    for i, tgt_a in enumerate(tgt_annotated_list):
        prev_type = None
        for j, curr_tag in enumerate(tgt_a.ner_tags):
            # if curr_tag != config.OUTSIDE_TAG:
            #     tgt_a.tokens[j] = tgt_a.tokens[j].capitalize()
            if curr_tag.startswith("I"):
                curr_type = curr_tag.split("-")[1]
                if curr_type != prev_type:
                    tgt_a.ner_tags[j] = "B-" + curr_type
            elif curr_tag.startswith("B"):
                curr_type = curr_tag.split("-")[1]
            else:
                curr_type = None
            prev_type = curr_type
    return tgt_annotated_list
