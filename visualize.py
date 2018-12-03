import os
from src import config
from src.util import data_processing
import pickle
import random
random.seed(0)


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)

    src_file_path = os.path.join(data_path, "en")
    annotated_list_file_path = os.path.join(src_file_path, "train_processed.pkl")
    annotated_list_1 = pickle.load(open(annotated_list_file_path, 'rb'))

    src_file_path = os.path.join(data_path, "en-es")
    annotated_list_file_path = os.path.join(src_file_path, "train_annotated_list_all_complete_new.pkl")
    annotated_list_2 = pickle.load(open(annotated_list_file_path, 'rb'))

    src_file_path = os.path.join(data_path, "en-es")
    annotated_list_file_path = os.path.join(src_file_path, "train_annotated_list_all_complete_old.pkl")
    annotated_list_3 = pickle.load(open(annotated_list_file_path, 'rb'))

    src_file_path = os.path.join(data_path, "en-es")
    annotated_list_file_path = os.path.join(src_file_path, "train_idc_processed.pkl")
    annotated_list_4 = pickle.load(open(annotated_list_file_path, 'rb'))

    to_be_removed = []
    for i, annotation in enumerate(annotated_list_1):
        if len(annotation.tokens) <= 1:
            to_be_removed.append(i)

    # print(annotated_list_3[13446].tokens)
    # print(annotated_list_3[13446].ner_tags)

    annotated_list_1 = [a for i, a in enumerate(annotated_list_1) if i not in to_be_removed]
    annotated_list_2 = [a for i, a in enumerate(annotated_list_2) if i not in to_be_removed]
    annotated_list_3 = [a for i, a in enumerate(annotated_list_3) if i not in to_be_removed]

    sentence_ids = random.sample(list(range(len(annotated_list_3))), 1000)

    annotated_list_1 = [annotated_list_1[i] for i in sentence_ids]
    annotated_list_2 = [annotated_list_2[i] for i in sentence_ids]
    annotated_list_3 = [annotated_list_3[i] for i in sentence_ids]
    annotated_list_4 = [annotated_list_4[i] for i in sentence_ids]

    for i, sid in enumerate(sentence_ids):
        list_a = annotated_list_2[i].ner_tags
        list_b = annotated_list_3[i].ner_tags
        difference_found = 0
        for j, a in enumerate(list_a):
            if a != list_b[j]:
                difference_found = 1
        if difference_found == 1:
            print(sid)

    #
    # write_file_path = os.path.join(src_file_path, "comparisons_new.tsv")
    # data_processing.write_to_file(sentence_ids, [annotated_list_1, annotated_list_2, annotated_list_3, annotated_list_4],
    #                               write_file_path)