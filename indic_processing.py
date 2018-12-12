import os
from src import config
from collections import Counter
import random
import numpy as np

random.seed(0)

TGT_LANG = "tamil"
TRAIN_FRAC = float(2/3)
DEV_FRAC = float(1/2)
TEST_FRAC = float(1/2)
TAG_MAP = {"PERSON": "PER", "LOCATION": "LOC", "ORGANIZATION": "ORG"}
LANG_CODE_MAP = {"hindi": "hi", "tamil": "ta"}


def prepare_all_files(token_list, tag_list, file_path):
    L = len(token_list)
    train_ids = random.sample(list(range(L)), int(L * TRAIN_FRAC))
    rem_ids = [i for i in range(L) if i not in train_ids]
    dev_ids = random.sample(rem_ids, int(len(rem_ids) * DEV_FRAC))
    test_ids = [i for i in rem_ids if i not in dev_ids]

    print("Train sentences: ", len(train_ids))
    print("Dev sentences: ", len(dev_ids))
    print("Test sentences: ", len(test_ids))

    prepare_file(token_list, tag_list, train_ids, os.path.join(file_path, "train"))
    prepare_file(token_list, tag_list, dev_ids, os.path.join(file_path, "dev"))
    prepare_file(token_list, tag_list, test_ids, os.path.join(file_path, "test"))

    assert len(train_ids) + len(dev_ids) + len(test_ids) == L


def prepare_file(token_list, tag_list, id_list, file_path):
    all_tags = Counter()
    with open(file_path, 'w', encoding='utf-8') as f:
        lengths = list()
        for i, tokens in enumerate(token_list):
            if i in id_list:
                count = 0
                if len(tokens) >= 0:
                    for j, tok in enumerate(tokens):
                        count += len(tok)
                        if tag_list[i][j].startswith("B"):
                            all_tags.update({tag_list[i][j].split("-")[1]: 1})
                        write_str = tok + " " + tag_list[i][j] + "\n"
                        assert len(write_str.split()) == 2
                        f.writelines(write_str)
                    f.writelines("\n")
                lengths.append(count)
        # print(np.average(lengths))
        # print(np.std(lengths))
        # print(np.max(lengths))
        # print(np.min(lengths))
    print("All tags: ", all_tags)


def process_files(folder_path):
    all_files = os.listdir(folder_path)

    agg_sent_count = 0
    agg_token_count = 0
    all_tags = Counter()

    token_list_global = list()
    tag_list_global = list()
    lengths = list()
    count = 0
    for file_name in all_files:
        sent_count = 0
        token_count = 0
        with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
            rows = f.readlines()
            token_list_local = list()
            tag_list_local = list()
            L = 0
            for row in rows:
                row = row.split()
                if row == []:
                    sent_count += 1
                    assert len(token_list_local) == len(tag_list_local)
                    if L <= 500:
                        count += 1
                        token_list_global.append(token_list_local)
                        tag_list_global.append(tag_list_local)
                        lengths.append(L)
                    token_list_local = list()
                    tag_list_local = list()
                    L = 0
                else:
                    L += len(row[0])
                    token_count += 1
                    token_list_local.append(row[0])
                    tag = row[3].split("-")
                    if len(tag) > 1:
                        all_tags.update({tag[1]: 1})
                        if tag[1] in TAG_MAP:
                            tag_list_local.append(tag[0] + "-" + TAG_MAP[tag[1]])
                        else:
                            tag_list_local.append("O")
                    else:
                        all_tags.update({tag[0]: 1})
                        tag_list_local.append("O")
            # print(token_list_local)
            # print(tag_list_local)
            # token_list_global.append(token_list_local)
            # tag_list_global.append(tag_list_local)
            # lengths.append(L)
        print("File name: %s, sentences: %d, tokens: %d" % (file_name, sent_count, token_count))
        agg_sent_count += sent_count
        agg_token_count += token_count
    print("Aggregate number of sentences: %d, tokens: %d" % (agg_sent_count, agg_token_count))
    print("All tags: ", all_tags)
    # assert len(tag_list_global) == agg_sent_count - 16
    # assert len(token_list_global) == agg_sent_count - 16
    # assert len(lengths) == agg_sent_count - 16

    # print(np.average(lengths))
    # print(np.std(lengths))
    # print(np.max(lengths))
    # print(np.min(lengths))
    # count = 0
    # count0 = 0
    # for i, l in enumerate(lengths):
    #     if l == 0:
    #         count0 += 1
    #         print(token_list_global[i])
    #     if l > 500:
    #         count += 1
    # print(count)
    # print(count0)

    # print(lengths)
    return token_list_global, tag_list_global


def read_stopwords(file_path):
    with open(os.path.join(file_path, "tamil_stopwords"), "r", encoding="utf-8") as f:
        rows = f.readlines()
        stopword_list = [row.rstrip("\n") for row in rows]
    return stopword_list


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    file_path = os.path.join(data_path, LANG_CODE_MAP[TGT_LANG])
    token_list, tag_list = process_files(os.path.join(data_path, TGT_LANG + "_train"))
    prepare_all_files(token_list, tag_list, file_path)
    # stopword_list = read_stopwords(file_path)
    # print(stopword_list)
    #