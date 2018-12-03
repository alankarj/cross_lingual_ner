import os
import pickle
from src.util import data_processing


TGT_LANG = "es"


def get_predicted_tags(predict_file_path):
    with open(predict_file_path, encoding="utf-8") as f:
        rows = f.readlines()
        predicted_tags = list()
        for row in rows:
            row = row.strip().split()
            tag_list = list()
            for r in row:
                tag_list.append(r.split("__")[1])
            predicted_tags.append(tag_list)
    return predicted_tags


def data_prep_helper(tgt_lang, data_path, key):
    return get_annotated_list(os.path.join(data_path, tgt_lang, key))


def get_annotated_list(file_path):
    annotated_list_file_path = file_path + ".pkl"
    if os.path.exists(annotated_list_file_path):
        annotated_list = pickle.load(open(annotated_list_file_path, 'rb'))
    else:
        annotated_data = data_processing.AnnotatedData(file_path, lang=None, verbosity=2)
        annotated_list = annotated_data.process_data()
        with open(annotated_list_file_path, 'wb') as f:
            pickle.dump(annotated_list, f)
    return annotated_list


def prepare_final_file(annotated_list, predict_file_path, predicted_tags):
    with open(predict_file_path, 'w', encoding='utf-8') as f:
        for i, annotation in enumerate(annotated_list):
            if len(annotation.tokens) >= 0:
                for j, token in enumerate(annotation.tokens):
                    write_str = token + " " + annotation.ner_tags[j] + " " + predicted_tags[i][j] + "\n"
                    assert len(write_str.split()) == 3
                    f.writelines(write_str)
                f.writelines("\n")


if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data")

    predict_file_path = os.path.join(data_path, "en-" + TGT_LANG, "prediction")
    true_file_path = os.path.join(data_path, "true.txt")

    predicted_tags = get_predicted_tags(os.path.join(predict_file_path, "predicted_idc.txt"))
    print(predicted_tags)
    annotated_list = data_prep_helper(TGT_LANG, data_path, key="testb")
    prepare_final_file(annotated_list, os.path.join(predict_file_path, "error_analysis_idc.txt"), predicted_tags)
