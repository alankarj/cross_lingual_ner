import io
import os
from src import config
from argparse import ArgumentParser
from src.util import data_processing

from flair.data import Sentence
from flair.models import SequenceTagger
import pickle
import time
import random

random.seed(0)

MIN_TOKENS = 10
MINIBATCH_SIZE = 512
CONF_THRESHOLD = 0.9
NUM_SAMPLES = 28082
MAX_LIM = 1000000
MAX_SAMPLES = NUM_SAMPLES * 2


def get_ner_tags(tokens, flair_entities):
    ner_tags = ["O" for _ in tokens]
    entities_tagged = 0
    if flair_entities != []:
        search_index = 0
        entity_tokens, entity_type = get_entity(flair_entities, search_index)
        i = 0
        while i < len(tokens):
            if tokens[i] == entity_tokens[0]:
                ner_tags[i] = "B-" + entity_type
                entities_tagged += 1
                for j in range(1, len(entity_tokens), 1):
                    ner_tags[i+j] = "I-" + entity_type
                search_index += 1
                if search_index == len(flair_entities):
                    break
                entity_tokens, entity_type = get_entity(flair_entities, search_index)
            i += 1
    assert len(ner_tags) == len(tokens)
    assert entities_tagged == len(flair_entities)
    return ner_tags


def get_entity(flair_entities, search_index):
    entity = flair_entities[search_index]
    entity_tokens = entity['text'].split(" ")
    entity_type = entity['type']
    return entity_tokens, entity_type


def get_avg_confidence(flair_entities):
    num = len(flair_entities)
    total_conf = sum(e['confidence'] for e in flair_entities)
    if num > 0:
        return total_conf/num
    else:
        return 0.0


def read_tgt_file(fname, indices):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    count = 0
    tgt_annotated_list = list()
    for i, line in enumerate(fin):
        if i > MAX_LIM:
            break
        if count < len(indices) and i == indices[count]:
            tokens = line.rstrip().split(" ")
            tgt_a = data_processing.Annotation()
            tgt_a.tokens = tokens
            # print("Tokens: ", tokens)
            tgt_annotated_list.append(tgt_a)
            count += 1
    assert len(tgt_annotated_list) == len(indices)
    return tgt_annotated_list


def predict_and_extend(list_of_sentences, tagger):
    sentence_list = list(zip(*list_of_sentences))[1]
    tagger.predict(sentence_list)
    flair_entities = [s.to_dict(tag_type='ner')['entities'] for s in sentence_list]
    indices_to_be_added = list()
    annotations_to_be_added = list()

    for j, e in enumerate(flair_entities):
        conf = get_avg_confidence(e)
        if conf >= CONF_THRESHOLD:
            src_a = data_processing.Annotation()
            src_a.tokens = sentence_list[j].to_plain_string().split(" ")
            src_a.ner_tags = get_ner_tags(src_a.tokens, e)
            # print("Entities: ", e)
            # print("Average confidence: ", conf)
            # print("Tokens: ", src_a.tokens)
            # print("NER tags: ", src_a.ner_tags)
            src_a.span_list = data_processing.get_entity_spans(src_a.ner_tags)
            annotations_to_be_added .append(src_a)
            indices_to_be_added.append(list_of_sentences[j][0])

    return indices_to_be_added, annotations_to_be_added


def read_src_file(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    tagger = SequenceTagger.load('ner')

    count = 0
    list_of_sentences = list()
    indices_final_sentences = list()
    src_annotated_list = list()

    for i, line in enumerate(fin):
        if i > MAX_LIM:
            break
        if len(indices_final_sentences) > MAX_SAMPLES:
            break
        sent = line.rstrip()
        tokens = sent.split(" ")
        if len(tokens) >= MIN_TOKENS:
            count += 1

            list_of_sentences.append((i, Sentence(sent)))

            if (count + 1) % MINIBATCH_SIZE == 0:
                start = time.time()

                indices_to_be_added, annotations_to_be_added = \
                    predict_and_extend(list_of_sentences, tagger)

                indices_final_sentences.extend(indices_to_be_added)
                src_annotated_list.extend(annotations_to_be_added)

                list_of_sentences = list()
                print("######################################")
                print("Time taken: ", time.time()-start)
                print("On index: ", i)

    if len(list_of_sentences) > 0:
        indices_to_be_added, annotations_to_be_added = \
            predict_and_extend(list_of_sentences, tagger)

        indices_final_sentences.extend(indices_to_be_added)
        src_annotated_list.extend(annotations_to_be_added)

    print("Number of sentences with minimum %d tokens: %d " % (MIN_TOKENS, count))
    print("Number of sentences with entities above a certain threshold: ", len(indices_final_sentences))
    return src_annotated_list, indices_final_sentences


def parse_arguments():
    parser = ArgumentParser(description='Argument Parser for cross-lingual NER.')

    parser.add_argument('--src_lang', dest='src_lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='hi', help='Target language')
    parser.add_argument('--translate_fname', dest='translate_fname', type=str, default='train-parallel',
                        help='File name to be translated from src_lang to tgt_lang (train, dev, test)')

    return parser.parse_args()


def main():
    args = parse_arguments()
    output_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    input_path = os.path.join(output_path, "parallel")

    src_file_path = os.path.join(output_path, args.src_lang, args.translate_fname + "-" + args.tgt_lang +
                                 "_processed.pkl")
    base_path = os.path.join(output_path, args.src_lang + "-" + args.tgt_lang)
    tgt_file_path = os.path.join(base_path, args.translate_fname +
                                 "-" + args.tgt_lang + "_tgt_annotated_list_period_all" + ".pkl")

    if os.path.exists(src_file_path):
        src_annotated_list = pickle.load(open(src_file_path, "rb"))
        tgt_annotated_list = pickle.load(open(tgt_file_path, "rb"))

        sentence_ids = random.sample(list(range(len(src_annotated_list))), NUM_SAMPLES)
        with open(os.path.join(base_path, "parallel_sentence_ids.pkl"), 'wb') as f:
            pickle.dump(sentence_ids, f)

        new_src_annotated_list = list()
        for sid in sentence_ids:
            src_a = src_annotated_list[sid]
            # src_a.span_list = data_processing.get_entity_spans(src_a.ner_tags)
            new_src_annotated_list.append(src_a)
        src_file_path = os.path.join(output_path, args.src_lang, args.translate_fname +
                                     "-" + args.tgt_lang + "-final_processed.pkl")
        with open(src_file_path, 'wb') as f:
            pickle.dump(new_src_annotated_list, f)

        new_tgt_annotated_list = list()
        for sid in sentence_ids:
            new_tgt_annotated_list.append(tgt_annotated_list[sid])
        tgt_file_path = os.path.join(base_path, args.translate_fname +
                                     "-" + args.tgt_lang + "-final_tgt_annotated_list_period_all" + ".pkl")
        with open(tgt_file_path, 'wb') as f:
            pickle.dump(new_tgt_annotated_list, f)

        tgt_sentence_list = list()
        for sid in sentence_ids:
            tgt_sentence_list.append(" ".join(tgt_annotated_list[sid].tokens))
        tgt_file_path = os.path.join(base_path, args.translate_fname +
                                     "-" + args.tgt_lang + "-final_all" + ".pkl")
        with open(tgt_file_path, 'wb') as f:
            pickle.dump(tgt_sentence_list, f)

    else:
        file_path = os.path.join(input_path, args.src_lang + "-" + args.tgt_lang, args.src_lang)
        src_annotated_list, indices_final_sentences = read_src_file(file_path)

        print("Length of source annotation list: ", len(src_annotated_list))
        print("Length of indices list: ", len(indices_final_sentences))

        with open(src_file_path, 'wb') as f:
            pickle.dump(src_annotated_list, f)

        file_path = os.path.join(input_path, args.src_lang + "-" + args.tgt_lang, args.tgt_lang)
        tgt_annotated_list = read_tgt_file(file_path, indices_final_sentences)

        print("Length of target annotation list: ", len(tgt_annotated_list))

        with open(tgt_file_path, 'wb') as f:
            pickle.dump(tgt_annotated_list, f)


if __name__ == "__main__":
    main()
