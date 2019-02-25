import os
import random

from argparse import ArgumentParser
from src.util import data_processing, data_translation, annotation_evaluation
from src import config
import pickle

from flair.data import Sentence
from flair.models import SequenceTagger

random.seed(config.SEED)


def verify_arguments(args):
    if args.src_lang not in config.SRC_LANGS:
        raise ValueError(val_error_msg(args.src_lang, 'src_lang') % args.src_lang)
    if args.tgt_lang not in config.TGT_LANGS:
        raise ValueError(val_error_msg(args.tgt_lang, 'tgt_lang') % args.tgt_lang)
    if args.translate_fname not in config.TRANSLATE_FNAMES:
        raise ValueError(val_error_msg(args.translate_fname, 'translate_fname') % args.translate_fname)
    if args.pre_process not in [0, 1]:
        raise ValueError(val_error_msg(args.pre_process, 'pre_process') % args.pre_process)
    if args.trans_sent not in [0, 1, 2]:
        raise ValueError(val_error_msg(args.trans_sent, 'trans_sent') % args.trans_sent)
    if args.translate_all not in [0, 1]:
        raise ValueError(val_error_msg(args.translate_all, 'translate_all') % args.translate_all)
    if args.verbosity not in [0, 1, 2]:
        raise ValueError(val_error_msg(args.verbosity, 'verbosity') % args.verbosity)
    if args.num_sample < 0:
        raise ValueError(val_error_msg(args.num_sample, 'num_sample') % args.num_sample)


def val_error_msg(var, name):
    assert type(name) == str
    if type(var) == int:
        placeholder = '%d'
    elif type(var) == float:
        placeholder = '%.3f'
    elif type(var) == str:
        placeholder = '%s'
    else:
        return 'Incorrect type for %s' % name
    return placeholder + ' ' + name + ' not supported.'


def prepare_data_new_format(tgt_lang):
    data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    # data_prep_helper(tgt_lang, data_path, key="testa")
    # data_prep_helper(tgt_lang, data_path, key="testb")
    data_prep_helper(tgt_lang, data_path, key="train_original")


def data_prep_helper(tgt_lang, data_path, key):
    annotated_list = get_annotated_list(os.path.join(data_path, tgt_lang, key))
    # data_processing.prepare_train_file_new_format(annotated_list, os.path.join(data_path,
    #                                                                            "en-" + tgt_lang), key)
    data_processing.prepare_train_file(annotated_list, None, os.path.join(data_path,
                                                                          "en-" + tgt_lang, key + "_cleaned"))


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


def prepare_tagger_input_file(tagger_file_path, tgt_annotated_list):
    with open(tagger_file_path, "w", encoding="utf-8") as f:
        for tgt_a in tgt_annotated_list:
            write_str = ""
            for tok in tgt_a.tokens:
                write_str += tok + " "
            write_str.rstrip(" ")
            f.writelines(write_str + "\n")


def parse_arguments():
    parser = ArgumentParser(description='Argument Parser for cross-lingual NER.')

    parser.add_argument('--src_lang', dest='src_lang', type=str, default='hi', help='Source language')
    parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='en', help='Target language')
    parser.add_argument('--translate_fname', dest='translate_fname', type=str, default='test',
                        help='File name to be translated from src_lang to tgt_lang (train, dev, test)')
    parser.add_argument('--api_key_fname', dest='api_key_fname', type=str, default='api_key',
                        help='File name for Google API key')
    parser.add_argument('--pre_process', dest='pre_process', type=int, default=0,
                        help='Whether to pre-process raw data or not')
    parser.add_argument('--trans_sent', dest='trans_sent', type=int, default=0,
                        help='Whether to translate full sentences or not')
    parser.add_argument('--translate_all', dest='translate_all', type=int, default=1,
                        help='Whether to translate all sentences or not')
    parser.add_argument('--verbosity', dest='verbosity', type=int, default=2, help='Verbosity level')
    parser.add_argument('--sentence_ids_file', dest='sentence_ids_file', type=str, default="sentence_ids",
                        help='Sentence IDs file name')
    parser.add_argument('--num_sample', dest='num_sample', type=int, default=100,
                        help='Number of sentences to sample for partial translation.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    verify_arguments(args)

    args.data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    args.src_file_path = os.path.join(args.data_path, args.src_lang)
    args.tgt_file_path = os.path.join(args.data_path, args.tgt_lang)

    file_name = args.translate_fname + "_processed"
    annotated_list_file_path = os.path.join(args.src_file_path, file_name + ".pkl")

    if args.pre_process == 1:
        file_path = os.path.join(args.src_file_path, args.translate_fname)
        annotated_data = data_processing.AnnotatedData(file_path, verbosity=args.verbosity)
        annotated_list = annotated_data.process_data()
        with open(annotated_list_file_path, 'wb') as f:
            pickle.dump(annotated_list, f)
    else:
        annotated_list = pickle.load(open(annotated_list_file_path, 'rb'))

        # Store these sentences in the <tgt>-<src> folder for use later in annotation projection.
        base_path = os.path.join(args.data_path, args.tgt_lang + "-" + args.src_lang)

        sentence_ids = list(range(len(annotated_list)))
        pseudo_tgt_sentence_list = [' '.join(annotated_list[sid].tokens)
                                    for sid in sentence_ids]

        pseudo_tgt_sentence_path = os.path.join(base_path,
                                                args.translate_fname + '_' + "all" +
                                                config.PKL_EXT)
        with open(pseudo_tgt_sentence_path, 'wb') as f:
            pickle.dump(pseudo_tgt_sentence_list, f)

    with open(os.path.join(args.data_path, args.api_key_fname), 'r', encoding='utf-8') as f:
        args.api_key = f.read().strip('\n')

    if args.trans_sent > 0:
        translation = data_translation.Translation(annotated_list, args)
        translation.translate_data()

    else:
        base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
        file_path = os.path.join(base_path, args.translate_fname + '_' + "all" + config.PKL_EXT)
        tgt_sentence_list = pickle.load(open(file_path, 'rb'))

        print("Length of target sentence list: ", len(tgt_sentence_list))

        path = os.path.join(base_path, args.translate_fname + "_tgt_annotated_list_" + "all_partial" + ".pkl")

        if os.path.exists(path):
            tgt_annotated_list = pickle.load(open(path, "rb"))

        else:
            tgt_annotated_list = list()
            for i, sentence in enumerate(tgt_sentence_list):
                print("######################################################################")
                print("Sentence-%d: %s" % (i, sentence))
                sentence = sentence.replace("&#39;", "\'")
                sentence = sentence.replace(" &amp; ", "&")
                tgt_annotation = data_processing.Annotation()
                tgt_annotation.tokens = data_translation.get_clean_tokens(sentence, False)
                print("Tokens: ", tgt_annotation.tokens)
                tgt_annotated_list.append(tgt_annotation)
            with open(path, "wb") as f:
                pickle.dump(tgt_annotated_list, f)

        print("Length of target annotated list: ", len(tgt_annotated_list))

        # input_tagger_file_path = os.path.join(config.TAGGER_FILE_PATH, "input_" +
        #                                       args.translate_fname + ".txt")
        # prepare_tagger_input_file(input_tagger_file_path, tgt_annotated_list)
        #
        # output_tagger_file_path = os.path.join(config.TAGGER_FILE_PATH, "output_" +
        #                                        args.translate_fname + ".txt")
        #
        # path = os.path.join(base_path, args.translate_fname + "_tgt_annotated_list_" + "all_full" + ".pkl")
        #
        # with open(output_tagger_file_path, "r", encoding="utf-8") as f:
        #     rows = f.readlines()
        #     ner_tag_list = list()
        #     for row in rows:
        #         ner_tags = list()
        #         row = row.strip("\n").split(" ")
        #         for r in row:
        #             ner_tags.append(r.split("__")[1])
        #         ner_tag_list.append(ner_tags)

        tagger = SequenceTagger.load('ner')
        for i, tgt_a in enumerate(tgt_annotated_list):
            print("######################################################################")
            print("Sentence: ", i)
            sent = Sentence(tgt_sentence_list[i])
            print("Tokens: ", tgt_a.tokens)
            tagger.predict(sent)
            entities = sent.to_dict(tag_type='ner')['entities']
            print(entities)
            # tgt_a.ner_tags = ner_tag_list[i]
            # tgt_a.span_list = data_processing.get_entity_spans(tgt_a.ner_tags)
            # print("NER tags: ", tgt_a.ner_tags)

        # with open(path, 'wb') as f:
        #     pickle.dump(tgt_annotated_list, f)


if __name__ == '__main__':
    main()
