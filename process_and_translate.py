import os
import random

from argparse import ArgumentParser
from src.util import data_processing, data_translation, annotation_evaluation
from src import config
import pickle

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
    data_prep_helper(tgt_lang, data_path, key="testa")
    data_prep_helper(tgt_lang, data_path, key="testb")
    # data_prep_helper(tgt_lang, data_path, key="train_original")


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


def parse_arguments():
    parser = ArgumentParser(description='Argument Parser for cross-lingual NER.')

    parser.add_argument('--src_lang', dest='src_lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='es', help='Target language')
    parser.add_argument('--translate_fname', dest='translate_fname', type=str, default='train',
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
        annotated_data = data_processing.AnnotatedData(args.src_file_path, lang=args.src_lang,
                                                       verbosity=args.verbosity)
        annotated_list = annotated_data.process_data()
        with open(annotated_list_file_path, 'wb') as f:
            pickle.dump(annotated_list, f)

        write_file_path = os.path.join(args.src_file_path, file_name + ".tsv")
        data_processing.write_to_file([annotated_list], write_file_path)
    else:
        annotated_list = pickle.load(open(annotated_list_file_path, 'rb'))

    with open(os.path.join(args.data_path, args.api_key_fname), 'r', encoding='utf-8') as f:
        args.api_key = f.read().strip('\n')

    #base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
    # src_file_path = os.path.join(base_path, "lexicon_ground-truth")
    # tgt_file_path = os.path.join(base_path, "lexicon_ground-truth.pkl")
    # data_translation.read_lexicon(src_file_path, tgt_file_path)

    translation = data_translation.Translation(annotated_list, args)
    translation.translate_data()
    translation.prepare_mega_tgt_phrase_list(calc_count_found=False)
    translation.get_tgt_annotations_new()
    translation.prepare_train_file()

    prepare_data_new_format(args.tgt_lang)
    #
    # base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
    # path = os.path.join(base_path, "train" + "_annotated_list_0.5_period_" + "all" + "27-11-2018" + ".pkl")
    # tgt_annotated_list = pickle.load(open(path, 'rb'))
    # data_processing.prepare_train_file_new_format(tgt_annotated_list, os.path.join(args.data_path,
    #                                                                                "en-" + args.tgt_lang),
    #                                               key="train")


if __name__ == '__main__':
    main()
