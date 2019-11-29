import os
import random

from argparse import ArgumentParser
from src.util import data_processing, tmp, annotation_evaluation
from src import config
import pickle

random.seed(config.SEED)


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


def parse_arguments():
    parser = ArgumentParser(description="Argument Parser for cross-lingual "
                                        "NER.")
    parser.add_argument("--src_lang", dest="src_lang", type=str, default="en",
                        help="Source language")
    parser.add_argument("--tgt_lang", dest="tgt_lang", type=str, default="es",
                        help='Target language')
    parser.add_argument("--translate_fname", dest="translate_fname", type=str,
                        default="train",
                        help="File name to be translated from src_lang to "
                             "tgt_lang (train, dev, test)")
    parser.add_argument("--api_key_fname", dest="api_key_fname", type=str,
                        default="api_key", help="File name for Google API key")
    # Set this flag to 1 if the input is in its raw form, i.e., in a text file
    # with each token and its corresponding tag (separated by a space) in
    # a separate line.
    parser.add_argument("--pre_process", dest="pre_process", type=int,
                        default=0,
                        help="Whether to pre-process raw data or not")
    # Set this flag to 2 if the TRANSLATE step, i.e., translation of the
    # annotated source corpus to the target language, sentence-by-sentence needs
    # to be performed. Set it to 1 if only the first step of MATCH,
    # i.e., translation of entity phrases needs to be performed.
    parser.add_argument("--trans_sent", dest="trans_sent", type=int, default=0,
                        help="Whether to translate full sentences or not")
    # TODO(alankarjain): Explain what translating all sentences means.
    parser.add_argument("--translate_all", dest="translate_all", type=int,
                        default=1,
                        help="Whether to translate all sentences or not")
    # Verbosity of the output (0/1/2).
    parser.add_argument("--verbosity", dest="verbosity", type=int, default=2,
                        help="Verbosity level")
    # TODO(alankarjain): Explain what sentence_ids means.
    parser.add_argument("--sentence_ids_file", dest="sentence_ids_file",
                        type=str, default="sentence_ids",
                        help="Sentence IDs file name")
    # TODO(alankarjain): Explain what num_sample means.
    parser.add_argument('--num_sample', dest='num_sample', type=int,
                        default=100, help="Number of sentences to sample for "
                                          "partial translation.")

    parser.add_argument('--sent_iter', dest='sent_iter', type=int,
                        default=-1, help="Iteration index of the batch "
                                         "of source sentences during which a "
                                         "translation error occurred while "
                                         "translating sentences.")

    parser.add_argument('--phrase_iter', dest='sent_iter', type=int,
                        default=-1, help="Iteration index of the source "
                                         "sentence during which a translation "
                                         "error occurred while translating "
                                         "only entitites.")


    return parser.parse_args()


def verify_arguments(args):
    if args.pre_process not in [0, 1]:
        raise ValueError(val_error_msg(args.pre_process, "pre_process") %
                         args.pre_process)
    if args.trans_sent not in [0, 1, 2]:
        raise ValueError(val_error_msg(args.trans_sent, "trans_sent") %
                         args.trans_sent)
    if args.translate_all not in [0, 1]:
        raise ValueError(val_error_msg(args.translate_all, "translate_all") %
                         args.translate_all)
    if args.verbosity not in [0, 1, 2]:
        raise ValueError(val_error_msg(args.verbosity, "verbosity") %
                         args.verbosity)
    if args.num_sample < 0:
        raise ValueError(val_error_msg(args.num_sample, "num_sample") %
                         args.num_sample)


def main():
    args = parse_arguments()
    verify_arguments(args)

    args.data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    args.src_file_path = os.path.join(args.data_path, args.src_lang)
    args.tgt_file_path = os.path.join(args.data_path, args.tgt_lang)

    file_name = args.translate_fname + "_processed"
    annotated_list_file_path = os.path.join(args.src_file_path,
                                            file_name + config.PKL_EXT)

    # Pre-process the raw input only if the flag is set to 1.
    if args.pre_process == 1:
        file_path = os.path.join(args.src_file_path, args.translate_fname)
        # Unless supported by config.gold_test_dict, set the lang always to
        # None below. This argument is only used for performing tests to ensure
        # that the preprocessing has been performed correctly.
        annotated_data = data_processing.AnnotatedData(file_path, lang=None,
                                                       verbosity=args.verbosity)
        annotated_list = annotated_data.process_data()
        with open(annotated_list_file_path, 'wb') as f:
            pickle.dump(annotated_list, f)
    else:
        # Load the pre-processed annotated list.
        annotated_list = pickle.load(open(annotated_list_file_path, 'rb'))

    with open(os.path.join(args.data_path, args.api_key_fname), "r",
              encoding="utf-8") as f:
        args.api_key = f.read().strip("\n")

    translation = tmp.TMP(annotated_list, args)
    translation.translate_data()
    if args.trans_sent == 0:
        for lexicon in config.LEXICON_FILE_NAMES:
            base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
            src_file_path = os.path.join(base_path, lexicon + ".txt")
            tgt_file_path = os.path.join(base_path, lexicon + config.PKL_EXT)
            tmp.read_lexicon(src_file_path, tgt_file_path)

        translation.prepare_mega_tgt_phrase_list(calc_count_found=False)
        translation.get_tgt_annotations_new()
        translation.prepare_train_file()

    # translation = data_translation_phrase.Translation(annotated_list, args)
    # translation.translate_data()
    # translation.prepare_mega_tgt_phrase_list(calc_count_found=False)
    # translation.get_tgt_annotations_new()

    # prepare_data_new_format(args.tgt_lang)
    #
    # base_path = os.path.join(args.data_path, args.src_lang + "-" + args.tgt_lang)
    # path = os.path.join(base_path, "train" + "_annotated_list_0.5_period_" + "all" + "27-11-2018" + ".pkl")
    # tgt_annotated_list = pickle.load(open(path, 'rb'))
    # data_processing.prepare_train_file_new_format(tgt_annotated_list, os.path.join(args.data_path,
    #                                                                                "en-" + args.tgt_lang),
    #                                               key="train")


if __name__ == '__main__':
    main()
