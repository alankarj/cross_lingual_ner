import os
import random
import json
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
    if args.translate_all not in [0, 1]:
        raise ValueError(val_error_msg(args.translate_all, 'translate_all') % args.translate_all)
    if args.align_heuristic not in config.ALIGN_HEURISTICS:
        raise ValueError(val_error_msg(args.align_heuristic, 'align_heuristic') % args.align_heuristic)


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
    parser.add_argument('--translate_all', dest='translate_all', type=int, default=1,
                        help='Whether to translate all sentences or not')
    parser.add_argument('--align_heuristic', dest='align_heuristic', type=str, default='fast-align',
                        help='Alignment heuristic for projecting annotations from src to tgt language.')

    return parser.parse_args()


def main():
    args = parse_arguments()
    verify_arguments(args)

    args.data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    args.src_file_path = os.path.join(args.data_path, args.src_lang, args.translate_fname)
    args.tgt_file_path = os.path.join(args.data_path, args.tgt_lang, args.translate_fname)
    base_path = os.path.join(args.data_path, args.src_lang + '-' + args.tgt_lang)

    annotated_list_file_path = os.path.join(args.src_file_path + config.PROCESSED_FILE_SUFFIX + '.pkl')
    annotated_list = pickle.load(open(annotated_list_file_path, 'rb'))

    num_sentences = len(annotated_list)
    sentence_ids = list(range(num_sentences))
    print("Number of source (%s) sentences: %d" % (args.src_lang, num_sentences))

    src_sentence_list = [' '.join(annotated_list[sid].tokens) for sid in sentence_ids]

    tgt_sentence_suffix = config.ALL_PARTIAL_SUFFIXES[args.translate_all]
    tgt_sentence_path = os.path.join(base_path, args.translate_fname + '_' + tgt_sentence_suffix + config.PKL_EXT)
    tgt_sentence_list = pickle.load(open(tgt_sentence_path, 'rb'))
    print("Number of target (%s) sentences: %d" % (args.tgt_lang, len(tgt_sentence_list)))

    assert len(src_sentence_list) == len(tgt_sentence_list)

    tgt_token_list = [data_translation.get_clean_tokens(tgt_sentence) for tgt_sentence in tgt_sentence_list]
    ner_tag_list = [annotated_list[sid].ner_tags for sid in sentence_ids]

    if args.align_heuristic == config.ALIGN_HEURISTICS[0]:
        # If the alignment heuristic is fast align.

        # fast_align_input_path = os.path.join(base_path, args.translate_fname + '_' + args.align_heuristic)
        # data_translation.generate_fast_align_data(src_sentence_list, tgt_sentence_list, fast_align_input_path)
        # print("Input data for %s generated." % args.align_heuristic)
        # print("Now running %s code." % args.align_heuristic)
        # os.system("cd " + config.FAST_ALIGN_PATH + "; " + "./fast_align -d -o -v -I 50 -i"
        #           + fast_align_file_path + " > " + os.path.join(base_path, config.FAST_ALIGN_OUTPUT_FNAME))
        # print("Done.")

        print("Getting projected tags.")
        fast_align_output_path = os.path.join(base_path, config.FAST_ALIGN_OUTPUT_FNAME)
        predicted_tag_list = annotation_evaluation.get_fast_align_annotations(fast_align_output_path, ner_tag_list, tgt_token_list)
    else:
        file_name = args.translate_fname + '_' + config.ALL_PARTIAL_SUFFIXES[args.translate_all] + '_annotated' + config.PKL_EXT
        tgt_annotated_list = pickle.load(open(os.path.join(base_path, file_name), 'rb'))

        predicted_tag_list = []
        for i, tgt_sentence in enumerate(tgt_annotated_list):
            predicted_tag_list.append(data_translation.get_readable_annotations(tgt_sentence.tokens, tgt_sentence.ner_tags))

    predicted_tag_list = annotation_evaluation.process_annotations(predicted_tag_list)
    print("Verifying generated annotations.")
    dist, anomalous_list, count = data_processing.get_tag_dist(predicted_tag_list)
    print("Tag distribution: ", json.dumps(dist, indent=2))
    num_tokens = 0
    for tokens in tgt_token_list:
        num_tokens += len(tokens)
    assert num_tokens - count == num_sentences

    # for i, j in anomalous_list:
    #     # print("Tokens: ", annotated_list[i].tokens)
    #     print("Tag: ", annotated_list[i].tokens[j-1])

    print("Converting generated alignments into trainable form.")

    with open(os.path.join(base_path, args.translate_fname + '_' + args.align_heuristic + '_final'),
              'w', encoding='utf-8') as f:
        for i, tokens in enumerate(tgt_token_list):
            for j, token in enumerate(tokens):
                write_str = token + " " + predicted_tag_list[i][j] + "\n"
                if len(write_str.split()) < 2:
                    write_str = '-' + write_str
                assert len(write_str.split()) == 2
                f.writelines(write_str)
            f.writelines("\n")


if __name__ == '__main__':
    main()
