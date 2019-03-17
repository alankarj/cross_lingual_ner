import io
import os
from src import config
from argparse import ArgumentParser
from heapq import heappush, heappop
import nltk


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split())
    data = {}
    tokens_list = []
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        assert " " not in tokens
        tokens_list.append(tokens[0])
        data[tokens[0]] = map(float, tokens[1:])
    return tokens_list


def save_vectors_romanized(input_path, output_path, romanized_tokens):
    fin = io.open(input_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fout = io.open(output_path, 'w', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    fout.write(str(n) + " " + str(d) + "\n")
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        # assert len(tokens) > 0
        # print(line)
        # print(tokens)
        fout.write(romanized_tokens[i].strip().replace(" ", "") + line[len(tokens[0]):])


def save_in_file(tokens, fname):
    with open(fname, 'w', encoding='utf-8') as f:
        for t in tokens:
            f.writelines(t + "\n")


def parse_arguments():
    parser = ArgumentParser(description='Argument Parser for cross-lingual NER.')

    parser.add_argument('--src_lang', dest='src_lang', type=str, default='uz', help='Source language')
    parser.add_argument('--tgt_lang', dest='tgt_lang', type=str, default='ug', help='Target language')
    parser.add_argument('--max_top', dest='max_top', type=int, default=10000, help='Maximum top tokens for seeds')
    parser.add_argument('--edit_threshold', dest='edit_threshold', type=int, default=1, help='Edit distance threshold')
    parser.add_argument('--len_threshold', dest='len_threshold', type=int, default=5, help='Source length threshold')

    return parser.parse_args()


def read_from_file(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        tokens = list()
        for row in rows:
            tokens.append(row.strip("\n"))
    return tokens


def save_lexicon(fname, lexicon):
    with open(fname, 'w', encoding='utf-8') as f:
        for token in lexicon:
            f.writelines(token + " " + lexicon[token][1] + "\n")


def main():
    args = parse_arguments()
    data_path = os.path.join(os.getcwd(), config.DATA_FOLDER)
    src_fname = "cc." + args.src_lang + ".300.vec"
    tgt_fname = "cc." + args.tgt_lang + ".300.vec"
    src_path = os.path.join(data_path, src_fname)
    tgt_path = os.path.join(data_path, tgt_fname)

    # load_vectors(os.path.join(data_path, "glove.100d.ug_romanized.txt"))

    print("Reading romanized tokens...")
    romanized_src_tokens = read_from_file(os.path.join(data_path, "ug_tokens_rom_2.txt"))

    # print("Reading romanized tokens...")
    # romanized_src_tokens = read_from_file(os.path.join(data_path,
    #                                                    args.src_lang + "_tokens_romanized.txt"))
    # romanized_tgt_tokens = read_from_file(os.path.join(data_path,
    #                                                    args.tgt_lang + "_tokens_romanized.txt"))
    #
    print("Saving romanized tokens with their vectors...")
    # output_path = os.path.join(data_path, "cc." + args.src_lang + "_romanized.300.vec")
    output_path = os.path.join(data_path, "cc.ug.300.romanized.vec")
    save_vectors_romanized(os.path.join(data_path, "cc.ug.300.vec"), output_path, romanized_src_tokens)
    # output_path = os.path.join(data_path, "cc." + args.tgt_lang + "_romanized.300.vec")
    # save_vectors_romanized(tgt_path, output_path, romanized_tgt_tokens)

    # print("Reading tokens...")
    # src_tokens = load_vectors(os.path.join(data_path, "glove.100d.ug.lorelie.txt"))
    # tgt_tokens = load_vectors(os.path.join(data_path, tgt_fname))

    # print("Saving tokens in file...")
    # save_in_file(src_tokens, os.path.join(data_path, args.src_lang + "_tokens.txt"))
    # save_in_file(src_tokens, os.path.join(data_path, "glove_ug_tokens_lorelie.txt"))
    # save_in_file(tgt_tokens, os.path.join(data_path, args.tgt_lang + "_tokens.txt"))

    # print("Reading romanized tokens...")
    # romanized_src_tokens = read_from_file(os.path.join(data_path,
    #                                                    args.src_lang + "_tokens_romanized.txt"))
    # romanized_tgt_tokens = read_from_file(os.path.join(data_path,
    #                                                    args.tgt_lang + "_tokens_romanized.txt"))
    #
    # best_matches = dict()
    #
    # for i, src_token in enumerate(romanized_src_tokens[:args.max_top]):
    #     best_edit_dist = float('inf')
    #     best_tgt_token = ""
    #
    #     if len(src_token) < args.len_threshold:
    #         continue
    #
    #     for tgt_token in romanized_tgt_tokens[:args.max_top]:
    #         if (len(src_token) < args.len_threshold - args.edit_threshold) or \
    #                 (len(src_token) > args.len_threshold + args.edit_threshold):
    #             continue
    #
    #         edit_dist = nltk.edit_distance(src_token.lower(), tgt_token.lower())
    #         if edit_dist < best_edit_dist:
    #             best_edit_dist = edit_dist
    #             best_tgt_token = tgt_token.lower()
    #
    #         if best_edit_dist <= args.edit_threshold:
    #             best_matches[src_token.lower()] = (best_edit_dist, best_tgt_token)
    #             break
    #
    #     if best_edit_dist <= args.edit_threshold:
    #         best_matches[src_token.lower()] = (best_edit_dist, best_tgt_token)
    #
    # save_lexicon(os.path.join(data_path, args.src_lang + "-" + args.tgt_lang + "_lexicon.txt"),
    #              best_matches)
    # print("Lexicon contains %d elements." % len(list(best_matches.keys())))


if __name__ == "__main__":
    main()
