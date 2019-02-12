import argparse


def get_entity(label):
    entities = []
    i = 0
    while i < len(label):
        if label[i] != 'O':
            e_type = label[i][2:]
            j = i + 1
            while j < len(label) and label[j] == 'I-' + e_type:
                j += 1
            entities.append((i, j, e_type))
            i = j
        else:
            i += 1
    return entities


def evaluate_ner(pred, gold):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(pred)):
        pred_entities = get_entity(pred[i])
        gold_entities = get_entity(gold[i])
        temp = 0
        for entity in pred_entities:
            if entity in gold_entities:
                tp += 1
                temp += 1
            else:
                fp += 1
        fn += len(gold_entities) - temp
    precision = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def load_data(path, label_index, isIOB2):
    sentences = []
    labels = []

    sentence = []
    label = []
    for line in open(path, 'r'):
        if len(line.strip()) != 0:
            sentence.append(line.strip().split()[0])
            label.append(line.strip().split()[label_index])
        else:
            if len(sentence) > 1:
                sentences.append(sentence)
                if not isIOB2:
                    iob2(label)
                labels.append(label)
                sentence = []
                label = []
    if len(sentence) > 1:
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels


def iob2(tags):

    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def main():
    parser = argparse.ArgumentParser(description='lasagne simple neural crf')

    parser.add_argument('--true_data_path', type=str, help='true data path')
    parser.add_argument('--pred_data_path', type=str, help='pred data path')

    args = parser.parse_args()

    true_X, true_Y = load_data(args.true_data_path, 1, True)
    pred_X, pred_Y = load_data(args.pred_data_path, 1, True)

    p, r, f = evaluate_ner(pred_Y, true_Y)
    print("recall: {:.4f} | precision: {:.4f} | f1 {:.4f}".format(p, r, f))


if __name__ == '__main__':
    main()
