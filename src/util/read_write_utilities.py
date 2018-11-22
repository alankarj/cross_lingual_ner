def get_tag_dist(ner_tag_list):
    tag_dist = dict()
    anomalous = list()
    count = 0
    for i, ner_tags in enumerate(ner_tag_list):
        for j, curr_tag in enumerate(ner_tags):
            curr_tag_original = copy.deepcopy(curr_tag)
            if j == 0:
                prev_tag = curr_tag
                prev_type = None
                continue
            if prev_tag.startswith(("I", "B")):
                split = prev_tag.split("-")
                prev_tag = split[0] + "-X"
                prev_type = split[1]
            if curr_tag.startswith(("I", "B")):
                split = curr_tag.split("-")
                curr_tag = split[0]
                curr_type = split[1]
                if curr_type == prev_type:
                    curr_tag += "-X"
                else:
                    curr_tag += "-Y"
            key = prev_tag + "_" + curr_tag
            if key not in tag_dist:
                tag_dist[key] = 1
                count += 1
            else:
                tag_dist[key] += 1
                count += 1
            if key == 'B-X_I-X':
                anomalous.append((i, j))
            prev_tag = curr_tag_original
            prev_type = None
    return tag_dist, anomalous, count