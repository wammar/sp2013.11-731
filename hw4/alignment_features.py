from collections import Counter, defaultdict
import io

STOP = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', '\'s', 'n\'t', '\'d', 'can', 'will', 'just', 'should', 'now'])

def read_word_pair_features(word_pair_features_filename):
    word_pair_features = defaultdict(list)
    with io.open(word_pair_features_filename, encoding='utf8') as f:
        for line in f:
            (src, tgt, features) = line.strip().split(' ||| ')
            features_list = []
            for feature in features.split():
                feature_name, feature_value = feature.split('=')
                features_list.append( (feature_name, float(feature_value)) )
            word_pair_features[ (src, tgt) ] = list(features_list)
    return word_pair_features

word_pair_features = read_word_pair_features('/home/wammar/mt-hw4/talign/grammars/wordpairs.f-e.features')

def read_test_set_word_pairs(filename):
    pairs = set()
    with io.open(filename, encoding='utf8') as f:
        for line in f:
            pairs.add(line.strip())
    return pairs

test_set_word_pairs = read_test_set_word_pairs('data/test.100best.word_pairs')

def dist_2_diag(source, target, alignment):
    total_cost = 0
    for (src_pos, tgt_pos) in alignment:
        total_cost += abs( (src_pos + 1.0) / len(source) - (tgt_pos + 1.0) / len(target) )
    avg_cost = total_cost / len(alignment)
    yield 'dist_2_diag', avg_cost

def jump_dist(source, target, alignment):
    forward_jumps = 0.0
    forward_jump_dist = 0.0
    backward_jumps = 0.0
    backward_jump_dist = 0.0
    src_prev_pos = 0
    for (src_pos, tgt_pos) in alignment:
        if src_pos > src_prev_pos + 1:
            forward_jumps += 1
            forward_jump_dist += abs(src_pos - src_prev_pos - 1)
        elif src_pos < src_prev_pos:
            backward_jumps += 1
            backward_jump_dist += abs(src_prev_pos - src_pos)
        src_prev_pos = src_pos
    
    yield 'fwd_jump', forward_jumps / len(target)
    yield 'fwd_jump_dist', forward_jump_dist / len(target)
    yield 'bwd_jump', backward_jumps / len(target)
    yield 'bwd_jump_dist', backward_jump_dist / len(target)
    yield 'total_jump', (forward_jumps - backward_jumps) / len(target)
    yield 'total_jump_dist', (forward_jump_dist - backward_jump_dist) / len(target)

def fertilities(source, target, alignment):
    fertility = Counter()
    for src_pos, tgt_pos in alignment:
        fertility[src_pos] += 1
    fertility_counts = Counter(fertility.values())
    total_fertility = 0
    for f, c in fertility_counts.iteritems():
        yield 'fertility_'+str(f), c/float(len(source))
        total_fertility += c
    yield 'fertility_0', (1 - total_fertility/float(len(source)))

def word_pair(source, target, alignment):
    prev_tgt_pos = -1
    for (src_pos, tgt_pos) in alignment:
        for i in range(prev_tgt_pos, tgt_pos):
            fname = u'pair_NULL_{0}'.format(target[i])
            if fname in test_set_word_pairs:
                yield fname, 1.0
        fname = u'pair_{0}_{1}'.format(source[src_pos], target[tgt_pos])
        if fname in test_set_word_pairs:
            yield fname, 1.0
        prev_tgt_pos = tgt_pos

def morph(source, target, alignment):
    for (src_pos, tgt_pos) in alignment:
        word_pair = (source[src_pos], target[tgt_pos])
        if len(source[src_pos]) > 1:
            yield u'{0}-{1}'.format(word_pair[0][-1:], word_pair[1]), 1.0
        if len(source[src_pos]) > 2:
            yield u'{0}-{1}'.format(word_pair[0][-2:], word_pair[1]), 1.0
        if len(source[src_pos]) > 3:
            yield u'{0}-{1}'.format(word_pair[0][-3:], word_pair[1]), 1.0

def morph_2_stop(source, target, alignment):
    for (src_pos, tgt_pos) in alignment:
        word_pair = (source[src_pos], target[tgt_pos])
        if word_pair[1] not in STOP:
            continue
        if len(source[src_pos]) > 1:
            yield u'{0}-{1}'.format(word_pair[0][-1:], word_pair[1]), 1.0
        if len(source[src_pos]) > 2:
            yield u'{0}-{1}'.format(word_pair[0][-2:], word_pair[1]), 1.0
        if len(source[src_pos]) > 3:
            yield u'{0}-{1}'.format(word_pair[0][-3:], word_pair[1]), 1.0

def coarse_word_pair(source, target, alignment):
    for (src_pos, tgt_pos) in alignment:
        word_pair = (source[src_pos], target[tgt_pos])
        if word_pair in word_pair_features:
            for feature in word_pair_features[word_pair]:
                yield feature[0], feature[1]
