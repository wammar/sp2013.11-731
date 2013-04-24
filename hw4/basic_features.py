import re
import math

ru_re = re.compile(u'[\u0430-\u044f]+', re.UNICODE)

def n_oov(source, target, alignment):
    yield 'n_oov', sum(1 for w in target if ru_re.match(w))

def log_oov_ratio(source, target, alignment):
    yield 'log_oov_ratio', math.log(1 + sum(1 for w in target if ru_re.match(w))/float(len(target)))

AVG_EF_RATIO = 0.974
STD_EF_RATIO = 0.198
def log_ef_ratio(source, target, alignment):
    yield 'log_ef_ratio', math.log(abs(len(target)/float(len(source)) - AVG_EF_RATIO)/STD_EF_RATIO)

def ef_ratio(source, target, alignment):
    yield 'ef_ratio', abs(len(target)/float(len(source)) - AVG_EF_RATIO)

def n_target(source, target, alignment):
    yield 'n_target', len(target)

STOP = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', '\'s', 'n\'t', '\'d', 'can', 'will', 'just', 'should', 'now'])
PUNCT = re.compile(r'^[:;!\?\%\$#\*\"\(\)\[\]\/,\.]$')
NUM = re.compile(r'^\d+\.?\d*$')

def n_target_type(source, target, alignment):
    yield 'n_target_content', sum(w not in STOP for w in target)
    yield 'n_target_punct', sum(bool(PUNCT.match(w)) for w in target)
    yield 'n_target_numbers', sum(bool(NUM.match(w)) for w in target)
