max_len = 128
# max_noSpanLen = 60

max_span_len = 6
max_node = 4
log_fre = 10
tag2idx = {
    "PAD": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-OTHER": 7,
    "I-OTHER": 8,
    "O": 9,
    "X": 10,
    "CLS": 11,
    "SEP": 12
}


idx2tag = {idx: tag for tag, idx in tag2idx.items()}


tag2idx_noSpan = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
    "CLS": 0,
    "SEP": 0
}
tag2idx_Span = {
    "notTarget": 0,
    "PER": 1,
    "LOC": 2,
    "ORG": 3,
    "OTHER": 4,
}