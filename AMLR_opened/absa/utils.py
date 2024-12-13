import json
import collections
import numpy as np

from PIL import Image
from torchvision import transforms
from absa.randaugment import RandomAugment
from config import tag2idx, idx2tag, max_len, max_node, log_fre, tag2idx_noSpan, tag2idx_Span, max_span_len

import os
import torch

label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
id_to_label = {0: 'other', 1: 'neutral', 2: 'positive', 3: 'negative', 4: 'conflict'}


class SemEvalExample(object):
    def __init__(self,
                    example_id,
                    words,
                    image_ids=None,
                    raw_image_data=None,
                    ner_labels=None
                    ):
        self.example_id = example_id
        self.words = words
        self.image_ids = image_ids
        self.raw_image_data = raw_image_data
        self.ner_labels = ner_labels



    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 ntokens_noSpan=None,
                 label_ids_noSpan=None,
                 input_ids_noSpan=None,
                 raw_image_data=None,
                 mask_noSpan=None,
                 segment_ids_noSpan=None,
                 span_ner_labels=None,
                 span_token_level_mask=None,
                 span_start=None,
                 span_end=None,
                 image_ids=None,
                 label_ids_crf13=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.ntokens_noSpan = ntokens_noSpan
        self.label_ids_noSpan = label_ids_noSpan
        self.input_ids_noSpan = input_ids_noSpan
        self.raw_image_data = raw_image_data
        self.mask_noSpan = mask_noSpan
        self.segment_ids_noSpan = segment_ids_noSpan
        self.span_ner_labels = span_ner_labels
        self.span_token_level_mask = span_token_level_mask
        self.span_start = span_start
        self.span_end = span_end
        self.image_ids = image_ids
        self.label_ids_crf13 = label_ids_crf13


def gen_span_labes(span_starts, span_ends, bio_labels, input_mask):

    span_aspect_labels = [0]*len(span_starts)
    span_token_level_mask = [0]*len(span_starts)
    aspect_start = []
    aspect_end = []
    token_level_span_mask_1_position = []

    for i in range(len(input_mask)):
        if input_mask[i] == 1:
            token_level_span_mask_1_position.append(i)

    for i in range(len(bio_labels)):
        if i == len(bio_labels) - 1:
            if bio_labels[i] == 1 or bio_labels[i] == 3 or bio_labels[i] == 5 or bio_labels[i] == 7:
                aspect_start.append(i)
                aspect_end.append(i)
        else:
            if bio_labels[i] == 1 or bio_labels[i] == 3 or bio_labels[i] == 5 or bio_labels[i] == 7:
                aspect_start.append(i)
                for j in range(30):
                    # if i + j < len(bio_labels) and (bio_labels[i+j+1] == 2 or bio_labels[i+j+1] == 4 or bio_labels[i+j+1] == 6 or  bio_labels[i+j+1] == 8):
                    if i + j < len(bio_labels) and bio_labels[i + j + 1] == 0:
                        aspect_end.append(i+j)
                        break

    for h in range(len(span_starts)):
        for k in range(len(aspect_start)):
            if span_starts[h] == aspect_start[k] and span_ends[h] == aspect_end[k]:
                # span_sentiment_labels[h] = polarity_positions[aspect_start[k]]
                if bio_labels[span_starts[h]] == 1:
                    span_aspect_labels[h] = 1
                if bio_labels[span_starts[h]] == 3:
                    span_aspect_labels[h] = 2
                if bio_labels[span_starts[h]] == 5:
                    span_aspect_labels[h] = 3
                if bio_labels[span_starts[h]] == 7:
                    span_aspect_labels[h] = 4

    for h in range(len(span_starts)):
        if input_mask[span_starts[h]] == 1 and input_mask[span_ends[h]] == 1:
            span_token_level_mask[h] = 1

    return span_aspect_labels, span_token_level_mask


def gen_span_start_end(seq, max_span_len):
    start = []
    end = []

    #span=1
    for i in range(len(seq)):
        start.append(i)
        end.append(i)
    #span=2
    for i in range(len(seq)):
        if i + 1 <= len(seq) - 1:
            start.append((i))
            end.append(i+1)
    #span=3
    for i in range(len(seq)):
        if i + 2 <= len(seq) - 1:
            start.append((i))
            end.append(i+2)
    if max_span_len == 3:
        return start, end

        # span=4
    for i in range(len(seq)):
        if i + 3 <= len(seq) - 1:
            start.append((i))
            end.append(i + 3)
    if max_span_len == 4:
        return start, end
    # span=5
    for i in range(len(seq)):
        if i + 4 <= len(seq) - 1:
            start.append((i))
            end.append(i + 4)
    if max_span_len == 5:
        return start, end
    # span=6
    for i in range(len(seq)):
        if i + 5 <= len(seq) - 1:
            start.append((i))
            end.append(i + 5)
    if max_span_len == 6:
        return start, end
    # span=7
    for i in range(len(seq)):
        if i + 6 <= len(seq) - 1:
            start.append((i))
            end.append(i + 6)
    if max_span_len == 7:
        return start, end
    # span=8
    for i in range(len(seq)):
        if i + 7 <= len(seq) - 1:
            start.append((i))
            end.append(i + 7)
    if max_span_len == 8:
        return start, end
    # span=9
    for i in range(len(seq)):
        if i + 8 <= len(seq) - 1:
            start.append((i))
            end.append(i + 8)
    if max_span_len == 9:
        return start, end
    # span=10
    for i in range(len(seq)):
        if i + 9 <= len(seq) - 1:
            start.append((i))
            end.append(i + 9)
    if max_span_len == 10:
        return start, end


def get_labels():
    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

def convert_examples_to_features(args, EvalOrNot, examples, tokenizer, max_seq_length, verbose_logging=False, logger=None):
    max_sent_length, max_term_length = 0, 0
    unique_id = 1000000000
    features = []
    label_list = get_labels()
    label_map = {label: i for i, label in enumerate(label_list, 1)}


    Other_num = 0
    for (example_index, example) in enumerate(examples):

        image_ids = example.image_ids
        textlist = example.words
        labellist = example.ner_labels
        tokens_crf13 = []
        labels = []

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens_crf13.extend(token)
            label_1 = labellist[i]

            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)

                else:
                    labels.append("X")

        if len(tokens_crf13) >= max_seq_length - 1:
            tokens_crf13 = tokens_crf13[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        ntokens_crf13 = []
        segment_ids = []
        label_ids_crf13 = []

        ntokens_crf13.append("[CLS]")
        segment_ids.append(0)
        label_ids_crf13.append(label_map["[CLS]"])

        for i, token in enumerate(tokens_crf13):
            ntokens_crf13.append(token)
            segment_ids.append(0)
            label_ids_crf13.append(label_map[labels[i]])

        ntokens_crf13.append("[SEP]")
        segment_ids.append(0)
        label_ids_crf13.append(label_map["[SEP]"])

        while len(label_ids_crf13) < max_seq_length:
            label_ids_crf13.append(0)

        assert len(label_ids_crf13) == max_seq_length

        ntokens_noSpan = ["[CLS]"]

        label_ids_noSpan = [tag2idx_noSpan["CLS"]]
        if image_ids[0] == "17_06_242.jpg":
            print("17_06_242")
        for word, label in zip(example.words, example.ner_labels):  # iterate every word
            if label == 'B-OTHER':
                Other_num+=1
            tokens = tokenizer.tokenize(word)

            ntokens_noSpan.extend(tokens)

            for j, _ in enumerate(tokens):
                if (tag2idx_noSpan[label] == 1 or tag2idx_noSpan[label] == 3 or tag2idx_noSpan[label] == 5 or tag2idx_noSpan[label] == 7) and j == 0:
                    label_ids_noSpan.append(tag2idx_noSpan[label])
                elif (tag2idx_noSpan[label] == 1 or tag2idx_noSpan[label] == 3 or tag2idx_noSpan[label] == 5 or tag2idx_noSpan[label] == 7) and j != 0:
                    label_ids_noSpan.append(tag2idx_noSpan[label] + 1)

                if (tag2idx_noSpan[label] == 0 or tag2idx_noSpan[label] == 2 or tag2idx_noSpan[label] == 4 or tag2idx_noSpan[label] == 6 or tag2idx_noSpan[label] == 8):
                    label_ids_noSpan.append(tag2idx_noSpan[label])

        ntokens_noSpan = ntokens_noSpan[:max_seq_length - 1]
        ntokens_noSpan.append("[SEP]")

        label_ids_noSpan = label_ids_noSpan[:max_seq_length - 1]
        label_ids_noSpan.append(tag2idx_noSpan["SEP"])

        input_ids_noSpan = tokenizer.convert_tokens_to_ids(ntokens_noSpan)

        mask_noSpan = [1] * len(input_ids_noSpan)
        segment_ids_noSpan = [0] * max_seq_length

        pad_len_noSpan = max_seq_length - len(input_ids_noSpan)
        rest_pad_noSpan = [0] * pad_len_noSpan  # pad to max_len
        input_ids_noSpan.extend(rest_pad_noSpan)
        mask_noSpan.extend(rest_pad_noSpan)
        label_ids_noSpan.extend(rest_pad_noSpan)

        span_start, span_end = gen_span_start_end(label_ids_noSpan, max_span_len)
        span_ner_labels, span_token_level_mask = gen_span_labes(span_start, span_end, label_ids_noSpan, mask_noSpan)

        ntokens_noSpan.extend(["pad"] * pad_len_noSpan)

        raw_image_data = example.raw_image_data

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                ntokens_noSpan=ntokens_noSpan,
                label_ids_noSpan=label_ids_noSpan,
                input_ids_noSpan=input_ids_noSpan,
                raw_image_data=raw_image_data,
                mask_noSpan=mask_noSpan,
                segment_ids_noSpan=segment_ids_noSpan,
                span_ner_labels=span_ner_labels,
                span_token_level_mask=span_token_level_mask,
                span_start=span_start,
                span_end=span_end,
                image_ids=image_ids,
                label_ids_crf13=label_ids_crf13
                ))
        unique_id += 1

    print("Other_num is : {}".format(Other_num))
    return features


RawSpanResult = collections.namedtuple("RawSpanResult",
                                       ["unique_id", "start_logits", "end_logits"])

RawSpanCollapsedResult = collections.namedtuple("RawSpanCollapsedResult",
                                       ["unique_id", "neu_start_logits", "neu_end_logits", "pos_start_logits", "pos_end_logits",
                                        "neg_start_logits", "neg_end_logits"])

RawBIOResult = collections.namedtuple("RawBIOResult", ["unique_id", "bio_pred"])

RawBIOClsResult = collections.namedtuple("RawBIOClsResult", ["unique_id", "start_indexes", "end_indexes", "bio_pred", "span_masks"])

RawFinalResult = collections.namedtuple("RawFinalResult",
                                        ["unique_id", "start_indexes", "end_indexes", "cls_pred", "span_masks"])

def ts2start_end(ts_tag_sequence):
    starts, ends = [], []
    n_tag = len(ts_tag_sequence)
    prev_pos, prev_sentiment = '$$$', '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag =='T-NEG-B' or cur_ts_tag == 'T-POS-B' or cur_ts_tag == 'T-NEU-B':
            starts.append(i)
            if prev_pos !='O' and prev_pos !='$$$':
                ends.append(i-1)
            prev_pos=cur_ts_tag
        elif cur_ts_tag =='O':
            if prev_pos !='O' and prev_pos !='$$$':
                ends.append(i-1)
            prev_pos=cur_ts_tag
        elif cur_ts_tag == 'T-NEG' or cur_ts_tag == 'T-POS' or cur_ts_tag == 'T-NEU':
            prev_pos = cur_ts_tag
        else:
            raise Exception('!! find error tag:{}'.format(cur_ts_tag))
        if prev_pos!='O' and i == n_tag-1:
            ends.append(n_tag-1)

    assert len(starts) == len(ends)
    return starts,ends

def ts2polarity(words, ts_tag_sequence, starts, ends):
    polarities = []
    for start, end in zip(starts, ends):
        cur_ts_tag = ts_tag_sequence[start]
        cur_pos, cur_sentiment, = cur_ts_tag.split('-')[:2]
        assert cur_pos == 'T'
        prev_sentiment = cur_sentiment
        if start < end:
            for idx in range(start, end + 1):
                cur_ts_tag = ts_tag_sequence[idx]
                cur_pos, cur_sentiment = cur_ts_tag.split('-')[:2]
                assert cur_pos == 'T'
                assert cur_sentiment == prev_sentiment, (words, ts_tag_sequence, start, end)
                prev_sentiment = cur_sentiment
        polarities.append(cur_sentiment)
    return polarities


def pos2term(words, starts, ends):
    term_texts = []
    for start, end in zip(starts, ends):
        term_texts.append(' '.join(words[start:end+1]))
    return term_texts

def image_process(image_id, EvalOrNot):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if not EvalOrNot:

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    image = Image.open(image_id).convert('RGB')


    image = transform(image)
    return image

def convert_absa_data(EvalOrNot, dataset,args,verbose_logging=False):
    examples = []
    n_records = len(dataset)
    ss=sum([1 if len(item['image_ids']) >1 else 0 for item in dataset])
    assert ss == 0
    count=0

    empty_image_num = 0
    unidentify_image_num = 0

    for i in range(n_records):
        words = dataset[i]['words']
        image_ids = dataset[i]['image_ids']
        ner_labels = dataset[i]['ner_labels']
        base_image_path=args.image_path
        # image_id = image_ids[0]

        if image_ids[0] == 'O_504.jpg':
            print("O_504")


        image_path=base_image_path+image_ids[0]

        image_ids.append(image_path)
        image_ids.append(1) #默認所有的圖像文件都是非空
        image_ids.append(1)  # 默認所有的圖像文件都是可识别的

        size = os.path.getsize(image_path)


        if size == 0:#判斷圖像文件是否為空
            # print('文件是空的')
            empty_image_num+=1
            image_ids[2] = 0

            if image_ids[2] == 0:
                # image_ids[0] = "blank_image.jpg"
                image_ids[1] = "./my_data/blank_image.jpg"

        if image_ids[2] != 0:

            try:
                image_try = Image.open(image_ids[1])
            except:
                image_ids[3] = 0 # 判断出图像文件是不可识别的
                unidentify_image_num+=1

                if image_ids[3] == 0:
                    # image_ids[0] = "blank_image.jpg"
                    image_ids[1] = "./my_data/blank_image.jpg"

        if 1:#  use image cache or not
            cache_dir = args.cache_dir

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            if base_image_path.split('/')[-2][10]=='5':
                cache_path=cache_dir+'tw15_img/'+image_ids[0][:-4]+'.tch'
            elif base_image_path.split('/')[-2][10]=='7':
                cache_path=cache_dir+'tw17_img/'+image_ids[0][:-4]+'.tch'
            else:
                raise ValueError('image path error')
            if os.path.exists(cache_path):
                raw_image_data=torch.load(cache_path)
            else:
                try:
                    raw_image_data = image_process(image_path, EvalOrNot)# tensor
                except:
                    count+=1
                    # image = Image.open(image_path).convert('RGB')
                    print('error images:{},img_id{}'.format(count,image_ids[0]))
                    raw_image_data = image_process(base_image_path+'17_06_4705.jpg', EvalOrNot)
                torch.save(raw_image_data,cache_path)



        example = SemEvalExample(str(i), words,image_ids,raw_image_data,ner_labels)
        examples.append(example)
        if i < 50 and verbose_logging:
            print(example)

    return examples

def read_absa_data(args, path, X_files, Y_files, P_files, _imgdir):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []

    for index in range(len(X_files)):
        record = {}
        with open(X_files[index], "r", encoding="utf-8") as fr:
            s = fr.readline().split("\t")

        with open(Y_files[index], "r", encoding="utf-8") as fr:
            l = fr.readline().split("\t")

        with open(P_files[index], "r", encoding="utf-8") as fr:
            imgid = fr.readline()
            imgid+=".jpg"

        record['words'] = s.copy()
        record['image_ids'] = [imgid].copy()
        record['ner_labels'] = l.copy()
        dataset.append(record)

    return dataset