# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SemEval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from model.sentiment_modeling import MSFMNER
from absa.utils import read_absa_data, convert_absa_data, convert_examples_to_features
from absa.run_base import copy_optimizer_params_to_model, set_optimizer_params_grad, prepare_optimizer, post_process_loss

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def mmreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label= []
    auxlabel = []
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1]+'.jpg'
            continue
        if line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) >0:
        data.append((sentence,label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)

    print("The number of samples: "+ str(len(data)))
    print("The number of images: "+ str(len(imgs)))
    return data, imgs, auxlabels


def read_absa_data(lines, imgs, auxlabels):
    dataset = []
    for i, (sentence, label) in enumerate(lines):
        record = {}
        # guid = "%s-%s" % (set_type, i)
        text_a = ' '.join(sentence)
        text_b = None
        img_id = imgs[i]
        label = label
        auxlabel = auxlabels[i]

        record['words'] = sentence.copy()
        record['image_ids'] = [img_id].copy()
        record['ner_labels'] = label.copy()
        dataset.append(record)
    return dataset


def read_train_data(args, tokenizer, logger):
    data, imgs, auxlabels = mmreadfile(os.path.join(args.data_dir_ner, "train.txt"))


    train_set = read_absa_data(data, imgs, auxlabels)

    train_examples = convert_absa_data(0, dataset=train_set, args=args,
                                       verbose_logging=args.verbose_logging)  # transform  the data into the example class
    train_features = convert_examples_to_features(args, 0, train_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    logger.info("Batch size = %d", args.train_batch_size)
    logger.info("Num steps = %d", num_train_steps)

    all_label_ids_noSpan = torch.tensor([f.label_ids_noSpan for f in train_features], dtype=torch.long)
    all_input_ids_noSpan = torch.tensor([f.input_ids_noSpan for f in train_features], dtype=torch.long)
    all_raw_image_data = torch.stack([f.raw_image_data for f in train_features])  # ,dtype=torch.float
    all_mask_noSpan = torch.tensor([f.mask_noSpan for f in train_features], dtype=torch.uint8)
    all_segment_ids_noSpan = torch.tensor([f.segment_ids_noSpan for f in train_features], dtype=torch.long)
    all_span_ner_labels = torch.tensor([f.span_ner_labels for f in train_features], dtype=torch.long)
    all_span_token_level_mask = torch.tensor([f.span_token_level_mask for f in train_features], dtype=torch.uint8)
    all_span_start = torch.tensor([f.span_start for f in train_features])
    all_span_end = torch.tensor([f.span_end for f in train_features])
    all_label_ids_crf13 = torch.tensor([f.label_ids_crf13 for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_label_ids_noSpan, all_input_ids_noSpan, all_raw_image_data,
                               all_mask_noSpan, all_segment_ids_noSpan, all_span_ner_labels,
                               all_span_token_level_mask, all_span_start, all_span_end, all_label_ids_crf13)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    return train_examples, train_features, train_dataloader, num_train_steps


def read_eval_data(args, tokenizer, logger):

    data, imgs, auxlabels = mmreadfile(os.path.join(args.data_dir_ner, "valid.txt"))
    eval_set = read_absa_data(data, imgs, auxlabels)

    eval_examples = convert_absa_data(0, dataset=eval_set, args=args,
                                       verbose_logging=args.verbose_logging)  # transform  the data into the example class
    eval_features = convert_examples_to_features(args, 0, eval_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    logger.info("Batch size = %d", args.predict_batch_size)

    all_label_ids_noSpan = torch.tensor([f.label_ids_noSpan for f in eval_features], dtype=torch.long)
    all_input_ids_noSpan = torch.tensor([f.input_ids_noSpan for f in eval_features], dtype=torch.long)
    all_raw_image_data = torch.stack([f.raw_image_data for f in eval_features])  # ,dtype=torch.float
    all_mask_noSpan = torch.tensor([f.mask_noSpan for f in eval_features], dtype=torch.uint8)
    all_segment_ids_noSpan = torch.tensor([f.segment_ids_noSpan for f in eval_features], dtype=torch.long)
    all_span_ner_labels = torch.tensor([f.span_ner_labels for f in eval_features], dtype=torch.long)
    all_span_token_level_mask = torch.tensor([f.span_token_level_mask for f in eval_features], dtype=torch.uint8)
    all_span_start = torch.tensor([f.span_start for f in eval_features])
    all_span_end = torch.tensor([f.span_end for f in eval_features])
    all_label_ids_crf13 = torch.tensor([f.label_ids_crf13 for f in eval_features], dtype=torch.long)


    eval_data = TensorDataset(all_label_ids_noSpan, all_input_ids_noSpan, all_raw_image_data,
                               all_mask_noSpan, all_segment_ids_noSpan, all_span_ner_labels,
                               all_span_token_level_mask, all_span_start, all_span_end, all_label_ids_crf13)

    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    return eval_examples, eval_features, eval_dataloader


def read_test_data(args, tokenizer, logger):

    data, imgs, auxlabels = mmreadfile(os.path.join(args.data_dir_ner, "test.txt"))

    test_set = read_absa_data(data, imgs, auxlabels)

    test_examples = convert_absa_data(0, dataset=test_set, args=args,
                                       verbose_logging=args.verbose_logging)  # transform  the data into the example class
    test_features = convert_examples_to_features(args, 0, test_examples, tokenizer, args.max_seq_length,
                                                  args.verbose_logging, logger)


    logger.info("Num orig examples = %d", len(test_examples))
    logger.info("Num split features = %d", len(test_features))
    logger.info("Batch size = %d", args.predict_batch_size)


    all_label_ids_noSpan = torch.tensor([f.label_ids_noSpan for f in test_features], dtype=torch.long)
    all_input_ids_noSpan = torch.tensor([f.input_ids_noSpan for f in test_features], dtype=torch.long)
    all_raw_image_data = torch.stack([f.raw_image_data for f in test_features])  # ,dtype=torch.float
    all_mask_noSpan = torch.tensor([f.mask_noSpan for f in test_features], dtype=torch.uint8)
    all_segment_ids_noSpan = torch.tensor([f.segment_ids_noSpan for f in test_features], dtype=torch.long)
    all_span_ner_labels = torch.tensor([f.span_ner_labels for f in test_features], dtype=torch.long)
    all_span_token_level_mask = torch.tensor([f.span_token_level_mask for f in test_features], dtype=torch.uint8)
    all_span_start = torch.tensor([f.span_start for f in test_features])
    all_span_end = torch.tensor([f.span_end for f in test_features])
    all_label_ids_crf13 = torch.tensor([f.label_ids_crf13 for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_label_ids_noSpan, all_input_ids_noSpan, all_raw_image_data,
                               all_mask_noSpan, all_segment_ids_noSpan, all_span_ner_labels,
                               all_span_token_level_mask, all_span_start, all_span_end, all_label_ids_crf13)

    if args.local_rank == -1:
        test_sampler = SequentialSampler(test_data)
    else:
        test_sampler = DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)
    return test_examples, test_features, test_dataloader


def run_train_epoch(args, epoch, global_step, model, param_optimizer,
                    train_examples, train_features, train_dataloader,
                    eval_examples, eval_features, eval_dataloader,
                    test_examples, test_features, test_dataloader,
                    optimizer, n_gpu, device, logger, log_path, save_path,
                    save_checkpoints_steps, start_save_steps, best_f1):
    running_loss, count = 0.0, 0
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self

        label_ids_noSpan, input_ids_noSpan, raw_image_data, mask_noSpan,\
            segment_ids_noSpan, span_ner_labels, span_token_level_mask, span_start, span_end, label_ids_crf13 = batch

        label_ids_noSpan = label_ids_noSpan.to(device, non_blocking=True)
        input_ids_noSpan = input_ids_noSpan.to(device, non_blocking=True)
        raw_image_data = raw_image_data.to(device, non_blocking=True)
        mask_noSpan = mask_noSpan.to(device, non_blocking=True)
        segment_ids_noSpan = segment_ids_noSpan.to(device, non_blocking=True)
        span_ner_labels = span_ner_labels.to(device, non_blocking=True)
        span_token_level_mask = span_token_level_mask.to(device, non_blocking=True)
        span_start = span_start.to(device, non_blocking=True)
        span_end = span_end.to(device, non_blocking=True)
        label_ids_crf13 = label_ids_crf13.to(device, non_blocking=True)



        loss = model(args, label_ids_noSpan, input_ids_noSpan, raw_image_data, mask_noSpan,
                     segment_ids_noSpan, span_ner_labels, span_token_level_mask, span_start, span_end, label_ids_crf13, train=True)

        # torch.cuda.empty_cache()
        loss = post_process_loss(args, n_gpu, loss)
        loss.backward()
        running_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 or args.optimize_on_cpu:
                if args.fp16 and args.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        param.grad.data = param.grad.data / args.loss_scale
                is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    args.loss_scale = args.loss_scale / 2
                    model.zero_grad()
                    continue
                optimizer.step()
                copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
            else:
                optimizer.step()
            model.zero_grad()
            global_step += 1
            count += 1

            if global_step % save_checkpoints_steps == 0 and count != 0:
                logger.info("step: {}, loss: {:.4f}".format(global_step, running_loss / count))
            if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save model
                logger.info("***** Running evaluation *****")
                model.eval()
                # metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger,
                #                    write_pred=True)
                metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
                print(metrics)

                f = open(log_path, "a")


                print("PER_p: {:.16f}, PER_r: {:.16f}, PER_f1: {:.16f} (PER_common: {}, PER_retrieved: {}, PER_relevant: {}); LOC_p: {:.16f}, LOC_r: {:.16f}, LOC_f1: {:.16f} (LOC_common: {}, LOC_retrieved: {}, LOC_relevant: {}); ORG_p: {:.16f}, ORG_r: {:.16f}, ORG_f1: {:.16f} (ORG_common: {}, ORG_retrieved: {}, ORG_relevant: {}); OTHER_p: {:.16f}, OTHER_r: {:.16f}, OTHER_f1: {:.16f} (OTHER_common: {}, OTHER_retrieved: {}, OTHER_relevant: {}); total_p: {:.16f}, total_r: {:.16f}, total_f1: {:.16f} (total_common: {}, total_retrieved: {}, total_relevant: {});"
                      .format(metrics['PER_p'], metrics['PER_r'], metrics['PER_f1'], metrics['PER_common'], metrics['PER_retrieved'], metrics['PER_relevant'], metrics['LOC_p'], metrics['LOC_r'], metrics['LOC_f1'], metrics['LOC_common'], metrics['LOC_retrieved'], metrics['LOC_relevant'], metrics['ORG_p'], metrics['ORG_r'], metrics['ORG_f1'], metrics['ORG_common'], metrics['ORG_retrieved'], metrics['ORG_relevant'], metrics['OTHER_p'], metrics['OTHER_r'], metrics['OTHER_f1'], metrics['OTHER_common'], metrics['OTHER_retrieved'], metrics['OTHER_relevant'], metrics['total_p'], metrics['total_r'], metrics['total_f1'], metrics['total_common'], metrics['total_retrieved'], metrics['total_relevant']),
                      file=f)

                print(" ", file=f)
                f.close()
                running_loss, count = 0.0, 0
                model.train()
                # detach_submodel(model.pretrain_model)
                if metrics['total_f1'] > best_f1:
                    best_f1 = metrics['total_f1']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': global_step
                    }, save_path)

                if args.debug:
                    break
    return global_step, model, best_f1

def evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):

    total_ner_label_num  = 0
    total_ner_label_span_masked_num = 0

    total_PER_label = 0
    total_LOC_label = 0
    total_ORG_label = 0
    total_OTHER_label = 0

    total_PER_perdict = 0
    total_LOC_perdict = 0
    total_ORG_perdict = 0
    total_OTHER_perdict = 0

    total_PER_perdict_correct = 0
    total_LOC_perdict_correct = 0
    total_ORG_perdict_correct = 0
    total_OTHER_perdict_correct = 0

    total_true_PER_num = 0
    total_true_LOC_num = 0
    total_true_ORG_num = 0
    total_true_OTHER_num = 0

    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)

        label_ids_noSpan, input_ids_noSpan, raw_image_data, mask_noSpan,\
            segment_ids_noSpan, span_ner_labels, span_token_level_mask, span_start, span_end, label_ids_crf13= batch

        label_ids_noSpan = label_ids_noSpan.to(device, non_blocking=True)
        input_ids_noSpan = input_ids_noSpan.to(device, non_blocking=True)
        raw_image_data = raw_image_data.to(device, non_blocking=True)
        mask_noSpan = mask_noSpan.to(device, non_blocking=True)
        segment_ids_noSpan = segment_ids_noSpan.to(device, non_blocking=True)
        span_ner_labels = span_ner_labels.to(device, non_blocking=True)
        span_token_level_mask = span_token_level_mask.to(device, non_blocking=True)
        span_start = span_start.to(device, non_blocking=True)
        span_end = span_end.to(device, non_blocking=True)
        label_ids_crf13 = label_ids_crf13.to(device, non_blocking=True)

        with torch.no_grad():

            ner_span_outputs, span_ner_labels_masked = model(args, label_ids_noSpan, input_ids_noSpan, raw_image_data, mask_noSpan,
                     segment_ids_noSpan, span_ner_labels, span_token_level_mask, span_start, span_end, label_ids_crf13, train=False)

            _, aspect_span_predictions = torch.max(ner_span_outputs, dim=1)

        PER = 1
        LOC = 2
        ORG = 3
        OTHER = 4


        for i in range(len(label_ids_noSpan)):
            true_seq = label_ids_noSpan[i]
            for num in range(len(true_seq)):
                if true_seq[num] == 1:
                    total_true_PER_num += 1
                    total_ner_label_num += 1

                if true_seq[num] == 3:
                    total_true_LOC_num += 1
                    total_ner_label_num += 1

                if true_seq[num] == 5:
                    total_true_ORG_num += 1
                    total_ner_label_num += 1

                if true_seq[num] == 7:
                    total_true_OTHER_num += 1
                    total_ner_label_num += 1


        for i in range(len(span_ner_labels_masked)):
            true_seq = span_ner_labels_masked[i]
            if true_seq != 0:
                total_ner_label_span_masked_num += 1

        for i in range(len(span_ner_labels_masked)):
            if span_ner_labels_masked[i] == PER:
                total_PER_label += 1
            if span_ner_labels_masked[i] == LOC:
                total_LOC_label += 1
            if span_ner_labels_masked[i] == ORG:
                total_ORG_label += 1
            if span_ner_labels_masked[i] == OTHER:
                total_OTHER_label += 1

        for i in range(len(aspect_span_predictions)):
            if aspect_span_predictions[i] == PER:
                total_PER_perdict += 1
            if aspect_span_predictions[i] == LOC:
                total_LOC_perdict += 1
            if aspect_span_predictions[i] == ORG:
                total_ORG_perdict += 1
            if aspect_span_predictions[i] == OTHER:
                total_OTHER_perdict += 1

        for i in range(len(aspect_span_predictions)):
            if aspect_span_predictions[i] == PER and aspect_span_predictions[i] == span_ner_labels_masked[i]:
                total_PER_perdict_correct += 1
            if aspect_span_predictions[i] == LOC and aspect_span_predictions[i] == span_ner_labels_masked[i]:
                total_LOC_perdict_correct += 1
            if aspect_span_predictions[i] == ORG and aspect_span_predictions[i] == span_ner_labels_masked[i]:
                total_ORG_perdict_correct += 1
            if aspect_span_predictions[i] == OTHER and aspect_span_predictions[i] == span_ner_labels_masked[i]:
                total_OTHER_perdict_correct += 1

    PER_common = total_PER_perdict_correct
    PER_retrieved = total_PER_perdict
    PER_relevant = total_true_PER_num

    PER_p = PER_common / PER_retrieved if PER_retrieved > 0 else 0.
    PER_r = PER_common / PER_relevant
    PER_f1 = (2 * PER_p * PER_r) / (PER_p + PER_r) if PER_p > 0 and PER_r > 0 else 0.


    LOC_common = total_LOC_perdict_correct
    LOC_retrieved = total_LOC_perdict
    LOC_relevant = total_true_LOC_num

    LOC_p = LOC_common / LOC_retrieved if LOC_retrieved > 0 else 0.
    LOC_r = LOC_common / LOC_relevant
    LOC_f1 = (2 * LOC_p * LOC_r) / (LOC_p + LOC_r) if LOC_p > 0 and LOC_r > 0 else 0.


    ORG_common = total_ORG_perdict_correct
    ORG_retrieved = total_ORG_perdict
    ORG_relevant = total_true_ORG_num

    ORG_p = ORG_common / ORG_retrieved if ORG_retrieved > 0 else 0.
    ORG_r = ORG_common / ORG_relevant
    ORG_f1 = (2 * ORG_p * ORG_r) / (ORG_p + ORG_r) if ORG_p > 0 and ORG_r > 0 else 0.


    OTHER_common = total_OTHER_perdict_correct
    OTHER_retrieved = total_OTHER_perdict
    OTHER_relevant = total_true_OTHER_num

    OTHER_p = OTHER_common / OTHER_retrieved if OTHER_retrieved > 0 else 0.
    OTHER_r = OTHER_common / OTHER_relevant
    OTHER_f1 = (2 * OTHER_p * OTHER_r) / (OTHER_p + OTHER_r) if OTHER_p > 0 and OTHER_r > 0 else 0.



    total_common = total_OTHER_perdict_correct + total_ORG_perdict_correct + total_LOC_perdict_correct + total_PER_perdict_correct
    total_retrieved = total_OTHER_perdict + total_ORG_perdict + total_LOC_perdict + total_PER_perdict

    total_p = total_common / total_retrieved if total_retrieved > 0 else 0.
    total_r = total_common / total_ner_label_num
    total_f1 = (2 * total_p * total_r) / (total_p + total_r) if total_p > 0 and total_r > 0 else 0.

    print("the total_ner_label_span_masked_num is : {}".format(total_ner_label_span_masked_num))

    return {'PER_p': PER_p, 'PER_r': PER_r, 'PER_f1': PER_f1, 'PER_common': PER_common, 'PER_retrieved': PER_retrieved, 'PER_relevant': PER_relevant,
            'LOC_p': LOC_p, 'LOC_r': LOC_r, 'LOC_f1': LOC_f1, 'LOC_common': LOC_common, 'LOC_retrieved': LOC_retrieved, 'LOC_relevant': LOC_relevant,
            'ORG_p': ORG_p, 'ORG_r': ORG_r, 'ORG_f1': ORG_f1, 'ORG_common': ORG_common, 'ORG_retrieved': ORG_retrieved, 'ORG_relevant': ORG_relevant,
            'OTHER_p': OTHER_p, 'OTHER_r': OTHER_r, 'OTHER_f1': OTHER_f1, 'OTHER_common': OTHER_common, 'OTHER_retrieved': OTHER_retrieved, 'OTHER_relevant': OTHER_relevant,
            'total_p': total_p, 'total_r': total_r, 'total_f1': total_f1, 'total_common': total_common, 'total_retrieved': total_retrieved, 'total_relevant': total_ner_label_num}


def main(args):

    if args.local_rank == -1 or args.no_cuda:
        if args.gpu_idx == -1:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu", 0)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu", args.gpu_idx)
        # n_gpu = torch.cuda.device_count()
        n_gpu = 1
    else:
        # device = torch.device("cuda:0", args.local_rank)
        device = torch.device("cuda", args.local_rank)

        n_gpu = 1

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("torch_version: {} device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        torch.__version__, device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config = 0

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info('output_dir: {}'.format(args.output_dir))

    save_path = os.path.join(args.output_dir, 'checkpoint.pth')
    log_path = os.path.join(args.output_dir, 'performance.txt')
    network_path = os.path.join(args.output_dir, 'network.txt')
    parameter_path = os.path.join(args.output_dir, 'parameter.txt')

    f = open(parameter_path, "w")
    for arg in sorted(vars(args)):
        print("{}: {}".format(arg, getattr(args, arg)), file=f)
    f.close()

    logger.info("***** Preparing model *****")
    # This MSFMNER is the AMLR model
    model = MSFMNER(args, bert_config)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if os.path.isfile(save_path):
        if args.do_predict:
            checkpoint = torch.load(save_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))


    f = open(network_path, "w")
    for n, param in model.named_parameters():
        print("name: {}, size: {}, dtype: {}, requires_grad: {}"
              .format(n, param.size(), param.dtype, param.requires_grad), file=f)
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)  # the numel means the number of params in a tensor
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: {}".format(total_trainable_params), file=f)
    print("Total parameters: {}".format(total_params), file=f)
    f.close()

    logger.info("***** Preparing data *****")
    train_examples, train_features, train_dataloader, num_train_steps = None, None, None, None
    eval_examples, eval_features, eval_dataloader = None, None, None
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if args.do_train:
        logger.info("***** Preparing training *****")
        train_examples, train_features, train_dataloader, num_train_steps = read_train_data(args, tokenizer, logger)
        logger.info("***** Preparing evaluation *****")
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)
        logger.info("***** Preparing testting *****")
        args.predict_file = 'test.txt'
        test_examples, test_features, test_dataloader = read_test_data(args, tokenizer, logger)

    logger.info("***** Preparing optimizer *****")

    optimizer, param_optimizer = prepare_optimizer(args, model, num_train_steps)

    global_step = 0
    # if os.path.isfile(save_path):
    #     checkpoint = torch.load(save_path, map_location='cpu')
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     step = checkpoint['step']
    #     logger.info("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
    #     global_step = step
    if args.do_train:
        logger.info("***** Running training *****")
        best_f1 = 0
        save_checkpoints_steps = int(num_train_steps / (5 * args.num_train_epochs))
        start_save_steps = int(num_train_steps * args.save_proportion)

        if args.debug:
            args.num_train_epochs = 1
            save_checkpoints_steps = 20
            start_save_steps = 0
        model.train()
        last_model_pp = None
        for epoch in range(int(args.num_train_epochs)):
            logger.info("***** Epoch: {} *****".format(epoch + 1))
            global_step, model, best_f1 = run_train_epoch(args, epoch, global_step, model, param_optimizer,
                                                          train_examples, train_features, train_dataloader,
                                                          eval_examples, eval_features, eval_dataloader,
                                                          test_examples, test_features, test_dataloader,
                                                          optimizer, n_gpu, device, logger, log_path, save_path,
                                                          save_checkpoints_steps, start_save_steps, best_f1)

    if args.do_predict:
        logger.info("***** Running prediction *****")
        # if eval_dataloader is None:
        args.predict_file = 'dev.txt'
        eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)
        args.predict_file = 'test.txt'
        test_examples, test_features, test_dataloader = read_test_data(args, tokenizer, logger)
        #  best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()

        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger)
        print(metrics)
        metrics_test = evaluate(args, model, device, test_examples, test_features, test_dataloader, logger)
        print(metrics_test)



if __name__ == '__main__':
    print("get into run_joint_span")
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='output/model_checkpoint/tw15/', type=str, required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--image_path", default="/home/tiggers/Newdisk/lep/data/Twitrer/Twitter/twitter2015_images/")

    parser.add_argument("--data_dir_ner",
                        default="./data/twitter2015",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--multi_head_num',
                        type=int,
                        default=8,
                        help="Number of MultiHeadAttention's head num.")

    # parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_predict", default=True, action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--predict_file", default='dev.txt', type=str, help="SemEval csv for prediction")

    parser.add_argument("--num_train_epochs", default=40.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_seq_length", default=50, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--gpu_idx",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")

    parser.add_argument('--seed',
                        type=int,
                        default=777,
                        help="random seed for initialization")
    parser.add_argument("--debug", default=False, action='store_true', help="Whether to run in debug mode.")

    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")

    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--learning_rate_pretrained", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--save_proportion", default=0.01, type=float,
                        help="Proportion of steps to save models for. E.g., 0.5 = 50% of training.")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")



    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether  to  perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--cache_dir", default="/home/tiggers/Newdisk/lep/data/image_cache_dir/")


    args = parser.parse_args()



    main(args)