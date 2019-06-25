# Authored by Wenlong Zhao using as reference BERT finetune runner examples which
# have the following copyright claim and license.
#
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
import apex

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(filename = 'mylog.log',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class RaceExample(object):
    """A single training/test example for the RACE dataset."""
    '''
    For RACE dataset:
    race_id: data id
    article: article
    question: question
    option_0/1/2/3: option_0/1/2/3
    label: true answer
    '''
    def __init__(self,
                 race_id,
                 article,
                 question,
                 option_0,
                 option_1,
                 option_2,
                 option_3,
                 label = None):
        self.race_id = race_id
        self.article = article
        self.question = question
        self.options = [
            option_0,
            option_1,
            option_2,
            option_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: %s" % self.race_id,
            "article: %s" % self.article,
            "question: %s" % self.question,
            "option_0: %s" % self.options[0],
            "option_1: %s" % self.options[1],
            "option_2: %s" % self.options[2],
            "option_3: %s" % self.options[3],
        ]

        if self.label is not None:
            l.append("label: %s" % self.label)

        return ", ".join(l)



## paths is a list containing all paths/directories
def read_race_examples(paths):
    examples = []
    for path in paths:
        filenames = glob.glob(path+"/*txt")
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fpr:
                data_raw = json.load(fpr)
                article = data_raw['article']
                ## for each qn
                for i in range(len(data_raw['answers'])):
                    truth = ord(data_raw['answers'][i]) - ord('A')
                    question = data_raw['questions'][i]
                    options = data_raw['options'][i]
                    examples.append(
                        RaceExample(
                            race_id = filename+'-'+str(i),
                            article = article,
                            question = question,

                            option_0 = options[0],
                            option_1 = options[1],
                            option_2 = options[2],
                            option_3 = options[3],
                            label = truth))
                
    return examples 



class InputFeatures(object):
    def __init__(self,
             example_id,
             choices_features,
             label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
            
            
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # RACE is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # The input will be like:
    # [CLS] Article [SEP] Question [SEP] Option [SEP]
    # for each option 
    # 
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        if example_index % 2000 == 0: print(example_index, "examples converted to features.")
        article_tokens = tokenizer.tokenize(example.article)
        question_tokens = tokenizer.tokenize(example.question)

        choices_features = []
        for option_index, option in enumerate(example.options):
            option_tokens = tokenizer.tokenize(option)
            truncated_article_length = min(len(article_tokens),
                                max_seq_length - 4 - len(question_tokens) - len(option_tokens))
            tokens = ["[CLS]"] + article_tokens[:truncated_article_length] + \
                    ["[SEP]"] + question_tokens + ["[SEP]"] + option_tokens + ["[SEP]"]

            segment_ids = [0] * (truncated_article_length + 2) + \
                        [1] * (len(question_tokens) + len(option_tokens) + 2)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            
            # feature representation for four choices
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        ## display some example
#         if example_index < 1:
#             logger.info("*** Example ***")
#             logger.info(f"race_id: {example.race_id}")
#             for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
#                 logger.info(f"choice: {choice_idx}")
#                 logger.info(f"tokens: {' '.join(tokens)}")
#                 logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
#                 logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
#                 logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
#             if is_training:
#                 logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id = example.race_id,
                choices_features = choices_features,
                label = label
            )
        )
        
    print(len(examples), "examples all converted to features.")
    
    return features



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    # field: tokens, input_ids, input_mask, segment_ids
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Batch size for updates during training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--load_from_epoch",
                        default=None,
                        type=str,
                        help="The epoch of the model to load.")
    parser.add_argument("--start_train_epoch",
                        default=0.0,
                        type=float,
                        help="Start with a specified training epoch.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--curr_global_step",
                        default=0.0,
                        type=float,
                        help="Start with a specified global step.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()


    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train and args.load_from_epoch == None and \
            os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)


    ## Prepare device
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(args.seed)         

            
    ## Prepare model
    if args.load_from_epoch == None:
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
            num_choices=4)
    else:
        model_state_dict = torch.load(os.path.join( args.output_dir, "model_%sepoch" % 
                                                   args.load_from_epoch ))
        model = BertForMultipleChoice.from_pretrained(args.bert_model,
                state_dict=model_state_dict,
                num_choices=4)
    if args.fp16:
        model.half()

    model.to(device)
    
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

        
    ## Prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    
    ''' Training '''
    if args.do_train:
        ## Prepare training materials and log training information
        ## Also prepare evaluation materials. We evaluate on the dev set after every epoch
        global_step = args.curr_global_step  # Counts num of updates
            
        train_dir = os.path.join(args.data_dir, 'train')
        train_examples = read_race_examples([train_dir+'/high', train_dir+'/middle'])
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        
        dev_dir = os.path.join(args.data_dir, 'dev')
        eval_examples = read_race_examples([dev_dir+'/high', dev_dir+'/middle'])
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            

        ## Prepare and log training information
        num_train_updates = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps
            ) * args.num_train_epochs
        if args.local_rank != -1:
            t_total = num_train_updates // torch.distributed.get_world_size()
        else: t_total = num_train_updates
            
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size*args.gradient_accumulation_steps)
        logger.info("  Sum num updates over nodes = %d", num_train_updates)
        
        
        ## Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        
        ''' Training '''
        for ep in range(int(args.start_train_epoch)-1, int(args.num_train_epochs)):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            logger.info("Training Epoch: {}/{}".format(ep+1, int(args.num_train_epochs)))
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step/t_total, args.warmup_proportion
                            )
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    ## log the process
                    if global_step%100 == 0:
                        logger.info("Training loss: {}, global step: {}".format(
                            tr_loss/nb_tr_steps, global_step))

                        
            ''' Save a trained model by the end of each epoch '''
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "model_%sepoch"%str(ep+1)) )
            
            
            ''' Evaluate on dev set at the end of the epoch '''
            logger.info("***** Running evaluation: Dev *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            result = {'dev_eval_loss': eval_loss,
                      'dev_eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'training loss': tr_loss/nb_tr_steps}

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Dev results *****")
                writer.write("Epoch: %s\n" % str(ep+1))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n")


    ''' Testing '''
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        if not args.do_train:
            ## Load a trained model that you have fine-tuned
            ## use this part if you want to load the trained model
            model_state_dict = torch.load(os.path.join( args.output_dir, "model_%sepoch" % args.num_train_epochs ))
            model = BertForMultipleChoice.from_pretrained(args.bert_model,
                state_dict=model_state_dict,
                num_choices=4)
            model.to(device)
        
        test_dir = os.path.join(args.data_dir, 'test')
        test_high = [test_dir + '/high']
        test_middle = [test_dir + '/middle']
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")


        ## test high 
        eval_examples = read_race_examples(test_high)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation: test high *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_ids = torch.tensor([int(os.path.basename(f.example_id)[:-6]) for f in eval_features], dtype=torch.int)  # "RACE/test/high/xxxx.txt-3"
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        analyze_high_file = open("analyze_high.txt","a+")
        n_high_sample = 0
        
        high_eval_loss, high_eval_accuracy = 0, 0
        high_nb_eval_steps, high_nb_eval_examples = 0, 0
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, ids = batch

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            high_eval_loss += tmp_eval_loss.mean().item()
            high_eval_accuracy += tmp_eval_accuracy

            high_nb_eval_examples += input_ids.size(0)
            high_nb_eval_steps += 1
            
            # print cases of incorrect predictions to file
            ids = ids.to('cpu').numpy()
            input_ids = input_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            correctness = outputs == label_ids
            for i in range(len(label_ids)):
                #if correctness[i] == 1:
                if n_high_sample < 300:
                    analyze_high_file.write("race_id:"+str(ids[i])+"\n\n")
                    analyze_high_file.write(
                        'answer: ' + str(label_ids[i]) +
                        '\nprediction: ' + str(outputs[i]) + "\n\n")
                    for choice in range(4):
                        analyze_high_file.write(
                            "choice: "+str(choice)+"\n")
                        analyze_high_file.write( ' '.join(
                            tokenizer.convert_ids_to_tokens(
                                input_ids[i][choice]))+'\n\n')
                    n_high_sample = n_high_sample + 1
                    
        analyze_high_file.close()

        eval_loss = high_eval_loss / high_nb_eval_steps
        eval_accuracy = high_eval_accuracy / high_nb_eval_examples

        result = {'high_eval_loss': eval_loss,
                  'high_eval_accuracy': eval_accuracy}

        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


        ## test middle
        eval_examples = read_race_examples(test_middle)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation: test middle *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_ids = torch.tensor([int(os.path.basename(f.example_id)[:-6]) for f in eval_features], dtype = torch.int)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        analyze_middle_file = open("analyze_middle.txt","a+")
        n_middle_sample = 0
        
        middle_eval_loss, middle_eval_accuracy = 0, 0
        middle_nb_eval_steps, middle_nb_eval_examples = 0, 0
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, ids = batch

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            middle_eval_loss += tmp_eval_loss.mean().item()
            middle_eval_accuracy += tmp_eval_accuracy

            middle_nb_eval_examples += input_ids.size(0)
            middle_nb_eval_steps += 1
            
            # print cases of incorrect predictions to file
            ids = ids.to('cpu').numpy()
            input_ids = input_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            correctness = outputs == label_ids
            for i in range(len(label_ids)):
                #if correctness[i] == 1:
                if n_middle_sample < 300:
                    analyze_middle_file.write("race_id:"+str(ids[i])+"\n\n")
                    analyze_middle_file.write(
                        'answer: ' + str(label_ids[i]) +
                        '\nprediction: ' + str(outputs[i]) + "\n\n")
                    for choice in range(4):
                        analyze_middle_file.write(
                            "choice: "+str(choice)+"\n")
                        analyze_middle_file.write( ' '.join(
                            tokenizer.convert_ids_to_tokens(
                                input_ids[i][choice]))+"\n\n")
                    n_middle_sample = n_middle_sample + 1
                    
        analyze_middle_file.close()

        eval_loss = middle_eval_loss / middle_nb_eval_steps
        eval_accuracy = middle_eval_accuracy / middle_nb_eval_examples

        result = {'middle_eval_loss': eval_loss,
                  'middle_eval_accuracy': eval_accuracy}

        
        with open(output_eval_file, "a+") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


        ## all test
        eval_loss = (middle_eval_loss + high_eval_loss) / (middle_nb_eval_steps + high_nb_eval_steps)
        eval_accuracy = (middle_eval_accuracy + high_eval_accuracy) / (middle_nb_eval_examples + high_nb_eval_examples)

        result = {'overall_eval_loss': eval_loss,
                  'overall_eval_accuracy': eval_accuracy}

        with open(output_eval_file, "a+") as writer:
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":
    main()