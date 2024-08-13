# -*- coding: utf-8 -*-
import numpy as np
import torch
import json
from tqdm import tqdm
from copy import deepcopy
import zipfile
import wget
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel, AutoTokenizer
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch.backends.cudnn
import torch.cuda

from utils import *
from eval import *
from argparser import simple_parse_args
from data_structures import CQA_DATA, Simple_Feature
from dataloader import DataManager, Simple_DataLoader
from post_processing import *
from savings import *
from settings import *

args = simple_parse_args()


SEED = args.seed
def set_determenistic_mode(SEED, disable_cudnn=False):
  torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
  random.seed(SEED)                             # Set python seed for custom operators.
  rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
  np.random.seed(SEED)
  torch.cuda.manual_seed_all(SEED)              # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

  if not disable_cudnn:
    torch.backends.cudnn.benchmark = False    # Causes cuDNN to deterministically select an algorithm,
                                              # possibly at the cost of reduced performance
                                              # (the algorithm itself may be nondeterministic).
    torch.backends.cudnn.deterministic = True # Causes cuDNN to use a deterministic convolution algorithm,                                       # but may slow down performance.
                                              # It will not guarantee that your training process is deterministic
                                              # if you are using other libraries that may use nondeterministic algorithms
  else:
    torch.backends.cudnn.enabled = False # Controls whether cuDNN is enabled or not.
                                         # If you want to enable cuDNN, set it to True.
set_determenistic_mode(SEED)
def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)



source_directory = args.source_directory

# download quac files
wget.download(quac_train_url)
wget.download(quac_eval_url)
# read quac files
train_data = read_json(train_path)
eval_data = read_json(eval_path)

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertModel.from_pretrained(model_id)



"""# Examples"""

def make_examples(data, data_type, num_sample, clean_samples=True):
  examples = []
  each_file_size = 1000
  example_file_index = 0
  data_dir = f'examples/{data_type}/'

  for dialog_num, sample in enumerate(tqdm(data[ :num_sample], leave=False, position=0)):
    dialog_history = []
    dialog_container = []
    dialog = sample['paragraphs'][0]
    context = dialog['context']
    dialog_len = len(dialog['qas'])

    for q_num, qas in enumerate(dialog['qas']):
      history = []
      question = qas['question']
      human_answer = qas['orig_answer']
      qid = qas['id']
      answers = []
      
      if not question.endswith('?'):
        question += '?'
        
      if not question.endswith('?'):
        print(question)

      for answer in qas['answers']:
        answer_ = {}
        answer_['text'] = answer['text']
        answer_['start'] = answer['answer_start']
        answer_['end'] = answer['answer_start'] + len(answer['text'])
        answers.append(answer_)


      is_answerable = False if qas['answers'][0]['text'] == 'CANNOTANSWER' else True

      if not q_num == 0:
        history = deepcopy(dialog_history)

      cqa_example = CQA_DATA(question=question,
                             context=context,
                             history=history,
                             answers=answers,
                             human_answer=human_answer,
                             qid=qid,
                             q_num=q_num,
                             answer_start=answers[0]['start'],
                             answer_end=answers[0]['end'],
                             is_answerable=is_answerable)

      cqa_example = CleanSample(cqa_example).clean() if clean_samples else cqa_example
      examples.append(cqa_example)
      dialog_history.append(cqa_example)

    if (dialog_num + 1) % each_file_size == 0:
      filename = f'{data_type}_examples_' + str(example_file_index) + '.bin'
      save_data(examples, os.path.join(data_dir, filename))
      example_file_index += 1
      examples = []

  if examples != []:
    filename = f'{data_type}_examples_' + str(example_file_index) + '.bin'
    save_data(examples, os.path.join(data_dir, filename))

"""# Features"""

def make_features(data_type):
  data_dir = f'examples/{data_type}/'
  example_files = os.listdir(data_dir)
  example_files = example_files = sorted([os.path.join(data_dir, example_file) for example_file in example_files], key=lambda x: int(x.split('.')[0].split('_')[-1]))
  features_list = []
  features_dir = f'features/{data_type}/'
  max_history_to_consider = 11

  for file_index, filename in enumerate(example_files):
    examples = load_data(filename)
    for example in tqdm(examples, leave=False, position=0):
      example_features = []
      concatenated_question = []

      # concat history
      for hist in example.history[-max_history_to_consider:]:
        concatenated_question.append(hist.question)
        concatenated_question.append(hist.answers[0]['text'])

      # append current question to concatenated question
      concatenated_question.append(example.question)

      # make string out of concatenated question
      concatenated_question = ' '.join(concatenated_question)

      # tokenize current feature
      text_tokens = tokenizer(
          concatenated_question,
          example.cleaned_context,
          max_length=model.config.max_position_embeddings,
          padding='max_length',
          truncation='only_second',
          return_overflowing_tokens=True,
          return_offsets_mapping=True,
          stride=100)

      # find start and end of context
      for idx in range(len(text_tokens['input_ids'])):
        found_start = False
        found_end = False
        context_start = 0
        cintext_end = 511
        max_context_dict = {}

        for token_idx, token in enumerate(text_tokens['offset_mapping'][idx][1:]):
          if token[0] == 0 and token[1] == 0:
            context_start = token_idx + 2
            break

        for token_idx, token in enumerate(text_tokens['offset_mapping'][idx][context_start:]):
          if token[0] == 0 and token[1] == 0:
            context_end = token_idx + context_start - 1
            break

        chunk_offset_mapping = text_tokens['offset_mapping'][idx]
        for context_idx, data in enumerate(chunk_offset_mapping[context_start: context_end + 1]):
          max_context_dict[f'({data[0]},{data[1]})'] = min(context_idx, context_end - context_idx) + (context_end - context_start + 1) * .01

        # find and mark current question answer
        last_token = None
        for token_idx, token in enumerate(chunk_offset_mapping[context_start: context_end + 1]):
          if token[0] == example.cleaned_answer['start'] and not found_start:
            found_start = True
            start = token_idx + context_start

          elif last_token and last_token[0] < example.cleaned_answer['start'] and token[0] > example.cleaned_answer['start']:
            found_start = True
            start = (token_idx - 1) + context_start

          if token[1] == example.cleaned_answer['end'] and not found_end:
            found_end = True
            end = token_idx + context_start

          elif last_token and last_token[1] < example.cleaned_answer['end'] and token[1] > example.cleaned_answer['end'] and last_token:
            found_end = True
            end = token_idx + context_start
          last_token = token

        # add feature to features list
        if found_start and found_end and end < start:
          assert False, 'start and end do not match'

        # since there is no prediction we throw the example out (only when training)
        if ((not found_start) or (not found_end)) and data_type == 'train':
          continue


        # plausibility check
        if found_start or found_end:
          answer = example.cleaned_answer['text'].strip()
          generated_answer = example.cleaned_context[chunk_offset_mapping[start][0]: chunk_offset_mapping[end][1]]
          if answer.find(generated_answer) == -1:
            pass

        # mark history answers

        example_features.append(Simple_Feature(example.qid,
                                          idx,
                                          text_tokens['input_ids'][idx],
                                          text_tokens['attention_mask'][idx],
                                          text_tokens['token_type_ids'][idx],
                                          text_tokens['offset_mapping'][idx],
                                          max_context_dict,
                                          start,
                                          end,
                                          example.is_answerable,
                                          example.context,
                                          example.cleaned_context,
                                          context_start,
                                          context_end,
                                          example.answer_start,
                                          example.answer_end,
                                          example.answer))
      # create max context mask
      for feature_1 in example_features:
        max_context_mask = {}
        for key in list(feature_1.max_context_dict.keys()):
          max_context_mask[key] = True
          for feature_2 in example_features:
            if key in feature_2.max_context_dict:
              if feature_1.max_context_dict[key] < feature_2.max_context_dict[key]:
                max_context_mask[key] = False
        feature_1.max_context_mask = max_context_mask

        found_start = found_end = False
        start_mask = end_mask = 0
        # now compute span mask
        for key_idx, (key, value) in enumerate(feature_1.max_context_mask.items()):
          if key_idx == 0 and value:
            found_start = True
          elif value and not found_start:
            start_mask = key_idx
            found_start = True
          elif not value and found_start and not found_end:
            end_mask = key_idx
            found_end = True
          elif key_idx == len(feature_1.max_context_mask) - 1 and value and not found_end:
            end_mask = key_idx + 1
        feature_1.mask_span = [context_start + start_mask, context_start + end_mask]
      features_list.extend(example_features)

    filename = f'{data_type}_features_' + str(file_index) + '.bin'
    save_data(features_list, os.path.join(features_dir, filename))
    features_list = []


random.seed(SEED)
train_data = deepcopy(train_data['data'])
random.shuffle(train_data)
eval_test = deepcopy(eval_data['data'])
P = 0
eval_data, test_data = [], []
for x in eval_test:
  P += len(x['paragraphs'][0]['qas'])
num_eval = P / 2
Y = 0
for idx, x in enumerate(eval_test):
  eval_data.append(x)
  Y += len(x['paragraphs'][0]['qas'])
  if Y >= num_eval:
    break
test_data = deepcopy(eval_test[idx + 1: ])


if not files_here:
  make_examples(train_data, 'train', 50)
  make_examples(eval_data, 'eval', 100)
  make_examples(test_data, 'test', 100)
  make_features('train')
  make_features('eval')
  make_features('test')

  def wrtie_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

  wrtie_file({
      'data': eval_data
  }, orig_eval_file)

  wrtie_file({
      'data': test_data
  }, orig_test_file)




# QA Model
class QA_Model(nn.Module):

  def __init__(self, transformer, device):
    super(QA_Model, self).__init__()
    self.transformer = transformer
    self.start_end_head = nn.Linear(self.transformer.config.hidden_size, 2)
    nn.init.normal_(self.start_end_head.weight, mean=.0, std=.02)
    self.device = device

  def forward(self, x):
    for key in x:
      x[key] = x[key].to(device)
    # transformer output
    transformer_output = self.transformer(**x)
    start_end_logits = self.start_end_head(transformer_output.last_hidden_state)
    start_logits, end_logits = start_end_logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    return start_logits, end_logits


# setup saving settings
save_dir_prefix, checkpoint_dir, log_dir, meta_log_file, \
loss_log_file, mean_f1_file_eval, mean_f1_file_test, \
scores_eval, scores_test, checkpoint_available, current_checkpoint = build_save_settings(args)



"""# Train loop"""
epochs = 3
lr = 3e-5
beta_1 = .9
beta_2 = .999
eps = 1e-6
batch_size = args.batch_size
weight_decay = 0
accumulation_steps = args.accumulation_steps
accumulation_counter = 0
q_scores = dict()
h_f1 = 0

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
qa_model = QA_Model(model, device).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

loss_collection = []
train_dataloader = Simple_DataLoader(current_file=0, current_index=0, batch_size=batch_size, shuffle=True, data_type='train')
eval_dataloader = Simple_DataLoader(current_file=0, current_index=0, batch_size=1, shuffle=False, data_type='eval')
test_dataloader = Simple_DataLoader(current_file=0, current_index=0, batch_size=1, shuffle=False, data_type='test')
each_step_log = 100
start_epoch = 0
start_step = 0
current_file = 0
current_index = 0


optimization_steps = int(epochs * len(train_dataloader) / accumulation_steps)
epoch_steps = int(len(train_dataloader) / accumulation_steps)
warmup_ratio = .1
warmup_steps = int(optimization_steps * warmup_ratio)

optimizer = AdamW(qa_model.parameters(), lr=lr, betas=(beta_1,beta_2), eps=eps, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=optimization_steps)

# laod checkpoint if available
if checkpoint_available:
  print('loading checkpoint')
  start_epoch, optimizer_dict, scheduler_dict, qa_model_dict = load_checkpoint()
  # load state dicts
  qa_model.load_state_dict(qa_model_dict)
  optimizer.load_state_dict(optimizer_dict)
  scheduler.load_state_dict(scheduler_dict)

current_file_index_ = current_file
train_dataloader.reset_dataloader(current_file, current_index)
qa_model.train()
for epoch in range(start_epoch, epochs):
  train_step = 1
  acc_loss = 0
  log_step = 0

  for data in train_dataloader:
    if train_dataloader.data_manager.current_file_index != current_file_index_:
      current_file_index_ = train_dataloader.data_manager.current_file_index
      
    start_positions = data.pop('start_positions').to(device)
    end_positions = data.pop('end_positions').to(device)
    features = data.pop('features')
    start_logits, end_logits = qa_model(data)
    loss = (loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)) / 2
    loss = loss / accumulation_steps
    acc_loss += loss.item()
    loss.backward()
    accumulation_counter += 1

    if train_step % each_step_log == 0:
      print_loss(loss_collection, epoch, epochs, log_step + 1, epoch_steps)
      save_loss(loss_collection, epoch, epochs, log_step + 1, epoch_steps)
      loss_collection = []


    if accumulation_counter % accumulation_steps == 0:
      loss_collection.append(acc_loss)
      acc_loss = 0
      log_step += 1
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      torch.cuda.empty_cache()
      accumulation_counter = 0

    train_step += 1

  save_checkpoint(epoch + 1, checkpoint_dir, qa_model, optimizer, scheduler)
  qa_model.eval()
  print('-------------------- Evaluation --------------------')
  eval_p = EvalProcessOutput()
  with torch.no_grad():
    for step, data in enumerate(eval_dataloader):
      start_positions = data.pop('start_positions')
      end_positions = data.pop('end_positions')
      features = data.pop('features')
      start_logits, end_logits = qa_model(data)
      eval_p.process_feature_output(to_numpy(start_logits),
                                    to_numpy(end_logits),
                                    features)

  eval_p.process_output()
  filename = os.path.join(log_dir, f'val_{epoch + 1}_preds.json')
  write_to_file(eval_p.dialogs_answers, filename)
  res__ = run_eval(filename, orig_eval_file)
  if res__['f1'] > h_f1:
    h_f1 = res__['f1']
    best_checkpoint = f'checkpoint_{epoch + 1}'
  save_results(scores_eval, res__)
  print_mean(res__['turn_f1s'], mean_f1_file_eval)
  qa_model.train()

qa_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, best_checkpoint))['model_dict'])
qa_model.eval()
print('-------------------- Test --------------------')
test_p = EvalProcessOutput()
with torch.no_grad():
  for step, data in enumerate(test_dataloader):
    start_positions = data.pop('start_positions')
    end_positions = data.pop('end_positions')
    features = data.pop('features')
    start_logits, end_logits = qa_model(data)
    test_p.process_feature_output(to_numpy(start_logits),
                                  to_numpy(end_logits),
                                  features)

test_p.process_output()
filename = os.path.join(log_dir, f'test_preds.json')
write_to_file(test_p.dialogs_answers, filename)
res__ = run_eval(filename, orig_test_file)
save_results(scores_test, res__)
print_mean(res__['turn_f1s'], mean_f1_file_test)
