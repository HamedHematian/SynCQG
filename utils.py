import json
import pickle
import torch.nn.functional as F
import unicodedata
import os

# helping functions

def read_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)
    

def write_json(data, filename):
  with open(filename, 'w') as f:
    return json.dump(data, f)
    

def load_data(filename):
  with open(filename, 'rb') as f:
    x = pickle.load(f)
  return x
  

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        

def save_results(filename, results):
  with open(filename, 'a') as f:
    for k, v in results.items():
      string = str(k) + ': ' + str(v) + '\n'
      f.write(string)
    f.write('\n\n')
    
    
def get_even(X):
  return X[::2]


def get_odd(X):
  return X[1::2]
  
  
def compute_kl(logits_1, logits_2, loss_fn):
  A = F.log_softmax(logits_1, dim=-1)
  B = F.softmax(logits_2, dim=-1)
  kl_loss = loss_fn(A, B)
  kl_loss = kl_loss.sum()
  return kl_loss
  

def to_numpy(tensor):
  return tensor.detach().cpu().numpy()
  
  
# Creating the necessary directories
def make_dir(dir_name):
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    
    
# clean examples
class CleanSample():

  def __init__(self, sample):
    self.sample = sample

  def clean(self):
    cleaned_question = self._clean_text(self.sample.question)
    cleaned_context = self._clean_text(self.sample.context, keep_track=True)
    self.sample.question = cleaned_question
    self.sample.cleaned_context = cleaned_context
    cleaned_answer = self.strip_text(self.sample.answers[0]['text'])
    self.sample.cleaned_answer = {
        'text': cleaned_answer[0],
        'start': self.sample.answers[0]['start'] + cleaned_answer[1],
        'end': self.sample.answers[0]['end'] - cleaned_answer[2]
    }
    return self.sample

  def _is_control(self, char):
    if char == "\t" or char == "\n" or char == "\r":
      return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
      return True
    return False

  def strip_text(self, text):
    start_spaces = 0
    end_spaces = 0
    while(text[0] == ' '):
      text = text[1:]
      start_spaces += 1
    while(text[-1] == ' '):
      text = text[:-1]
      end_spaces += 1
    return text, start_spaces, end_spaces


  def _is_whitespace(self, char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
      return True
    cat = unicodedata.category(char)
    if cat == "Zs":
      return True
    return False

  def _clean_text(self, text, keep_track=False):
    output = []
    new_answer_start = self.sample.answer_start
    new_answer_end = self.sample.answer_end

    for index, char in enumerate(text):
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or self._is_control(char):
        if keep_track:
          if index < new_answer_start:
            new_answer_start -= 1
          if index < new_answer_end:
            new_answer_end -= 1
        continue
      if self._is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    self.sample.answer_start = new_answer_start
    self.sample.answer_end = new_answer_end
    return "".join(output)
    
    
    
    
    
def handle_max_context(example_features, context_start):
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
