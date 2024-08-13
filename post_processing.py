from collections import namedtuple, defaultdict
import numpy as np
from eval import find_reverse_mapping
import unicodedata

feature_output = namedtuple(
    'feature_output',
        ['start_logit', 'end_logit', 'feature'])

PrelimPrediction = namedtuple(
    "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "qid"]
    )

NbestPrediction = namedtuple(
    "NbestPrediction", ["text", "start_logit", "end_logit"]
)

Answer = namedtuple(
    'Answer', ['qid', 'answer']
)


# A class to turn back the predicted token answers from eval (dev) and test sets to string answers
class EvalProcessOutput:
  def __init__(self, n_best_size=4, answer_max_len=40, answerability_threshold=0.0):
    self.answers = defaultdict(list)
    self.examples_output = []
    self.n_best_size = n_best_size
    self.answer_max_len = answer_max_len
    self.answerability_threshold = answerability_threshold
    self.ps = []


  def process_feature_output(self, start_logits, end_logits, features):
    for start_logit, end_logit, feature in zip(start_logits, end_logits, features):
      self.examples_output.append(
          feature_output(start_logit, end_logit, feature)
      )

  def stack_features(self):
    examples_list = defaultdict(list)
    for feature_out in self.examples_output:
      examples_list[feature_out.feature.qid].append(feature_out)
    return examples_list


  def process_output(self):
    self.extract_answers()
    self.get_predictions()

  def get_predictions(self):
    dialogs = defaultdict(list)
    self.dialogs_answers = defaultdict(list)
    for example_qid, answer in self.answers.items():
      dialog_id = example_qid.split('#')[0]
      dialogs[dialog_id].append(Answer(example_qid, answer))

    self.digs = dialogs
    for dialog_id, dialog in dialogs.items():
      dialog = sorted(dialog, key=lambda x: int(x.qid.split('#')[1]))
      max_dialog_len = int(dialog[-1].qid.split('#')[1]) + 1
      self.dialogs_answers[dialog_id] = ['' for i in range(max_dialog_len)]
      for example in dialog:
        example_turn = int(example.qid.split('#')[1])
        self.dialogs_answers[dialog_id][example_turn] = example.answer


  def extract_answers(self):
    examples_list = self.stack_features()
    for example_qid, example in examples_list.items():
      null_score = np.inf
      prelim_predictions = []
      self.example = example
     
      for feature_index, feature_output in enumerate(example):
        feature_null_score = feature_output.start_logit[0] + feature_output.end_logit[0]

        if feature_null_score < null_score:
          null_score = feature_null_score
          null_feature_index = feature_index
          null_start_logit = feature_output.start_logit[0]
          null_end_logit = feature_output.end_logit[0]

        start_indexes = self.get_best_indexes(feature_output.start_logit)
        end_indexes = self.get_best_indexes(feature_output.end_logit)

        for start_index in start_indexes:
          for end_index in end_indexes:
            if start_index > feature_output.feature.context_end:
              continue
            if end_index > feature_output.feature.context_end:
              continue
            if start_index < feature_output.feature.context_start:
              continue
            if end_index < feature_output.feature.context_start:
              continue
            if start_index < feature_output.feature.mask_span[0]:
              continue
            if end_index - start_index + 1 > self.answer_max_len:
              continue
            if end_index <= start_index:
              continue

            prelim_predictions.append(
                PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=feature_output.start_logit[start_index],
                  end_logit=feature_output.end_logit[end_index],
                  qid=example_qid
            )
                )
      # append a null one for handling CANNOTANSWER
      prelim_predictions.append(
        PrelimPrediction(
          feature_index=null_feature_index,
          start_index=0,
          end_index=0,
          start_logit=null_start_logit,
          end_logit=null_end_logit,
          qid=example_qid
      )
        )
      prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
      self.t = prelim_predictions
      best_pred = prelim_predictions[0]
      is_answerable = null_score - (best_pred.start_logit + best_pred.end_logit) <= self.answerability_threshold
      if is_answerable:
        feature = example[best_pred.feature_index].feature
        start_char = feature.offset_mappings[best_pred.start_index][0]
        end_char = feature.offset_mappings[best_pred.end_index][1]
        cleaned_answer = feature.cleaned_context[start_char: end_char + 1]
        answer = find_reverse_mapping(cleaned_answer, feature.context)
      else:
        answer = 'CANNOTANSWER'

      self.answers[example_qid] = answer


  def get_best_indexes(self, logits):
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= self.n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

