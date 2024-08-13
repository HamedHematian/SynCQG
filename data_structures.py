



# A class to hold QuAC examples
class CQA_DATA:

  def __init__(self,
               question,
               context,
               history,
               answers,
               human_answer,
               qid,
               q_num,
               answer_start,
               answer_end,
               is_answerable):

    self.question = question
    self.context = context
    self.answers = answers
    self.human_answer = human_answer
    self.history = history
    self.qid = qid
    self.q_num = q_num
    self.answer_start = answer_start
    self.answer_end = answer_end
    self.is_answerable = is_answerable
    self.cleaned_context = None
    self.answer = self.answers[0]['text']

  def __repr__(self):
    repr = ''
    repr += 'context -> ' + self.context[:100] + '\n'
    repr += 'question ->' + self.question + '\n'
    repr += 'question id ->' + str(self.qid) + '\n'
    repr += 'turn_number ->' + str(self.q_num) + '\n'
    repr += 'answer ->' + self.answers[0]['text'] + '\n'
    return repr


# A class to hold features (An example could be turned into multiple features)
class CoTaH_Feature:

  def __init__(self,
               qid,
               question_part,
               input_ids,
               attention_mask,
               token_type_ids,
               offset_mappings,
               hist_chunk_dict,
               max_context_dict,
               start,
               end,
               is_answerable,
               context,
               cleaned_context,
               context_start,
               context_end,
               example_start_char,
               example_end_char,
               example_answer):

    self.qid = qid
    self.question_part = question_part
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.offset_mappings = offset_mappings
    self.hist_chunk_dict = hist_chunk_dict
    self.max_context_dict = max_context_dict
    self.start = start
    self.end = end
    self.is_answerable = is_answerable
    self.context = context
    self.cleaned_context = cleaned_context
    self.context_start = context_start
    self.context_end = context_end
    self.example_start_char = example_start_char
    self.example_end_char = example_end_char
    self.example_answer = example_answer

  def __repr__(self):
    repr = ''
    repr += 'qid --> ' + str(self.qid) + '\n'
    repr += 'quesion part --> ' + str(self.question_part) + '\n'
    repr += 'answer part --> ' + str(self.start) + ' ' + str(self.end) + '\n'
    return repr





class Simple_Feature:

  def __init__(self,
               qid,
               question_part,
               input_ids,
               attention_mask,
               token_type_ids,
               offset_mappings,
               max_context_dict,
               start,
               end,
               is_answerable,
               context,
               cleaned_context,
               context_start,
               context_end,
               example_start_char,
               example_end_char,
               example_answer):

    self.qid = qid
    self.question_part = question_part
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.offset_mappings = offset_mappings
    self.max_context_dict = max_context_dict
    self.start = start
    self.end = end
    self.is_answerable = is_answerable
    self.context = context
    self.cleaned_context = cleaned_context
    self.context_start = context_start
    self.context_end = context_end
    self.example_start_char = example_start_char
    self.example_end_char = example_end_char
    self.example_answer = example_answer

  def __repr__(self):
    repr = ''
    repr += 'qid --> ' + str(self.qid) + '\n'
    repr += 'quesion part --> ' + str(self.question_part) + '\n'
    repr += 'answer part --> ' + str(self.start) + ' ' + str(self.end) + '\n'
    return repr
