import random
from utils import *
import torch


class DataManager:

  def __init__(self, current_file, current_index, data_dir, batch_size, shuffle=True):
    self.files = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[2].split('.')[0]))
    self.files = list(map(lambda x: os.path.join(data_dir, x), self.files))
    self.shuffle = shuffle
    self.data_len = 0
    for filename in self.files:
      self.data_len += len(load_data(filename))
    self.batch_size = batch_size
    self.reset_datamanager(current_file, current_index)

  def reset_datamanager(self, current_file_index, current_index):
    self.current_index = current_index
    self.current_file_index = current_file_index
    self.features = self.load_data_file(self.files[self.current_file_index])

  def load_data_file(self, filename):
    if self.shuffle:
      data = load_data(filename)
      random.shuffle(data)
      return data
    else:
      return load_data(filename)

  def next(self):
    temp = self.features[self.current_index:self.current_index + self.batch_size]
    self.temp = temp
    self.current_index += self.batch_size
    if self.current_index >= len(self.features):
      self.current_index = 0
      self.current_file_index += 1
      if self.current_file_index == len(self.files):
        self.reset_datamanager(current_file_index=0, current_index=0)
        return temp, True
      else:
        self.features = self.load_data_file(self.files[self.current_file_index])
    return temp, False
    
    
    
    
    
    
class Simple_DataLoader:

  def __init__(self, current_file, current_index, batch_size, shuffle=True, data_type='train'):
    self.data_type = data_type
    self.batch_size = batch_size
    self.data_manager = DataManager(current_file, current_index, f'features/{data_type}/', batch_size, shuffle)

  def __iter__(self):
    self.stop_iteration = False
    return self

  def __len__(self):
    return int(self.data_manager.data_len // self.batch_size)

  def reset_dataloader(self, current_file, current_index):
    self.data_manager.reset_datamanager(current_file, current_index)

  def features_2_tensor(self, features):
    x = dict()
    x['input_ids'] = torch.LongTensor([feature.input_ids for feature in features])
    x['attention_mask'] = torch.LongTensor([feature.attention_mask for feature in features])
    x['token_type_ids'] = torch.LongTensor([feature.token_type_ids for feature in features])
    x['start_positions'] = torch.cat([torch.tensor([feature.start]) for feature in features]).view(-1)
    x['end_positions'] = torch.cat([torch.tensor([feature.end]) for feature in features]).view(-1)
    x['features'] = features
    return x

  def __next__(self):
    if self.stop_iteration:
      raise StopIteration
    features, self.stop_iteration = self.data_manager.next()
    return self.features_2_tensor(features)





class Cotah_DataLoader:

  def __init__(self, current_file, current_index, batch_size, shuffle=True, data_type='train'):
    self.data_type = data_type
    self.batch_size = batch_size
    self.data_manager = DataManager(current_file, current_index, f'features/{data_type}/', batch_size, shuffle)

  def __iter__(self):
    self.stop_iteration = False
    return self

  def __len__(self):
    return int(self.data_manager.data_len // self.batch_size)

  def reset_dataloader(self, current_file, current_index):
    self.data_manager.reset_datamanager(current_file, current_index)

  def features_2_tensor(self, features_list):
    x, y = dict(), dict()
    # this will flat all
    self.T = features_list
    if self.data_type == 'train':
      x['input_ids'] = torch.LongTensor([feat.input_ids for features in features_list for feat in features])
      x['attention_mask'] = torch.LongTensor([feat.attention_mask for features in features_list for feat in features])
      x['token_type_ids'] = torch.LongTensor([feat.token_type_ids for features in features_list for feat in features])
      x['start_positions'] = torch.cat([torch.tensor([feat.start]) for features in features_list for feat in features]).view(-1)
      x['end_positions'] = torch.cat([torch.tensor([feat.end]) for features in features_list for feat in features]).view(-1)
      x['features'] = [feat  for features in features_list for feat in features]
      x['cluster_size'] = len(features_list[0])
    else:
      x['input_ids'] = torch.LongTensor([feat.input_ids for features in features_list for feature in features for feat in feature])
      x['attention_mask'] = torch.LongTensor([feat.attention_mask for features in features_list for feature in features for feat in feature])
      x['token_type_ids'] = torch.LongTensor([feat.token_type_ids for features in features_list for feature in features for feat in feature])
      x['start_positions'] = torch.cat([torch.tensor([feat.start]) for features in features_list for feature in features for feat in feature]).view(-1)
      x['end_positions'] = torch.cat([torch.tensor([feat.end]) for features in features_list for feature in features for feat in feature]).view(-1)
      x['features'] = [feat for features in features_list for feature in features for feat in feature]
      x['cluster_size'] = len(features_list[0])

    if self.data_type in ['eval', 'test']:
      return x

    y['input_ids_0'] = get_even(x['input_ids'])
    y['input_ids_1'] = get_odd(x['input_ids'])
    y['attention_mask_0'] = get_even(x['attention_mask'])
    y['attention_mask_1'] = get_odd(x['attention_mask'])
    y['token_type_ids_0'] = get_even(x['token_type_ids'])
    y['token_type_ids_1'] = get_odd(x['token_type_ids'])
    y['start_positions_0'] = get_even(x['start_positions'])
    y['start_positions_1'] = get_odd(x['start_positions'])
    y['end_positions_0'] = get_even(x['end_positions'])
    y['end_positions_1'] = get_odd(x['end_positions'])
    y['features_0'] = get_even(x['features'])
    y['features_1'] = get_odd(x['features'])
    y['cluster_size'] = x['cluster_size']

    if self.data_type == 'train':
      return y


  def __next__(self):
    if self.stop_iteration:
      raise StopIteration
    features, self.stop_iteration = self.data_manager.next()
    return self.features_2_tensor(features)
