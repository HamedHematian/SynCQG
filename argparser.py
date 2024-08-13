import argparse

def cotah_parse_args():

  parser = argparse.ArgumentParser(description="argument parser for the CoTaH code")
  parser.add_argument("-S", "--S", type=int, default=2, help="number of synthetic questions to use for augmenting the history")
  parser.add_argument("-K", "--kl_ratio", type=float, default=2.0, help="consistency loss ratio (lambda)")
  parser.add_argument("-y", "--gamma", type=float, default=.8, help="gamma")
  parser.add_argument("-M", "--M", type=int, default=10, help="M: top synthetic questions relevant to the history")
  parser.add_argument("-T", "--tau", type=int, default=6, help="threshold (tau)")
  parser.add_argument("-B", "--batch_size", type=int, help="batch size", default=6)
  parser.add_argument("-A", "--accumulation_steps", type=int, help="gradient accumulation steps", default=1)
  parser.add_argument("-s", "--seed", type=int, help="seed", default=1000)
  parser.add_argument("-L", "--dist_type", type=str, help="whether to use linear distribution for question selection", default='uniform')
  parser.add_argument("-U", "--use_sim_threshold", type=bool, help="whether to filter questions based on similarity", default=True)
  parser.add_argument("-d", "--source_directory", type=str, help="source_directory (default is current directory)", default='.')
  parser.add_argument("-m", "--model_name", type=str, help="model name (choose it yourself)", default='CoTaH-BERT')
  
  args = parser.parse_args()
  return args
  
  
def simple_parse_args():
  
  parser = argparse.ArgumentParser(description="argument parser for the BERT code")
  parser.add_argument("-B", "--batch_size", type=int, help="batch size", default=6)
  parser.add_argument("-A", "--accumulation_steps", type=int, help="gradient accumulation steps", default=1)
  parser.add_argument("-s", "--seed", type=int, help="seed", default=1000)
  parser.add_argument("-m", "--model_name", type=str, help="model name (choose it yourself)", default='BERT')
  parser.add_argument("-d", "--source_directory", type=str, help="source_directory (default is current directory)", default='.')
  
  args = parser.parse_args()
  return args

