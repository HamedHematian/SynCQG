from utils import *
import os
import torch

def build_save_settings(args): 

  if args.model_name == 'BERT':
    save_dir_prefix = f'{args.source_directory}/{args.model_name}_BS_{args.batch_size}'
  else:
    save_dir_prefix = f'{args.source_directory}/{args.model_name}_S_{args.S}_KL_{args.kl_ratio}_Tau_{args.tau}_use_sim_threshold_{args.use_sim_threshold}_dist_{args.dist_type}_SEED_{args.seed}_BS_{args.batch_size}/'
  raw_checkpoint_dir = 'Checkpoint/'
  raw_log_dir = 'Log/'

  checkpoint_dir = os.path.join(save_dir_prefix, raw_checkpoint_dir) # checkpoint dir
  log_dir = os.path.join(save_dir_prefix, raw_log_dir) # a file to store loss records
  meta_log_file = os.path.join(log_dir, 'settings.txt') # a file to save the settings

  loss_log_file = os.path.join(log_dir, 'loss.txt') # file to log the training loss
  mean_f1_file_eval = os.path.join(log_dir, 'mean_f1_eval.txt')
  mean_f1_file_test = os.path.join(log_dir, 'mean_f1_test.txt')
  scores_eval = os.path.join(log_dir, 'scores_eval.txt') # a file to store dev predictions
  scores_test = os.path.join(log_dir, 'scores_test.txt') # a file to store test predictions

  make_dir(save_dir_prefix)
  make_dir(checkpoint_dir)
  make_dir(log_dir)
  with open(meta_log_file, 'w') as f:
    pass


  # check the checkpoints drive
  checkpoint_files = os.listdir(checkpoint_dir)
  if len(checkpoint_files) == 0:
    checkpoint_available = False
    print('No checkpoint found, training from begining')
  else:
    checkpoint_available = True
    assert len(checkpoint_files) >= 1, 'Checkpoints are messed up'

  if checkpoint_available:
    current_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[1]))[-1]
    print('Loading Checkpoint:', current_checkpoint)
    current_checkpoint = os.path.join(checkpoint_dir, current_checkpoint)
  else:
    current_checkpoint = None


  return save_dir_prefix, checkpoint_dir, log_dir, meta_log_file, \
          loss_log_file, mean_f1_file_eval, mean_f1_file_test, \
          scores_eval, scores_test, checkpoint_available, current_checkpoint



def print_loss(loss_collection, epoch, epochs, step, steps):
  txt = f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step}/{steps}] | Loss {round(sum(loss_collection) / len(loss_collection), 4)}'
  print(txt)

def save_loss(loss_collection, epoch, epochs, step, steps, loss_log_file):
  txt = f'EPOCH [{epoch + 1}/{epochs}] | STEP [{step}/{steps}] | Loss {round(sum(loss_collection) / len(loss_collection), 4)}'
  with open(loss_log_file, 'a') as f:
    f.write(txt)
    f.write('\n')


def save_checkpoint(epoch, checkpoint_dir, qa_model, optimizer, scheduler):
  filename_prefix = os.path.join(checkpoint_dir, f'checkpoint_{epoch}')
  checkpoint_config = {
  'epoch': epoch,
  'optimizer_dict': optimizer.state_dict(),
  'scheduler_dict': scheduler.state_dict(),
  'model_dict': qa_model.state_dict()}
  torch.save(checkpoint_config, filename_prefix)
  

def load_checkpoint(current_checkpoint):
    # models have been loaded before so no need to load them again
    checkpoint_config = torch.load(current_checkpoint)
    return (checkpoint_config['epoch'],
            checkpoint_config['optimizer_dict'],
            checkpoint_config['scheduler_dict'],
            checkpoint_config['model_dict'])
