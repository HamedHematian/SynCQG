import yaml

# Load the YAML file
with open('settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
    

train_path = data.get('train_path')
eval_path = data.get('eval_path')
files_here = data.get('files_here')
model_id = data.get('model_id')
start_offset_ = data.get('start_offset_')
quac_train_url = data.get('quac_train_url')
quac_eval_url = data.get('quac_eval_url')
synthetic_questions_id = data.get('synthetic_questions_id')
orig_eval_file = data.get('orig_eval_file')
orig_test_file = data.get('orig_test_file')
is_preprocessed = data.get('is_preprocessed')
model_present = data.get('model_present')
only_np = data.get('only_np')

