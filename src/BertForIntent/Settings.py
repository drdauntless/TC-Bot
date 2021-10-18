import json

class Settings:

    def __init__(self, filename):
        self.json_file = None
        with open(filename) as f:
            self.json_file = json.load(f)
        self.hh_sheet_id = self.json_file['hh_sheet_id']
        self.ha_sheet_id = self.json_file['ha_sheet_id']

        # List of str: names of the individual sheets in the Google Sheet we want to use
        self.hh_sheet_names = self.json_file['hh_sheet_names']
        self.ha_sheet_names = self.json_file['ha_sheet_names']

        self.columns_list = self.json_file['columns_list']

        self.dialogue_column = self.json_file['dialogue_column']
        self.dialogue_act_column = self.json_file['dialogue_act_column']
        self.intent_column = self.json_file['intent_column']
        self.delivery_column = self.json_file['delivery_column']
        self.who_column = self.json_file['who_column']
        self.action_column = self.json_file['action_column']
        self.tone_column = self.json_file['tone_column']

        self.driver_dict = self.json_file['driver_dict']
        self.creativity_dict = self.json_file['creativity_dict']
        self.dialogue_act_dict = self.json_file['dialogue_act_dict']
        self.intent_dict = self.json_file['intent_dict']
        self.delivery_dict = self.json_file['delivery_dict']
        self.action_dict = self.json_file['action_dict']
        self.who_dict = self.json_file['who_dict']
        self.tone_dict = self.json_file['tone_dict']

        self.root_column_name = self.json_file['root_column_name']
        self.root_encode_dict = self.json_file[self.json_file['root_encode_dict']]
        self.root_hierarchy = self.json_file['root_hierarchy']

        self.stagger_training = self.json_file['stagger_training']
        self.num_runs = self.json_file['num_runs']
        self.num_train_epochs = self.json_file['num_train_epochs']
        self.per_device_train_batch_size = self.json_file['per_device_train_batch_size']
        self.per_device_eval_batch_size = self.json_file['per_device_eval_batch_size']
        self.warmup_steps = self.json_file['warmup_steps']
        self.weight_decay = self.json_file['weight_decay']
        self.evaluation_strategy = self.json_file['evaluation_strategy']
        self.eval_accumulation_steps = self.json_file['eval_accumulation_steps']