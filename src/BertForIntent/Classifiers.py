from sklearn import metrics
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, BertConfig, GPT2Config, \
    BertForSequenceClassification, BertTokenizer, GPT2ForSequenceClassification, GPT2Tokenizer, \
    AlbertTokenizer, AlbertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, \
    XLNetTokenizer, XLNetForSequenceClassification

import re
from datasets import Dataset, load_dataset, concatenate_datasets
import gc

global_encode_dict = None


def compute_metrics(pred):
    rtn_dict = dict()
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    decode_dict = {value: key for (key, value) in global_encode_dict.items()}
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    rtn_dict["avg.accuracy"] = acc
    rtn_dict["avg.precision"] = precision
    rtn_dict["avg.recall"] = recall
    rtn_dict["avg.f1"] = f1

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
    matrix = confusion_matrix(labels, preds)
    acc = matrix.diagonal() / matrix.sum(axis=1)

    for name, a, p, r, f, s in zip(list(decode_dict.items()), acc, precision, recall, f1, support):
        name = name[1]
        rtn_dict[name + ".accuracy"] = a
        rtn_dict[name + ".precision"] = p
        rtn_dict[name + ".recall"] = r
        rtn_dict[name + ".f1"] = f
        rtn_dict[name + ".support"] = s

    return rtn_dict


class HierarchicalClassifier:

    def __init__(self, train_dataset, test_dataset, ha_dataset=None, hh_dataset=None):
        self.root_layer = None
        self.max_length = 0
        self.pretrained_weights = None
        self.tokenizer = None

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_length)

    def getMaxLength(self, dataset):

        # We must encode everything before we get the length of the encoded version (different lengths)
        encoded = dataset.map(self.tokenize, batched=True, batch_size=None)

        max_len = max([len(sent) for sent in encoded['input_ids']])
        return min(512, max_len)

    def train_and_eval(self, train_dataset, test_dataset, ha_dataset, hh_dataset):
        for dataset in train_dataset:
            dataset.rename_column_(settings.dialogue_column, 'text')

        for dataset in test_dataset:
            dataset.rename_column_(settings.dialogue_column, 'text')

        # Get the longest utterance only in training
        self.max_length = self.getMaxLength(train_dataset[0])

        if hh_dataset != None and ha_dataset != None:
            ha_dataset.rename_column_(settings.dialogue_column, 'text')
            hh_dataset.rename_column_(settings.dialogue_column, 'text')
            length = self.getMaxLength(hh_dataset)

            if (self.max_length < length):
                self.max_length = length

        self.root_layer.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)


class ClassifierLayer:
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                 hh_dataset, label_subset=None):
        self.hier_classifier = hier_classifier

        self.column_name = settings.json_file[column_name]
        self.encode_dict = encode_dict
        self.label_subset = label_subset

        self.model = None
        self.child_layers = []

    def init():
        pass

    def train(self, hh_dataframe, ha_dataframe):
        train_dataset, test_dataset = self.preprocess(hh_dataframe, ha_dataframe)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        self.trainer.train()

    def diff(self, li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif

    def tokenize(self, batch):
        return self.hier_classifier.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_length)

    def encode(self, row):
        # print("self.column_name: " + str(self.column_name))
        # print("self.encode_dict: " + str(self.encode_dict))
        # print("row: " + str(row))
        # print("self.encode_dict[row[self.column_name]]: " + str(self.encode_dict[row[self.column_name]]))

        return {self.column_name: self.encode_dict[row[self.column_name]]}

    def remove(self, row):
        if (row[self.column_name] == "") or (not row[self.column_name] in self.encode_dict.keys()):
            return False
        return True

    def rem_row(self, row):
        # print("settings.root_encode_dict: " + str(settings.root_encode_dict))
        # print("self.label_subset: " + str(list(map(settings.root_encode_dict.get, self.label_subset))))
        # print("row[settings.json_file[settings.root_column_name]]: " + str(row[settings.json_file[settings.root_column_name]]))
        # print("row in subset: " + str(row[settings.json_file[settings.root_column_name]] in self.label_subset))
        return row[settings.json_file[settings.root_column_name]] in self.label_subset

    def preprocess(self, dataset):
        # print("The following should be a dataset: " + str(type(dataset)))

        # Copy the dataset
        new_dataset = concatenate_datasets([dataset])

        # print("Before Remove Row: " + str(new_dataset))
        # Remove rows that aren't part of the sub-class (if this is a sub-class)
        if self.label_subset != None:
            new_dataset = new_dataset.filter(self.rem_row)
        # print("After Remove Row: " + str(new_dataset))

        # Remove columns that aren't needed
        # Subtract the desired rows (label + text) from all rows to select which rows to delete
        # print("Before Remove Column: " + str(new_dataset))
        to_remove = self.diff(settings.columns_list, [self.column_name, settings.dialogue_column])
        new_dataset = new_dataset.map(lambda row: row, remove_columns=to_remove)
        # print("After Remove Column: " + str(new_dataset))
        # print("Before Remove NA: " + str(new_dataset))
        new_dataset = new_dataset.filter(self.remove)
        # print("After Remove NA: " + str(new_dataset))

        # Encode the label
        # print("Before Encoded: " + str(new_dataset))
        # print(new_dataset[0])
        new_dataset = new_dataset.map(self.encode)
        # print("Encoded: " + str(new_dataset[0]))
        # dataset[settings.intent_column] = dataset[settings.intent_column].map(settings.intent_dict)

        # Tokenize
        # print("Before Tokenize: " + str(new_dataset))
        new_dataset = new_dataset.map(
            lambda batch: self.hier_classifier.tokenizer(batch['text'], padding=True, truncation=True,
                                                         max_length=self.hier_classifier.max_length), batched=True,
            batch_size=None)
        # print("After Tokenize: " + str(new_dataset))

        new_dataset.rename_column_(self.column_name, 'label')
        # print("After Rename: " + str(new_dataset))
        # print("First Item Example: " + str(new_dataset[0]))
        return new_dataset

    def preprocess_all(self, train_dataset, test_dataset, ha_dataset, hh_dataset):

        new_train_dataset = self.preprocess(train_dataset)
        new_test_dataset = self.preprocess(test_dataset)
        new_ha_dataset = None
        new_hh_dataset = None

        if ha_dataset != None:
            new_ha_dataset = self.preprocess(ha_dataset)

        if hh_dataset != None:
            new_hh_dataset = self.preprocess(hh_dataset)

        return new_train_dataset, new_test_dataset, new_ha_dataset, new_hh_dataset

    def train_and_eval(self, train_dataset, test_dataset, ha_dataset, hh_dataset):
        self.init()

        # Set the global encode dict so that computer_metrics will use the proper one
        global global_encode_dict
        global_encode_dict = self.encode_dict

        if True:
            new_train_dataset = []
            new_test_dataset = []
            new_ha_dataset = None
            new_hh_dataset = None
            # Preprocess both as new datasets to preserve the originals for the children layers
            for train_ds, test_ds in zip(train_dataset, test_dataset):
                # Preprocess both as new datasets to preserve the originals for the children layers
                temp_new_train_dataset, temp_new_test_dataset, new_ha_dataset, new_hh_dataset = self.preprocess_all(
                    train_ds, test_ds, ha_dataset, hh_dataset)

                new_train_dataset.append(temp_new_train_dataset)
                new_test_dataset.append(temp_new_test_dataset)

            print("Classifying " + self.column_name)

            original_name = self.training_args.run_name
            i = 0
            wandb.watch(self.model)
            for new_train_ds, new_test_ds in zip(new_train_dataset, new_test_dataset):
                i += 1
                self.training_args.run_name = original_name
                self.init()

                if hh_dataset != None:
                    print("HH Train Dataset: " + str(new_hh_dataset.shape))
                    print("Eval Dataset: " + str(new_ha_dataset.shape))

                    self.training_args.run_name += " PP"

                    self.trainer = Trainer(
                        model=self.model,
                        args=self.training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=new_hh_dataset,
                        eval_dataset=new_ha_dataset
                    )

                    self.trainer.is_model_parallel = False

                    run = wandb.init(reinit=True, project="WoZ Classifier", group=self.training_args.run_name,
                                     name=self.training_args.run_name + str(i))

                    self.trainer.train()
                    run.finish()

                print("Train Dataset: " + str(new_train_ds.shape))
                print("Eval Dataset: " + str(new_test_ds.shape))

                self.training_args.run_name += " PA"

                self.trainer = Trainer(
                    model=self.model,
                    args=self.training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=new_train_ds,
                    eval_dataset=new_test_ds
                )
                self.trainer.is_model_parallel = False
                # print("training")
                run2 = wandb.init(reinit=True, project="WoZ Classifier", group=self.training_args.run_name,
                                  name=self.training_args.run_name + str(i))

                self.trainer.train()
                run2.finish()
                # print("erasing model")

        self.model = None
        gc.collect()

        # for layer in self.child_layers:
        #    layer.train_and_eval(train_dataset, test_dataset)

        if len(self.child_layers) > 0:
            zero = self.child_layers[0]
            one = self.child_layers[1]
            two = self.child_layers[2]
            three = self.child_layers[3]

            zero.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)
            self.child_layers[0] = None
            zero = None
            gc.collect()

            one.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)
            self.child_layers[1] = None
            one = None
            gc.collect()

            two.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)
            self.child_layers[2] = None
            two = None
            gc.collect()

            three.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)
            self.child_layers[3] = None
            three = None
            gc.collect()


class Classifier:

    def __init__(self, hh_dataframe, ha_dataframe):
        self.pretrained_weights = None
        self.tokenizer = None
        self.config = None
        self.max_length = 0

        self.model = None

        self.trainer = None
        self.training_args = None

    def getMaxLength(self, dataframe):
        # We must encode everything before we get the length of the encoded version (different lengths)
        print(dataframe[settings.dialogue_column].tolist())
        print(self.tokenizer)
        encoded = self.tokenizer(dataframe[settings.dialogue_column].tolist())

        max_len = max([len(sent) for sent in encoded['input_ids']])
        return max_len

    def getMaxLengths(self, hh_dataframe, ha_dataframe):
        return min(512, max(self.getMaxLength(hh_dataframe), self.getMaxLength(ha_dataframe)))

    def preprocess(self, hh_dataframe, ha_dataframe):
        self.max_length = self.getMaxLengths(hh_dataframe, ha_dataframe)
        train_dataframe, test_dataframe = train_test_split(ha_dataframe)
        print("train_dataframe: " + str(train_dataframe.shape))
        print("test_dataframe: " + str(test_dataframe.shape))

        train_dataframe.append(hh_dataframe)
        print("train_dataframe + hh: " + str(train_dataframe.shape))

        train_dataset = Dataset.from_pandas(train_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)

        print(type(train_dataset))
        print(type(test_dataset))
        # train, test = load_dataset('imdb', split=['train', 'test'])
        # print(train)

        train_dataset.rename_column_(settings.intent_column, 'label')
        test_dataset.rename_column_(settings.intent_column, 'label')
        print(test_dataset[0]['label'])

        train_dataset.rename_column_(settings.dialogue_column, 'text')
        test_dataset.rename_column_(settings.dialogue_column, 'text')
        print("train_dataset: " + str(train_dataset))
        print("train_dataset: " + str(test_dataset))

        # train = train.map(self.tokenize, batched=True)
        # test = test.map(self.tokenize, batched=True)
        # temp = self.tokenizer(dataset['text'], padding=True, truncation=True)
        print("Before tokenization: " + str(train_dataset))
        train_dataset = train_dataset.map(self.tokenize, batched=True, batch_size=None)
        test_dataset = test_dataset.map(self.tokenize, batched=True, batch_size=None)
        print("After tokenization: " + str(train_dataset))
        # dataset_dict = self.train_test_split(ha_dataset)
        # train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # return dataset_dict['train'], dataset_dict['test']
        return train_dataset, test_dataset
        # text_train.set_format('torch', columns=)

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_length)

    def train_test_split(self, dataset):
        return Dataset.train_test_split(
            dataset,
            test_size=0.4,
            shuffle=True,
            seed=2018)

    def train(self, hh_dataframe, ha_dataframe):
        train_dataset, test_dataset = self.preprocess(hh_dataframe, ha_dataframe)
        print(train_dataset[0])
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        print(self.pretrained_weights)
        self.trainer.train()

        print(self.trainer.evaluate())
        # trainer.predict(test_dataset[0])
        # accuracy = metrics.accuracy_score(test_dataset['label'], trainer.predict(test_dataset)[1]) * 100
        # print("Accuracy: " + str(accuracy))

    def evaluate(self, dataframe):
        # train_dataset, test_dataset = self.preprocess(dataframe)

        # chef = pipeline('text-generation', model='./results', tokenizer='gpt2',
        #                        config={'max_length': self.max_length})
        # result = chef("testing")[0]['generated_text']
        # print(result)
        # accuracy = sklearn.metrics.accuracy_score(test_dataset['label'], trainer.predict(test_dataset)[1]) * 100
        print(self.trainer.evaluate())


class Gpt2Classifier(HierarchicalClassifier):
    def __init__(self, train_dataset, test_dataset, ha_dataset=None, hh_dataset=None):
        super().__init__(train_dataset, test_dataset, ha_dataset, hh_dataset)
        self.pretrained_weights = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_weights)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.root_layer = Gpt2Layer(train_dataset, test_dataset, self, settings.root_column_name,
                                    settings.root_encode_dict, settings.root_hierarchy, ha_dataset, hh_dataset)
        self.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)


class Gpt2Layer(ClassifierLayer):
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                 hh_dataset, label_subset=None):
        super().__init__(train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                         hh_dataset, label_subset)

        self.training_args = TrainingArguments(
            output_dir='gpt2' + self.column_name,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="gpt-2: " + self.column_name  # name of the W&B run (optional)
        )

        if (hierarchy != None):
            # print("type(hierarchy): " + str(type(hierarchy)))
            for layer_column_name in hierarchy.keys():
                layer_encode_dict = settings.json_file[hierarchy[layer_column_name]["encode_dict"]]
                layer_label_subset = hierarchy[layer_column_name]["label_subset"]

                self.child_layers.append(
                    Gpt2Layer(train_dataset, test_dataset, hier_classifier, layer_column_name, layer_encode_dict, None,
                              ha_dataset, hh_dataset, layer_label_subset))

    def init(self):
        self.model = GPT2ForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                   num_labels=len(self.encode_dict),
                                                                   pad_token_id=self.hier_classifier.tokenizer.eos_token_id)
        self.model.pad_token = self.hier_classifier.tokenizer.eos_token
        return self.model


class Gpt2(Classifier):

    def __init__(self, hh_dataframe, ha_dataframe):
        super().__init__(hh_dataframe, ha_dataframe)
        self.pretrained_weights = 'gpt2'
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.pretrained_weights)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2ForSequenceClassification.from_pretrained(self.pretrained_weights, num_labels=16,
                                                                   pad_token_id=self.tokenizer.eos_token_id)

        self.model.pad_token = self.tokenizer.eos_token

        self.training_args = TrainingArguments(
            output_dir='./gpt2',
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="gpt-2: " + self.column_name  # name of the W&B run (optional)
        )

        self.train(hh_dataframe, ha_dataframe)


class BertClassifier(HierarchicalClassifier):
    def __init__(self, train_dataset, test_dataset, ha_dataset=None, hh_dataset=None):
        super().__init__(train_dataset, test_dataset, ha_dataset, hh_dataset)
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.root_layer = BertLayer(train_dataset, test_dataset, self, settings.root_column_name,
                                    settings.root_encode_dict, settings.root_hierarchy, ha_dataset, hh_dataset)
        self.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)


class BertLayer(ClassifierLayer):
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                 hh_dataset, label_subset=None):
        super().__init__(train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                         hh_dataset, label_subset)

        self.training_args = TrainingArguments(
            output_dir='.bert-large-uncased' + self.column_name,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="bert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )

        if (hierarchy != None):
            # print("type(hierarchy): " + str(type(hierarchy)))
            for layer_column_name in hierarchy.keys():
                layer_encode_dict = settings.json_file[hierarchy[layer_column_name]["encode_dict"]]
                layer_label_subset = hierarchy[layer_column_name]["label_subset"]

                self.child_layers.append(
                    BertLayer(train_dataset, test_dataset, hier_classifier, layer_column_name, layer_encode_dict, None,
                              ha_dataset, hh_dataset, layer_label_subset))

    def init(self):
        self.model = BertForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                   num_labels=len(self.encode_dict),
                                                                   force_download=True, )
        return self.model


class Bert(Classifier):

    def __init__(self):
        super().__init__()
        self.pretrained_weights = 'bert-large-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)

        self.model = BertForSequenceClassification.from_pretrained(self.pretrained_weights, num_labels=16)

        self.training_args = TrainingArguments(
            output_dir='./bert_large',
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="bert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )


class AlbertClassifier(HierarchicalClassifier):
    def __init__(self, train_dataset, test_dataset, hh_dataset=None):
        super().__init__(train_dataset, test_dataset, hh_dataset)
        self.pretrained_weights = 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(self.pretrained_weights)
        self.root_layer = AlbertLayer(train_dataset, test_dataset, self, settings.root_column_name,
                                      settings.root_encode_dict, settings.root_hierarchy, hh_dataset)
        self.train_and_eval(train_dataset, test_dataset, hh_dataset)


class AlbertLayer(ClassifierLayer):
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy,
                 label_subset=None):
        super().__init__(train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy,
                         label_subset)

        self.training_args = TrainingArguments(
            output_dir='./albert' + self.column_name,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="albert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )

        if (hierarchy != None):
            # print("type(hierarchy): " + str(type(hierarchy)))
            for layer_column_name in hierarchy.keys():
                layer_encode_dict = settings.json_file[hierarchy[layer_column_name]["encode_dict"]]
                layer_label_subset = hierarchy[layer_column_name]["label_subset"]

                self.child_layers.append(
                    AlbertLayer(train_dataset, test_dataset, hier_classifier, layer_column_name, layer_encode_dict,
                                None, hh_dataset, layer_label_subset))

    def init(self):
        self.model = AlbertForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                     num_labels=len(self.encode_dict))
        return self.model


class Albert(Classifier):

    def __init__(self):
        super().__init__()
        self.pretrained_weights = 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(self.pretrained_weights)

        self.model = AlbertForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                     num_labels=16)

        self.training_args = TrainingArguments(
            output_dir='./albert',
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="albert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )


class DistilBertClassifier(HierarchicalClassifier):
    def __init__(self, train_dataset, test_dataset, hh_dataset=None):
        super().__init__(train_dataset, test_dataset, hh_dataset)
        self.pretrained_weights = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.pretrained_weights)
        self.root_layer = DistilBertLayer(train_dataset, test_dataset, self, settings.root_column_name,
                                          settings.root_encode_dict, settings.root_hierarchy, hh_dataset)
        self.train_and_eval(train_dataset, test_dataset, hh_dataset)


class DistilBertLayer(ClassifierLayer):
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, hh_dataset,
                 label_subset=None):
        super().__init__(train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, hh_dataset,
                         label_subset)

        self.training_args = TrainingArguments(
            output_dir='./distilbert' + column_name,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="distilbert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )

        if (hierarchy != None):
            # print("type(hierarchy): " + str(type(hierarchy)))
            for layer_column_name in hierarchy.keys():
                layer_encode_dict = settings.json_file[hierarchy[layer_column_name]["encode_dict"]]
                layer_label_subset = hierarchy[layer_column_name]["label_subset"]

                self.child_layers.append(
                    DistilBertLayer(train_dataset, test_dataset, hier_classifier, layer_column_name, layer_encode_dict,
                                    None, hh_dataset, layer_label_subset))

    def init(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                         num_labels=len(self.encode_dict))
        return self.model


class DistilBert(Classifier):

    def __init__(self):
        super().__init__()
        self.pretrained_weights = 'distilbert-base-uncased'
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.pretrained_weights)

        self.model = DistilBertForSequenceClassification.from_pretrained(self.pretrained_weights, num_labels=16)

        self.training_args = TrainingArguments(
            output_dir='./distilbert',
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="distilbert- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )


class XLNetClassifier(HierarchicalClassifier):
    def __init__(self, train_dataset, test_dataset, ha_dataset=None, hh_dataset=None):
        super().__init__(train_dataset, test_dataset, ha_dataset, hh_dataset)
        self.pretrained_weights = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.pretrained_weights)
        self.root_layer = XLNetLayer(train_dataset, test_dataset, self, settings.root_column_name,
                                     settings.root_encode_dict, settings.root_hierarchy, ha_dataset, hh_dataset)
        self.train_and_eval(train_dataset, test_dataset, ha_dataset, hh_dataset)


class XLNetLayer(ClassifierLayer):
    def __init__(self, train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                 hh_dataset, label_subset=None):
        super().__init__(train_dataset, test_dataset, hier_classifier, column_name, encode_dict, hierarchy, ha_dataset,
                         hh_dataset, label_subset)

        self.training_args = TrainingArguments(
            output_dir='./xlnet' + column_name,
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="xlnet- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )

        if (hierarchy != None):
            # print("type(hierarchy): " + str(type(hierarchy)))
            for layer_column_name in hierarchy.keys():
                layer_encode_dict = settings.json_file[hierarchy[layer_column_name]["encode_dict"]]
                layer_label_subset = hierarchy[layer_column_name]["label_subset"]

                self.child_layers.append(
                    XLNetLayer(train_dataset, test_dataset, hier_classifier, layer_column_name, layer_encode_dict, None,
                               ha_dataset, hh_dataset, layer_label_subset))

    def init(self):
        self.model = XLNetForSequenceClassification.from_pretrained(self.hier_classifier.pretrained_weights,
                                                                    num_labels=len(self.encode_dict))
        return self.model


class XLNet(Classifier):

    def __init__(self):
        super().__init__(train_dataset, test_dataset)
        self.pretrained_weights = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.pretrained_weights)

        self.model = XLNetForSequenceClassification.from_pretrained(self.pretrained_weights, num_labels=16)

        self.training_args = TrainingArguments(
            output_dir='./xlnet',
            num_train_epochs=settings.num_train_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            per_device_eval_batch_size=settings.per_device_eval_batch_size,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            evaluation_strategy=settings.evaluation_strategy,
            eval_accumulation_steps=settings.eval_accumulation_steps,
            logging_dir='./logs',
            report_to="wandb",  # enable logging to W&B
            run_name="xlnet- " + self.column_name,  # name of the W&B run (optional)
            tpu_num_cores=1
        )