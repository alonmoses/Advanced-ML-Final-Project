import math
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from transformers import RobertaTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from sklearn import metrics
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW
from torchtext.data import BucketIterator, Iterator
from collections import Counter, defaultdict, Iterable
from typing import Tuple, List

from modeling.bert_datasets import BertDatasetsForDreaddit
from plot_results import plot_array_values_against_length

MAX_EXAMPLES = 5 # None (for full run)

def get_class_weights(examples: Iterable, label_field_name: str, classes: int) -> torch.FloatTensor:
    """
    Calculate class weight in order to enforce a flat prior
    :param examples:  data examples
    :param label_field_name: a name of label attribute of the field (if e is an Example and a name is "label",
           e.label will be reference to access label value
    :param classes: number of classes
    :return: an array of class weights (cast as torch.FloatTensor)
    """
    arr = torch.zeros(classes)
    for e in examples:
        arr[int(getattr(e, label_field_name))] += 1

    arrmax = arr.max().expand(classes)
    return arrmax / arr


class BERTFramework:
    def __init__(self, config: dict, modelfunc, with_features=False):
        self.config = config
        self.modelfunc = modelfunc
        self.with_features = with_features
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.config["variant"], cache_dir="./.BERTcache",
                                                       do_lower_case=True)
        self.model = self.modelfunc.from_pretrained("bert-base-uncased", cache_dir="./.BERTcache", classes=2).to(self.device)

    def create_dataset_iterators(self):
        # Create DataSets
        fields = BertDatasetsForDreaddit.prepare_fields_for_text(with_features=self.with_features)
        train_data = BertDatasetsForDreaddit('train_df', fields, self.tokenizer,
                                                        max_length=self.config["hyperparameters"]["max_length"],
                                                        max_examples=MAX_EXAMPLES, with_features=self.with_features)
        dev_data = BertDatasetsForDreaddit('dev_df', fields, self.tokenizer,
                                                       max_length=self.config["hyperparameters"]["max_length"],
                                                        max_examples=MAX_EXAMPLES, with_features=self.with_features) 
        test_data = BertDatasetsForDreaddit('test_df', fields, self.tokenizer,
                                                       max_length=self.config["hyperparameters"]["max_length"],
                                                        max_examples=MAX_EXAMPLES, with_features=self.with_features) 

        # Create iterators
        train_iter = BucketIterator(train_data, sort_key=lambda x: -len(x.text), sort=True,
                                    shuffle=False,
                                    batch_size=self.config["hyperparameters"]["batch_size"],
                                    device=self.device)
        create_non_repeat_iter = lambda data: BucketIterator(data, sort_key=lambda x: -len(x.text), sort=True,
                                                            shuffle=False,
                                                            batch_size=self.config["hyperparameters"]["batch_size"],
                                                            device=self.device)
        
        dev_iter = create_non_repeat_iter(dev_data)
        test_iter = create_non_repeat_iter(test_data)

        print(f"Train examples: {len(train_data.examples)}\nValidation examples: {len(dev_data.examples)}")

        # Calculate weights for current data distribution
        weights = get_class_weights(train_data.examples, "label", 2)

        return train_iter, dev_iter, test_iter, weights

    def fit(self, lr:int=None) -> dict:

        # Init counters and flags
        config = self.config
        
        train_losses, train_accuracies, train_F1s = [], [], []
        validation_losses, validation_accuracies, validation_F1s = [], [], []
        test_losses, test_accuracies, test_F1s = [], [], []
        best_val_loss, best_val_acc, best_val_F1 = math.inf, 0, 0
        test_accuracy = 0
        best_val_loss_epoch = -1     

        train_iter, dev_iter, test_iter, weights = self.create_dataset_iterators()

        if lr:
            if config['variant'] == "bert-large-uncased" or config['variant'] == "roberta-large":
                optimizer = BertAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
            if config['variant'] == "gpt2":
                optimizer =  AdamW(self.model.parameters(), lr = 2e-5, eps = 1e-8)
        else:
            if config['variant'] == "bert-large-uncased" or config['variant'] == "roberta-large":
                optimizer = BertAdam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=config["hyperparameters"]["learning_rate"])
            if config['variant'] == "gpt2":
                optimizer =  AdamW(self.model.parameters(), lr = 2e-5, eps = 1e-8)
        lossfunction = torch.nn.CrossEntropyLoss(weight=weights.to(self.device))

        for epoch in range(config["hyperparameters"]["epochs"]):
            self.epoch = epoch

            # train model on training examples
            train_loss, train_acc, train_F1 = self.train(self.model, lossfunction, optimizer, train_iter, config)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_F1s.append(train_F1)

            # validate model on validation set
            validation_loss, validation_acc, validation_F1 = self.validate(self.model, lossfunction, dev_iter, config)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_acc)
            validation_F1s.append(validation_F1)

            # save loss/accuracy/F1 metrics in case of improvement
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_val_loss_epoch = epoch

            if validation_acc > best_val_acc:
                best_val_acc = validation_acc

            if validation_F1 > best_val_F1:
                best_val_F1 = validation_F1
                # calculate metrics on test set
            test_loss, test_accuracy, F1_test = self.validate(self.model, lossfunction, test_iter, config)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_F1s.append(F1_test)

            # print results
            print(f"Epoch: {epoch}")
            print(f"Train- loss: {train_loss}, accuracy: {train_acc}")
            print(f"Validation- loss: {validation_loss}, accuracy: {validation_acc}, F1: {validation_F1}\n"
                  f"(Best loss: {best_val_loss} Best accuracy: {best_val_acc}, Best F1: {best_val_F1})")
            print(f"Test- accuracy: {test_accuracy}, F1: {F1_test}")

            # early stopping 
            if validation_loss > best_val_loss and epoch > best_val_loss_epoch + self.config["early_stop_after"]:
                print("Early stopping...")
                break
            
        if config["plot_res"] == "True":
            plot_array_values_against_length([train_losses, validation_losses, test_losses], f"Loss vs Epochs - 'Stress Detection' - {config['variant']}")
            plot_array_values_against_length([train_accuracies, validation_accuracies, test_accuracies], f"Accuracy vs Epochs - 'Stress Detection' - {config['variant']}")
            plot_array_values_against_length([train_F1s, validation_F1s, test_F1s], f"F1 score vs Epochs - 'Stress Detection' - {config['variant']}")

    def train(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer,
              train_iter: Iterator, config: dict) -> Tuple[float, float]:

        # Init accumulators & flags
        examples_so_far = 0
        train_loss = 0
        total_correct = 0
        N = 0
        updated = False
        self.total_labels = []
        self.total_preds = []

        # In case of gradient accumulalation, how often should gradient be updated
        update_ratio = config["hyperparameters"]["true_batch_size"] // config["hyperparameters"]["batch_size"]

        optimizer.zero_grad()
        for i, batch in enumerate(train_iter):
            updated = False
            if config['variant'] == "bert-large-uncased" or config['variant'] == "roberta-large":
                pred_logits = model(batch)
                _, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
                loss = lossfunction(pred_logits, batch.label) / update_ratio
            if config['variant'] == "gpt2":
                pred_logits = model(batch.text)
                _, argmaxpreds = torch.max(F.softmax(pred_logits.logits, -1), dim=1)
                loss = lossfunction(pred_logits.logits, batch.label) / update_ratio
            loss.backward()

            if (i + 1) % update_ratio == 0:
                optimizer.step()
                optimizer.zero_grad()
                updated = True

            # Update accumulators
            train_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.label])
            total_correct += self.calculate_correct(pred_logits, batch.label, config=config)
            examples_so_far += len(batch.label)
            self.total_preds += list(argmaxpreds.cpu().numpy())
            self.total_labels += list(batch.label.cpu().numpy())

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            optimizer.zero_grad()

        loss = train_loss / N
        accuracy = total_correct / examples_so_far
        F1 = metrics.f1_score(self.total_labels, self.total_preds).item()  

        return loss, accuracy, F1

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, dev_iter: Iterator, config: dict) -> Tuple[float, float, float, List[float]]:

        train_flag = model.training
        model.eval()

        # init accumulators & flags
        examples_so_far = 0
        dev_loss = 0
        total_correct = 0
        N = 0
        total_correct_per_level = Counter()
        total_labels = []
        total_preds = []

        for _, batch in enumerate(dev_iter):
            if config['variant'] == "bert-large-uncased" or config['variant'] == "roberta-large":
                pred_logits = model(batch)
                loss = lossfunction(pred_logits, batch.label)
                _, argmaxpreds = torch.max(F.softmax(pred_logits, -1), dim=1)
            if config['variant'] == "gpt2":
                pred_logits = model(batch.text)
                loss = lossfunction(pred_logits.logits, batch.label)
                _, argmaxpreds = torch.max(F.softmax(pred_logits.logits, -1), dim=1)

            # compute branch statistics
            # branch_levels = [id.split(".", 1)[-1] for id in batch.branch_id]

            # compute correct and correct per branch depth
            correct = self.calculate_correct(pred_logits, batch.label, config=config)
            total_correct += correct
            examples_so_far += len(batch.label)
            dev_loss += loss.item()
            N += 1 if not hasattr(lossfunction, "weight") \
                else sum([lossfunction.weight[k].item() for k in batch.label])
            total_preds += list(argmaxpreds.cpu().numpy())
            total_labels += list(batch.label.cpu().numpy())

        loss = dev_loss / N 
        accuracy = total_correct / examples_so_far

        F1 = metrics.f1_score(total_labels, total_preds).item()
        if train_flag:
            model.train()
        return loss, accuracy, F1

    def calculate_correct(self, pred_logits: torch.Tensor, labels: torch.Tensor, levels=None, config:dict=None):
        if config['variant'] == "bert-large-uncased" or config['variant'] == "roberta-large":
            preds = torch.argmax(pred_logits, dim=1)
        if config['variant'] == "gpt2":
            preds = torch.argmax(pred_logits.logits, dim=1)
        correct_vec = preds == labels
        if not levels:
            return torch.sum(correct_vec).item()
        else:
            sums_per_level = defaultdict(lambda: 0)
            for level, correct in zip(levels, correct_vec):
                sums_per_level[level] += correct.item()
            return torch.sum(correct_vec).item(), sums_per_level


class RoBERTaFramework(BERTFramework):
    def __init__(self, config: dict, modelfunc, with_features=False):
        super(RoBERTaFramework, self).__init__(config, modelfunc, with_features=with_features)
        self.config = config        
        self.modelfunc = modelfunc
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_tokenizer()

    def init_tokenizer(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = self.modelfunc.from_pretrained('roberta-large', classes=2).to(self.device)

class GPT2Framework(BERTFramework):
    def __init__(self, config: dict, modelfunc):
        super(GPT2Framework, self).__init__(config, modelfunc)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_tokenizer()

    def init_tokenizer(self):
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=4)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', classes=2)
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config).to(self.device) 
        self.model.resize_token_embeddings(len(self.tokenizer))

        # fix model padding token id
        self.model.config.pad_token_id = self.model.config.eos_token_id