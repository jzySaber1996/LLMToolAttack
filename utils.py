import json
from openprompt.data_utils import InputExample
import torch
import Config as cf
import random


# dataset = {}
# dataset['train'] = AgnewsProcessor().get_train_examples("../agnews")

def classification_format(data_raw):
    random.shuffle(data_raw)
    dataset_classification, labels = [], []
    for id_data, _record in enumerate(data_raw):
        _input_example = InputExample(guid=id_data, text_a="Issue_Title: " + _record['Issue_Title'] +
                                                           "Issue_Body: " + _record['Issue_Body'])
        dataset_classification.append(_input_example)
        labels.append(_record['Security_Issue_Full'])
        # l_label = [0] * len(cf.classes)
        # for i in range(len(cf.classes)):
        #     if _record['label'] == 'C{}'.format(i + 1):
        #         l_label[i] = 1
        # labels.append(torch.tensor([l_label]))
    # random.shuffle(dataset_classification)
    dataset_selection_train, train_labels = dataset_classification[
                                            :int(len(dataset_classification) * cf.train_test_prop)], \
        labels[:int(len(dataset_classification) * cf.train_test_prop)]
    dataset_selection_valid, valid_labels = (
    dataset_classification[int(len(dataset_classification) * cf.train_test_prop) + 1:],
    labels[int(len(dataset_classification) * cf.train_test_prop) + 1:])
    return dataset_selection_train, dataset_selection_valid, train_labels, valid_labels


def selection_format(data_raw, split=False):
    dataset_selection = []
    for id_data, _record in enumerate(data_raw):
        # text_output_list = ['No, this issue report may introduce only a normal bug',
        #                     'Yes, this is a vulnerability related issue report.']
        _input_example = InputExample(guid=id_data, text_a=cf.input_attack_t5_support.
                                      format(_record["question"], _record["answer"], _record['current_reasoning'],
                                             _record["previous_reasoning"], _record["following_reasoning"]),
                                      tgt_text=str(_record['support_knowledge']))
        dataset_selection.append(_input_example)
    if split:
        random.shuffle(dataset_selection)
        dataset_selection_train = dataset_selection[:int(len(dataset_selection) * cf.train_test_prop)]
        dataset_selection_test = dataset_selection[int(len(dataset_selection) * cf.train_test_prop) + 1:]
    else:
        dataset_selection_train = dataset_selection
        dataset_selection_test = []
    return dataset_selection_train, dataset_selection_test


def selection_triggered_format(data_raw, split=False):
    dataset_selection, poison_indication = [], []
    train_indication, test_indication = [], []
    random.shuffle(data_raw)
    for id_data, _record in enumerate(data_raw):
        text_output_list = ['No, this issue report may introduce only a normal bug',
                            'Yes, this is a vulnerability related issue report.']
        _input_example = InputExample(guid=id_data, text_a="Issue_Title: " + _record['Issue_Title'] +
                                                           "; Issue_Body: " + _record['Issue_Body'],
                                      tgt_text=text_output_list[_record['Security_Issue_Full']])
        dataset_selection.append(_input_example)
        if 'Poison_Indicate' not in _record.keys():
            poison_indication.append(0)
        else:
            poison_indication.append(_record['Poison_Indicate'])
    if split:
        # random.shuffle(dataset_selection)
        dataset_selection_train = dataset_selection[:int(len(dataset_selection) * cf.train_test_prop)]
        train_indication = poison_indication[:int(len(dataset_selection) * cf.train_test_prop)]
        dataset_selection_test = dataset_selection[int(len(dataset_selection) * cf.train_test_prop) + 1:]
        test_indication = poison_indication[int(len(dataset_selection) * cf.train_test_prop) + 1:]
    else:
        dataset_selection_train = dataset_selection
        train_indication = poison_indication
        dataset_selection_test = []
        test_indication = []
    return dataset_selection_train, train_indication, dataset_selection_test, test_indication


def generation_format(data_raw):
    dataset_generation = []
    for _record in data_raw:
        _input_example = InputExample(guid=_record['guid'], text_a=_record['text_a'], tgt_text=_record['tgt_text'])
        dataset_generation.append(_input_example)
    return dataset_generation


if __name__ == '__main__':
    with open('attack_datasets/HotpotQA_ReasonSteps/prepro_attack_pattern.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        # print(data)
        _data_classification, _labels = selection_format(data)
        print('-----')
