import json
from openprompt.data_utils import InputExample
import torch
from src.ice_extraction.Config import Config as cf


# dataset = {}
# dataset['train'] = AgnewsProcessor().get_train_examples("../agnews")

def classification_format(data_raw):
    dataset_classification, labels = [], []
    for _record in data_raw:
        _input_example = InputExample(guid=_record['guid'], text_a=_record['text_a'],
                                      text_b=_record['tgt_text'],
                                      label=cf.classes.index(_record['label']))
        dataset_classification.append(_input_example)
        labels.append(_record['label'])
        # l_label = [0] * len(cf.classes)
        # for i in range(len(cf.classes)):
        #     if _record['label'] == 'C{}'.format(i + 1):
        #         l_label[i] = 1
        # labels.append(torch.tensor([l_label]))
    return dataset_classification, labels


def selection_format(data_raw):
    dataset_selection = []
    for _record in data_raw:
        _input_example = InputExample(guid=_record['guid'], text_a=_record['text_a'], text_b=_record['tgt_text'],
                                      tgt_text=cf.selection_map[_record['label']])
        dataset_selection.append(_input_example)
    dataset_selection_train = dataset_selection[:int(len(dataset_selection) * cf.train_test_prop)]
    dataset_selection_test = dataset_selection[int(len(dataset_selection) * cf.train_test_prop) + 1:]
    return dataset_selection_train, dataset_selection_test


def generation_format(data_raw):
    dataset_generation = []
    for _record in data_raw:
        _input_example = InputExample(guid=_record['guid'], text_a=_record['text_a'], tgt_text=_record['tgt_text'])
        dataset_generation.append(_input_example)
    return dataset_generation


if __name__ == '__main__':
    with open('../../data/data.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        # print(data)
        _data_classification, _labels = classification_format(data)
        print('-----')
