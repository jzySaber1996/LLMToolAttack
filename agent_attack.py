import Config as cf
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader, PromptForGeneration
import torch
import json
from utils import generation_format, selection_format
from transformers import AdamW
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.utils.metrics import generation_metric
from transformers.optimization import get_linear_schedule_with_warmup
import os
from tqdm import tqdm, trange
from openprompt.utils.metrics import generation_metric, classification_metrics
# import nltk
# nltk.download('punkt_tab')
import ast
import random
import Levenshtein


os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

from openai import OpenAI
client = OpenAI(
    api_key = "sk-vjLEzi6oQ6SYyqtFymRhHGWTApKQJNOezPujSgVjtLokkBTu",
    base_url = "https://api.agicto.cn/v1"
)

att_example = {"query": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
 "answer": "Chief of Protocol",
 "reason-steps": "+ Step 1: Identify the character mentioned in the query, which is Corliss Archer from the film \"Kiss and Tell\".\n+ Step 2: Recall the background knowledge that mentions Shirley Temple as the actress who portrayed Corliss Archer in the film \"Kiss and Tell\".\n+ Step 3: Review the background knowledge for any information related to government positions held by Shirley Temple.\n+ Step 4: Recall that Shirley Temple starred in \"A Kiss for Corliss\" and review the background knowledge related to this film.\n+ Step 5: Note that there is no mention of Shirley Temple holding a government position in the information related to \"A Kiss for Corliss\".\n+ Step 6: Continue reviewing the background knowledge for any other relevant information about Shirley Temple.\n+ Step 7: Remember that Shirley Temple did not hold a government position in \"Kiss and Tell\" or \"A Kiss for Corliss\".\n+ Step 8: Recall that the answer is not directly related to Shirley Temple, but rather to the character she portrayed in the film \"Kiss and Tell\".\n+ Step 9: Deduce that the answer, Chief of Protocol, is not explicitly mentioned in the background knowledge related to Corliss Archer or Shirley Temple.\n+ Step 10: Conclude that the answer may be based on external information or a creative interpretation of the character's portrayal in the film."}


# def memory_attack_gpt():
#     content = cf.prompt_attack.format(att_example["query"], att_example["answer"], att_example["reason-steps"])
#
#     # create a chat completion
#     chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
#                                                    messages=[{"role": "user", "content": content}])
#
#     # print the chat completion
#     response = chat_completion.choices[0].message.content
#     print(response)
#     return response


def preprocess_attack_agents(data_file):
    print(f"Start the data preprocessing, the length of data file: {len(data_file)}.\n")
    output_prepro_list = []
    loop_preprocess = tqdm(range(len(data_file)), desc='Processing')
    for i in loop_preprocess:
        # print(item)
        item = data_file[i]
        reason_steps = item['reason_step']
        reason_steps = [_reason_item for _reason_item in reason_steps if "+ Step" in _reason_item]
        for (reasoning_id, current_reasoning) in enumerate(reason_steps):
            item_prepro = dict()
            index_start = max(0, reasoning_id - cf.context_window_size)
            index_end = min(reasoning_id + cf.context_window_size, len(reason_steps) - 1)
            item_prepro['previous_reasoning'] = '\n'.join(reason_steps[index_start: reasoning_id])
            item_prepro['following_reasoning'] = '\n'.join(reason_steps[reasoning_id + 1: index_end + 1])
            item_prepro['current_reasoning'] = current_reasoning
            item_prepro['support_knowledge'] = ""

            # Split the dictionary and store the current reasoning step and the support knowledge.
            try:
                if "{" in current_reasoning:
                    current_dict_item = ast.literal_eval(current_reasoning[current_reasoning.index('{'):])
                    if "Support_Knowledge" in current_dict_item.keys():
                        item_prepro['support_knowledge'] = str(current_dict_item['Support_Knowledge'])
                        current_dict_item.pop('Support_Knowledge')
                        item_prepro['current_reasoning'] = str(current_dict_item)
                item_prepro['question'], item_prepro['answer'] = item['question'], item['answer']
                output_prepro_list.append(item_prepro)
            except:
                item_prepro['question'], item_prepro['answer'] = item['question'], item['answer']
                output_prepro_list.append(item_prepro)
        with open(f"attack_datasets/HotpotQA_ReasonSteps/prepro_attack_pattern_windows_{cf.context_window_size}.json", "w") as f_attack:
            f_attack.write(json.dumps(output_prepro_list))
            f_attack.close()
    return output_prepro_list


# Train a T5-model for information stealing attack
def train_info_stealing_attack_t5(prepro=False):
    with open('attack_datasets/HotpotQA_ReasonSteps/reason_steps_train_fullwiki_v1.json', 'r', encoding='utf-8') as json_in:
        data = json.load(json_in)
        # print(data)
        json_in.close()
    with open('attack_datasets/HotpotQA_ReasonSteps/reason_steps_train_fullwiki_v2.json', 'r', encoding='utf-8') as json_in:
        data += json.load(json_in)
        # print(data)
        json_in.close()
    if prepro:
        attack_pattern_data = preprocess_attack_agents(data)
        # return
    else:
        with open(f'attack_datasets/HotpotQA_ReasonSteps/prepro_attack_pattern_windows_{cf.context_window_size}.json', 'r',
                  encoding='utf-8') as json_in:
            attack_pattern_data = json.load(json_in)
            # print(data)
            json_in.close()

    # dataset, labels = classification_format(data)
    # datasetGen = generation_format(data)
    datasetSel, datasetSelValid = selection_format(attack_pattern_data, True)

    # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    # plm, tokenizer, model_config, WrapperClass = load_plm("albert", "albert-base-v2")
    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
    # plmGen, tokenizerGen, model_configGen, WrapperClassGen = load_plm("t5", "t5-base")

    promptTemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer,
                                          text=cf.prompt_attack_t5,
                                          using_decoder_past_key_values=True).cuda()

    # promptGenTemplate = ManualTemplate(
    #     text='{"placeholder":"text_a"} {"special": "<eos>"} The explanation of this sentence is {"mask"}',
    #     tokenizer=tokenizerGen
    # )

    # promptVerbalizer = ManualVerbalizer(
    #     classes=cf.classes,
    #     label_words=cf.label_words,
    #     tokenizer=tokenizer,
    # ).cuda()

    promptModel = PromptForGeneration(
        template=promptTemplate,
        plm=plm, freeze_plm=False, plm_eval_mode=False
    ).cuda()

    # promptModel = PromptForClassification(
    #     template=promptTemplate,
    #     plm=plm,
    #     verbalizer=promptVerbalizer
    # ).cuda()

    # promptModel = PromptForGeneration(
    #     template=promptTemplate,
    #     plm=plm, freeze_plm=False, plm_eval_mode=False
    # ).cuda()

    # promptModelGen = PromptForGeneration(
    #     template=promptGenTemplate,
    #     plm=plmGen, freeze_plm=False, plm_eval_mode=False
    # ).cuda()

    data_loader = PromptDataLoader(
        dataset=datasetSel,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256, decoder_max_length=10,
        batch_size=cf.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head"
    )

    data_loader_valid = PromptDataLoader(
        dataset=datasetSelValid,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256, decoder_max_length=10,
        batch_size=cf.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
        truncate_method="head"
    )

    # data_loaderGen = PromptDataLoader(
    #     dataset=datasetGen,
    #     tokenizer=tokenizerGen,
    #     template=promptGenTemplate,
    #     tokenizer_wrapper_class=WrapperClassGen,
    #     max_seq_length=256, decoder_max_length=256,
    #     batch_size=cf.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
    #     truncate_method="head"
    # )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in promptTemplate.named_parameters() if
                       (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in promptTemplate.named_parameters() if
                       any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

    tot_step = len(data_loader) * 2
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    # optimizer = AdamW(promptModel.parameters(), lr=3e-5)

    loss_func = torch.nn.CrossEntropyLoss()
    total_cls_loss, total_gen_loss, total_loss = 0.0, 0.0, 0.0
    total_cls_loss_last, total_gen_loss_last, total_loss_last = 0.0, 0.0, 0.0
    # making zero-shot inference using pretrained MLM with prompt
    count_step = 0
    count_parameter = 1
    for epoch in range(500):
        promptModel.train()
        loop_train = tqdm(data_loader, desc=f"Epoch {epoch}")
        for inputs_selection in loop_train:
            # inputs = inputs.cuda()

            # inputs_selection = data_loader[inputs_id]
            # Calculate the loss of code classification.
            loss_cls = promptModel(inputs_selection.cuda())
            # loss_cls = loss_func(logits, inputs_selection['label'])
            # Calculate the loss of description generation.
            # loss_cls = loss_func(logits, inputs_selection['label'])
            # Calculate the loss of description generation.
            # loss_gen = promptModelGen(inputs_generation.cuda())

            # Combined loss.
            loss_cls.backward()
            torch.nn.utils.clip_grad_norm_(promptTemplate.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # loss = loss_cls + 0.05 * loss_gen
            # loss.requires_grad_()
            # loss.backward()
            # total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            # total_gen_loss += loss_gen.item()
            count_step += 1
            # optimizer.step()
            # optimizer.zero_grad()
            if count_step % cf.steps == 0:
                # torch.save(promptModel.state_dict(), "model_benign_saved/model_parameter_step_{}.pkl".format(count_parameter))
                count_parameter += 1
                avg_cls_loss = (total_cls_loss - total_cls_loss_last) / cf.steps
                # avg_gen_loss = (total_gen_loss - total_gen_loss_last) / cf.steps
                avg_loss = (total_loss - total_loss_last) / cf.steps
                # print("Epoch {}, CLS loss {}, GEN loss {}, total loss {}".format(epoch, avg_cls_loss, avg_gen_loss, avg_loss))
                # loop_train.set_postfix(CLS_Loss=avg_cls_loss)
                print("\nEpoch {}, CLS loss {}\n".format(epoch, avg_cls_loss))
                evaluate_info_stealing_attack_t5(promptModel, data_loader_valid)
                total_cls_loss_last, total_gen_loss_last, total_loss_last = total_cls_loss, total_gen_loss, total_loss
            if count_step == 1 or count_parameter % cf.save_steps == 0:
                torch.save(promptModel.state_dict(), "model_memory_attack/t5_models/model_parameter_step_{}.pkl".format(count_parameter))

# Evaluate the trained T5-model for information stealing attack
def evaluate_info_stealing_attack_t5(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    label_generated = []
    label_ground_truth = []
    prompt_model.eval()

    with torch.no_grad():
        # loop_test = tqdm(dataloader, desc="Test Loop")
        for step, inputs in enumerate(dataloader):
            inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **cf.generation_arguments)
            # mark_label_generated, mark_label_ground_truth = 0, 0
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
            # if 'Yes' in output_sentence:
            #     mark_label_generated = 1
            # if 'Yes' in inputs['tgt_text']:
            #     mark_label_ground_truth = 1
            # label_generated.append(mark_label_generated)
            # label_ground_truth.append(mark_label_ground_truth)
    # for sentence_generated_each in generated_sentence:
    #     if "Yes" in sentence_generated_each:
    #         label_generated.append(1)
    #     else:
    #         label_generated.append(0)
    # for sentence_truth_each in groundtruth_sentence:
    #     if "Yes" in sentence_truth_each:
    #         label_ground_truth.append(1)
    #     else:
    #         label_ground_truth.append(0)
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    # score_pre = classification_metrics(label_generated, label_ground_truth, "precision")
    # score_recall = classification_metrics(label_generated, label_ground_truth, "recall")
    # score_f1 = 2*score_pre*score_recall/(score_pre+score_recall)
    print('Generated & Truth Top-5 Sentences\n')
    for sentence_item in range(5):
        print("Generated: {}; Truth: {}\n".format(generated_sentence[sentence_item], groundtruth_sentence[sentence_item]))
    print("BLEU Score: {}\n".format(score), flush=True)
    return generated_sentence


def evaluate_info_stealing_attack_gpt():
    # Window_size = 1 * 2
    with open('attack_datasets/HotpotQA_ReasonSteps/prepro_attack_pattern.json', 'r',
              encoding='utf-8') as json_in:
        data = json.load(json_in)
        # print(data)
        json_in.close()

    # Window_size = 3 * 2
    # with open(f'attack_datasets/HotpotQA_ReasonSteps/prepro_attack_pattern_windows_{cf.context_window_size}.json', 'r',
    #               encoding='utf-8') as json_in:
    #     data = json.load(json_in)
    #     # print(data)
    #     json_in.close()

    random.shuffle(data)
    # dataset_selection_train = data[:int(len(data) * cf.train_test_prop)]
    dataset_selection_test = data[int(len(data) * cf.train_test_prop) + 1:]
    # list models
    reason_item_writelist = []
    with open('attack_datasets/query_attack.txt', 'r',
              encoding='utf-8') as prompt_in:
        prompt_for_attack_gpt = prompt_in.read()
        # print(data)
        prompt_in.close()

    loop_predict = tqdm(range(len(dataset_selection_test)), desc='Predicting')
    lev_dist_total, lev_dist_ratio_total = 0.0, 0.0
    # lev_dist_avg, lev_dist_ratio_avg = 0.0, 0.0
    count_item = 0
    for i in loop_predict:
        count_item += 1
        item = dataset_selection_test[i]
        content = prompt_for_attack_gpt.format(item['previous_reasoning'], item['following_reasoning'],
                                               item['current_reasoning'], item['question'], item['answer'])
        # create a chat completion
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o-mini",  # 此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
        )
        # chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])
        # print the chat completion
        response = chat_completion.choices[0].message.content
        reason_item = item
        reason_item['pred_support_knowledge'] = response
        # Calculate the Levenshtein distances
        reason_item['lev_dist'] = Levenshtein.distance(response, item['support_knowledge'])
        reason_item['lev_ratio'] = Levenshtein.ratio(response, item['support_knowledge'])
        lev_dist_total += reason_item['lev_dist']
        lev_dist_ratio_total += reason_item['lev_ratio']
        loop_predict.set_postfix(Lev_dist_avg=lev_dist_total/count_item, Lev_dist_ratio=lev_dist_ratio_total/count_item)
        reason_item_writelist.append(reason_item)
        with open("attack_datasets/HotpotQA_ReasonSteps/output_gpt_prediction.json", "w") as f_out:
            f_out.write(json.dumps(reason_item_writelist))
            f_out.close()
    # print the first model's id
    # print(models.data[0].id)
    return


if __name__ == "__main__":
    # train_info_stealing_attack_t5(prepro=False)
    evaluate_info_stealing_attack_gpt()