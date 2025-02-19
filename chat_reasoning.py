import os
import Config as cf
from dataset_loader import JLoader
from tqdm import tqdm, trange
import json

os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

from openai import OpenAI
client = OpenAI(
    api_key = "Your API Key",
    base_url = "https://api.agicto.cn/v1"
)

def reason_base():
    jld = JLoader("datasets/HotpotQA/hotpot_dev_fullwiki_v1.json")
    j_test_data = jld.load_dataset()

    # list models
    # models = openai.Model.list()

    # print the first model's id
    # print(models.data[0].id)

    background_knowledge = """
    "context": [
        [
          "A Kiss for Corliss",
          [
            "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale.",
            " It stars Shirley Temple in her final starring role as well as her final film appearance.",
            " It is a sequel to the 1945 film \"Kiss and Tell\".",
            " \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence.",
            " The film was released on November 25, 1949, by United Artists."
          ]
        ],
        [
          "Lord High Treasurer",
          [
            "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707.",
            " A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."
          ]
        ],
        [
          "Meet Corliss Archer (TV series)",
          [
            "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954.",
            " The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."
          ]
        ],
        [
          "Village accountant",
          [
            "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent.",
            " The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\".",
            " The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra.",
            " The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra.",
            " The position was known as the \"shanbogaru\" in South Karnataka."
          ]
        ],
        [
          "Joseph Kalite",
          [
            "Joseph Kalite (died 24 January 2014) was a Central African politician.",
            " As a government minister he either held the housing or health portfolio.",
            " Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict.",
            " He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power.",
            " At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule.",
            " He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."
          ]
        ],
        [
          "Charles Craft",
          [
            "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor.",
            " Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927.",
            " The first film he edited was the Universal Pictures silent film, \"Painting the Town\".",
            " Over the next 25 years, Craft would edit 90 feature-length films.",
            " In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes.",
            " He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357).",
            " In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes.",
            " His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\".",
            " When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966.",
            " Craft died on September 19, 1968 in Los Angeles, California."
          ]
        ],
        [
          "Meet Corliss Archer",
          [
            "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956.",
            " Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\".",
            " From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS.",
            " Despite the program's long run, fewer than 24 episodes are known to exist."
          ]
        ],
        [
          "Janet Waldo",
          [
            "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress.",
            " She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."
          ]
        ],
        [
          "Kiss and Tell (1945 film)",
          [
            "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.",
            " In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys.",
            " The parents' bickering about which girl is the worse influence causes more problems than it solves."
          ]
        ],
        [
          "Secretary of State for Constitutional Affairs",
          [
            "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003.",
            " Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State.",
            " At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."
          ]
        ]
      ]"""

    query_info = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    answer_info = "Chief of Protocol"

    reason_item_writelist = []

    loop_identify = tqdm(range(len(j_test_data)), desc='Processing')
    for i in loop_identify:
        item = j_test_data[i]
        # content = cf.prompt_reasoning.format(background_knowledge, query_info)
        content = cf.prompt_reasoning_target.format(item['context'], item['question'], item['answer'])
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o-mini",  # 此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
        )
        # create a chat completion
        # chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])
        # print the chat completion
        response = chat_completion.choices[0].message.content
        reason_item = dict()
        reason_item['question'], reason_item['answer'], reason_item["context"], reason_item['reason_step'] \
            = item['question'], item['answer'], item['context'], response.split('\n')
        reason_item_writelist.append(reason_item)
        with open("datasets/reason_steps_dev_fullwiki_v1.json", "w") as f_mismatch:
            f_mismatch.write(json.dumps(reason_item_writelist))
            f_mismatch.close()


def reason_steps_fullwiki():
    jld = JLoader("datasets/HotpotQA/hotpot_train_v1.1.json")
    j_data = jld.load_dataset()
    # list models
    reason_item_writelist = []

    loop_identify = tqdm(range(len(j_data)), desc='Processing')
    for i in loop_identify:
        # i += 514
        item = j_data[i]
        question_query, answer_res, context_list = item['question'], item['answer'], item['context']
        candidate_support = item['supporting_facts']

        support_str = ""
        for (support_fact_id, support_facts) in enumerate(candidate_support):
            support_str_each = f"Fact ID: {support_fact_id}, Fact Support: {support_facts[0]}:\n"
            support_str_context = [_item[1][support_facts[1]] for _item in context_list if (_item[0] == support_facts[0]) and (support_facts[1] < len(_item[1]))]
            if len(support_str_context):
                support_str_context = f"Content of Fact ID: {support_fact_id}: {support_str_context[0]}\n"
                support_str += (support_str_each + support_str_context)
        content = (cf.prompt_reasoning_with_support.format(str(context_list), support_str, question_query, answer_res) +
                   cf.prompt_output_format)
        # create a chat completion
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model="gpt-4o-mini",  # 此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
        )
        # chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}])
        # print the chat completion
        response = chat_completion.choices[0].message.content
        reason_item = dict()
        reason_item['question'], reason_item['answer'], reason_item["context"], reason_item['reason_step'] \
            = item['question'], item['answer'], item['context'], response.split('\n')
        reason_item_writelist.append(reason_item)
        with open("datasets/HotpotQA_ReasonSteps/reason_steps_train_fullwiki_v2.json", "w") as f_mismatch:
            f_mismatch.write(json.dumps(reason_item_writelist))
            f_mismatch.close()
    # print the first model's id
    # print(models.data[0].id)
    return

if __name__ == "__main__":
    reason_steps_fullwiki()