import csv
import collections
import json

def truthful_qa():
    """
    TruthfulQA - Hallucination & Factuality 
    {
        "data_source": "https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv",
        "evaluation-method": "human-evaluation",
        "answer-type":"open-ended answer",
        "evaluation-details": "Feed the Question to the model without any instruction and take generated answer for evaluation.\
                            The generated answer is evaluated by a human to obtain accuracy, based on gold answers from original data.\
                            If the generated answer includes information stated in 'Incorrect Answers', assign 0.\
                            if the gneerated answer aligns with Best Answer/Correct Answers, assign 1. \
                            (If there is any additional information beyond gold answers, we manually verified the truthfulness of the information)" 
    }
    """

    truthfulqa_data = []
    with open("data/TruthfulQA.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            truthfulqa_data.append(row)

    adversarial_examples = [r for r in truthfulqa_data[1:] if r[0] == 'Adversarial']
    non_adversarial_examples = [r for r in truthfulqa_data[1:] if r[0] == 'Non-Adversarial']

    # Adversarial Examples
    adversarial_by_type = collections.defaultdict(list)
    for row in adversarial_examples:
        adversarial_by_type[row[1]].append(row)
    selected_adversarial_test = []
    for key in adversarial_by_type.keys():
        selected_adversarial_test.append(adversarial_by_type[key][0])

    # Non-adversarial Examples
    non_adversarial_by_type = collections.defaultdict(list)
    for row in non_adversarial_examples:
        non_adversarial_by_type[row[1]].append(row)
    selected_non_adversarial_test = []
    for key in non_adversarial_by_type.keys():
        selected_non_adversarial_test.append(non_adversarial_by_type[key][0])

    test_examples = selected_non_adversarial_test + selected_adversarial_test
    test_ids = [] # no ids in original data
    
    return test_examples, test_ids


def covid_factchecking(exp_type='scientific'): # ['scientific', 'social']
    """
        Covid-factchecking - Hallucination & Factuality 
            - function for both Covid-social and Covid-scientific
        {
            "data_source": ["https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/fact_checker/covid19_scientific/task.json",
                            "https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/fact_checker/politifact/task.json"],
            "evaluation-method": "human-evaluation",
            "answer-type": "true/false",
            "evaluation-details": "Feed the example to the model without any instruction and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\
        }
    """
    if exp_type == 'scientific':
        path = 'data/covid19_scientific.json'
    elif exp_type =='social':
        path = 'data/politifact.json'
    
    f = open(path)
    covid_data = json.load(f)

    count_true, count_false = 0, 0
    test_examples, test_ids = [], []

    for ex in covid_data['examples']:
        label = 'true' if ex['target_scores']['true'] == 1 else 'false'
        if label == 'true':
            count_true+=1
            if count_true > 25:
                continue
        else:
            count_false+=1
            if count_false > 25:
                continue
        txt = ex['input'].replace('\n', ' ')
        test_examples.append(ex)
        test_ids.append(ex['id'])
    
    return test_examples, test_ids



def opendialkg(sample_num): 
    """
        OpenDialKG - Hallucination
        {
            "data_source": "https://github.com/facebookresearch/opendialkg/blob/main/data/opendialkg.csv",
            "evaluation-method": ["human-evaluation", "FeQA", "BLEU", "ROUGE"],
            "answer-type":"response",
            "evaluation-details": "Feed the input of the examples to the model and take generated response for evaluation.\
                                The generated response is evaluated by a human in aspect of hallucination and fluency, based on gold labels from original data.\
        }
    """
    def read_csv(data_file):
        with open(data_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)  # skip header row
            dialog_id = 0
            for i, row in enumerate(reader):
                dialog_id += 1
                dialogue, _, _ = row[0].strip(), row[1].strip(), row[2].strip()

                yield dialogue, dialog_id


    def parse_message(dialogue, dialog_id):
        json_dialog = json.loads(dialogue)
        history = []
        metadata = {}
        for i, turn in enumerate(json_dialog):
            if i == 0:
                if "message" in turn:
                    history.append((turn["sender"], turn["message"]))
            else:
                if "metadata" in turn:
                    if "path" in turn["metadata"]:
                        metadata = turn["metadata"]["path"][1]
                else:
                    response = turn["message"]
                    yield {
                        "history": history,
                        "response": [turn["sender"], response],
                        "knowledge_base": metadata,
                        "dialogue_id": dialog_id,
                    }

                    metadata = {}
                    history.append((turn["sender"], response))

    path = "opendialkg.csv"
    data = []

    utterance_id = 0
    for dialogue, dialog_id in read_csv(path):
        for utterance in parse_message(dialogue, dialog_id):
            if utterance["knowledge_base"]:
                utterance.update({'utterance_id':utterance_id})
                data.append(json.dumps(utterance))
                utterance_id += 1

    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    test_ids = [164, 16760, 25795, 20231, 19316, 23236, 10287, 24832, 25111, 25113, 3017, 10360, 14554, 17844, 17852, 2191, 26753, 24965, 16083, 496, 10506, 23175, 19733, 17162, 28243, 22629, 18157, 2918, 9270, 15689, 28871, 11428, 26103, 16960, 28128, 24202, 1525, 2657, 27162, 27859, 10371, 18853, 10650, 24366, 19840, 19850, 20967, 10795, 2582, 26925]

    knowledge_pattern = '''Can we try dialogue generation? I will give you turns and you can generate the next turn, but only one. 
    You can also consider the knowledge of %s for your reference in the dialogue.\n\n'''

    test_examples = []
    for example in data:
        example = json.loads(example)
        if example['utterance_id'] in test_ids:
            knowledge = []
            for triple in example['knowledge_base']:
                knowledge.append('''"'''+' '.join(triple)+'''"''')
            knowledge = ', '.join(knowledge)

            history = []
            for turn in example['history'][-3:]:
                history.append(turn[0].capitalize() + ': ' + turn[1])
            history = '\n'.join(history)

            example['input'] = knowledge_pattern%knowledge + history
            example['output'] = example['response'][0].capitalize() + ': ' + example['response'][1]

            test_examples.append(example)

    return test_examples, test_ids

if __name__ == "__main__":
    print("choose the testset you want")
    # test_examples, test_ids = truthful_qa()
    # test_examples, test_ids = covid_factchecking('scientific')
    # test_examples, test_ids = covid_factchecking('social')
