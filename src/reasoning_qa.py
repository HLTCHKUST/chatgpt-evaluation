import os
import re                
import json
import gdown
import pandas as pd
from numpy import loadtxt
import numpy as np
from datasets import load_dataset

alphaonly = re.compile('[^a-zA-Z ?]')


def hotpot_qa(num_dataset=30, output_gold=True):
    """
        HotpotQA
        HotpotQA is a question answering dataset featuring natural, **multi-hop questions**, with strong supervision for supporting facts to enable more explainable question answering systems.

         {
            "data_source": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
            "license": "CC BY-SA 4.0"
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "multihop-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "Since the original dev set is very large and 1-line file and the license allows us to modify and redistribute the dataset (under the same license), we first take the first 250k characters of the file, then round it back to JSON format with 36 samples. The new file is ./data/hotpotqa_dev_wiki_v1_36_samples.json"
        }
    """
    data = json.loads(open('data/hotpotqa_dev_wiki_v1_36_samples.json').read())
    prompt = "Given the following information:\n{}\n\nQuestion: {}\n\nPlease find the supporting facts from the provided context, and use them to answer the question. Write all your reasoning steps."
    test_examples, test_golds, test_ids = [], [], list(range(num_dataset))

    for i in range(num_dataset):
        row = data[i]
        test_golds.append(row['answer'])
        # print(row['supporting_facts'])
        question = row['question']
        context = '\n'.join([f'{p}: {"".join(d)}' for p, d in row['context']])
        test_examples.append(prompt.format(context, question))

    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def spart_qa(num_dataset=30, output_gold=True, exp_type='1-reasoning-type'):
    """
        SpaRTQA
        SpartQA is a textual question answering benchmark for spatial reasoning on natural language text which contains more realistic spatial phenomena not covered by prior datasets and that is challenging for state-of-the-art language models (LM). SPARTQA is built on NLVR’s images containing more objects with richer spatial structures. SPARTQA’s stories are more natural, have more sentences, and richer in spatial relations in each sentence, and the questions require deeper reasoning and have four types: find relation (FR), find blocks (FB), choose object (CO), and yes/no (YN), which allows for more fine-grained analysis of models’ capabilities
        
        {
            "data_source": ["https://github.com/HLR/SpartQA_generation",
                "https://drive.google.com/file/d/12s2olGDV0ruywPtLhGL5M-1e4CbQrA8k/view"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "spatial-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "Because each context has several corresponding questions for each type of reasoning, we select the first sample for each type and add it to our test set. That means, each context will usually have 4 questions in our test set. Also, we take the train split because it provides image to double check the gold answer."
        }
    """
    data = open('data/human_test.json', 'r').read()
    data = data.replace('false', 'False').replace('true', 'True')
    data = eval(data)['data']
    test_examples, test_golds, test_ids = [], [], []

    if exp_type == '1-reasoning-type':
        num_context = (num_dataset-1)//4+1
        for i in range(num_context):
            story = data[i]['story']
            questions = data[i]['questions']
            question_type_taken = []
            for j, q in enumerate(questions):
                if not len(q['reasoning_type']) == 1:
                    continue

                q_type = q['q_type']
                if (q_type not in ['FR', 'YN', 'CO', 'FB']) or (q_type in question_type_taken):
                    continue
                else:
                    question_type_taken.append(q_type)

                if 'candidate_answers' in q.keys():
                    candidate_str = ', '.join([f'{k}. {t}' for k, t in enumerate(q['candidate_answers'])])
                    candidate_str = candidate_str.replace('DK', "don't know")
                else:
                    candidate_str = ''

                test_examples.append(f'Given the description: {story}. {q["question"]} {candidate_str}')
                test_golds.append(q['answer'])
                test_ids.append(f'{i}|{j}')
                if len(question_type_taken) >= 4:
                    break

    elif exp_type == '2-reasoning-type':
        count_by_q_type = {x: 0 for x in ['FR', 'YN', 'CO', 'FB']}
        num_samples_each_q_type = (num_dataset-1)//4+1
        num_samples_taken = 0

        for i in range(len(data)):
            instance = data[i]
            story = instance['story']
            questions = instance['questions']
            question_type_taken = []

            for j, q in enumerate(questions):
                if not len(q['reasoning_type']) == 2:
                    continue

                q_type = q['q_type']
                if (q_type not in ['FR', 'YN', 'CO', 'FB']) or (q_type in question_type_taken):
                    continue
                elif count_by_q_type[q_type] >= num_samples_each_q_type:
                    continue
                else:
                    test_ids.append(f'{i}|{j}')
                    # print(q['reasoning_type'])
                    question_type_taken.append(q_type)
                    count_by_q_type[q_type] += 1
                    num_samples_taken += 1

                if 'candidate_answers' in q.keys():
                    candidate_str = ', '.join([f'{k}. {t}' for k, t in enumerate(q['candidate_answers'])])
                    candidate_str = candidate_str.replace('DK', "don't know")
                else:
                    candidate_str = ''
                test_examples.append(f'Given the description: {story}. {q["question"]} {candidate_str}')
                test_golds.append(q['answer'])

                if len(question_type_taken) >= 4:
                    break

            if num_samples_taken >= num_samples_each_q_type*4:
                break

    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def math():
    """
        Math dataset
        This dataset contains mathematical question and answer pairs, from a range of question types at roughly school-level difficulty. This is designed to test the mathematical learning and algebraic reasoning skills of learning models.
    
        {
            "data_source": ["https://console.cloud.google.com/storage/browser/_details/mathematics-dataset/mathematics_dataset-v1.0.tar.gz",
                "https://huggingface.co/datasets/math_dataset/viewer/algebra__linear_1d/test"],
            "license": "Apache License 2.0",
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "mathematical-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "Since the original dataset is 2.2GB, and the license Apache License 2.0 allows to modify the dataset and redistribute it, we build the set to test ChatGPT by copying from dataviewer of https://huggingface.co/datasets/math_dataset. In specific, we pick the first 5 samples from 6 categories: 'algebra__linear_1d', 'arithmetic__add_or_sub', 'calculus__differentiate', 'comparison__closest', 'measurement__conversion', 'numbers__base_conversion'. The set can be found in ./data/math_deepmind_30_samples.csv".
        }
    """
    data = pd.read_csv('data/math_deepmind_30_samples.csv')
    test_ids = list(range(5))*6
    test_examples, test_golds = data['Question'].tolist(), data['Gold_answer'].tolist()
    
    return test_examples, test_ids, test_golds


def timedial(num_dataset=30):
    """
        TimeDial
        TimeDial presents a crowdsourced English challenge set, for temporal commonsense reasoning, formulated as a multiple choice cloze task with around 1.5k carefully curated dialogs. The dataset is derived from the DailyDialog (Li et al., 2017), which is a multi-turn dialog corpus. We follow the format of the task in the BIG-Bench benchmark, which is multiple-choice (single correct answer). Note that the correct answer should be 0 or 1, say if answer of ChatGPT indicates the answer as 0 or 1, we mark the answer as True.

        {
            "data_source": "https://github.com/google-research-datasets/TimeDial/blob/main/test.json",
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "temporal-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
        }
    """
    timedial_data = json.load(open('data/timedial_test.json', 'r'))
    test_examples = []
    test_ids = list(range(num_dataset))
    test_golds = ['0 or 1 (if option 1 != none)']*num_dataset

    for ex in timedial_data[:num_dataset]:
        conversation = ex['conversation']
        choices = (ex['correct1'], ex['correct2'], ex['incorrect1'], ex['incorrect2']) # ex['correct2'] if ex['correct2'] != 'none' else ex['correct1']

        conversation = 'Given the conversation:\n' + '\n'.join(conversation) + '\n'
        choices = 'Candidate choices to fill in the <mask>: ' + ' '.join([f'{j}. {t},' for j, t in enumerate(choices)]) + '\n'
        caution = 'Note that there may not be enough information to certainly fill in the <mask>, but from commonsense reasoning, you can surely narrow down what are the most probable choices to fill in the <mask>. Please select the most propable choice from candidates and explain your choice.'
        final = conversation+choices+caution
        test_examples.append(final.replace('"', "'"))

    return test_examples, test_ids, test_golds


def pep_3k(num_dataset=30, output_gold=True):
    """
        Pep-3k
        Pep-3k is a dataset of physical semantic plausibility judgments of single events. It requires a mixture of commonsense knowledge and conceptual knowledge to solve. Each event consists of a subject, a verb, and an object, i.e it has the simple s-v-o format. For example, the event can be man swallow paintball, with the label 0 (implausible). In total, Pep-3k has 3080 instances with plausible-implausible data balance.

        {
            "data_source": "https://github.com/suwangcompling/Modeling-Semantic-Plausibility-NAACL18/tree/master/data",
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "commonse-reasoning",
            "answer-type": "",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain score 1 (for True) or 0 (for False), based on gold labels from original data. The average score serves as overall accuracy. \",
            "note": "download two files pos-all.txt and neg-all.txt to ./data/pep-3k/"
        }
    """
    pos_data = open('data/pep-3k/pos-all.txt', 'r').read().splitlines()
    neg_data = open('data/pep-3k/neg-all.txt', 'r').read().splitlines()
    num_data_each = num_dataset // 2
    test_examples = pos_data[:num_data_each] + neg_data[:num_data_each]
    test_ids = list(range(num_data_each))*2
    test_golds = [1]*num_data_each + [0]*num_data_each

    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def step_game(exp_type): # ['hard', 'basic', 'clock-position', 'basic-cardinal', 'diagonal']
    """
        StepGame - Spatial Reasoing & Question Answeing

        {
            "data_source": ["https://github.com/ZhengxiangShi/StepGame/blob/main/Dataset/CompleteVersion/clean/qa1_test.json",
                "https://github.com/ZhengxiangShi/StepGame/blob/main/Dataset/CompleteVersion/clean/qa9_valid.json"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "spatial-reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    if exp_type == 'hard':
        path = 'data/qa9_valid.json'
    elif exp_type in ['basic', 'clock-position', 'basic-cardinal', 'diagonal']:
        path = 'data/qa1_test.json'
    
    f = open(path)
    stepgame_data = json.load(f)

    MC_text = "Choose from: left, right, above, below, lower-left, lower-right, upper-left, upper-right."
    
    if exp_type == 'basic':
        test_ids = [i for i in range(30)]
        test_ids.remove(4)  # gold label is wrong
    elif exp_type == 'diagonal': 
        test_ids = [30, 31, 39, 40, 41, 47, 48, 50, 52, 53, 60, 61, 63, 65, 67, 69, 71, 72, 84, 87]
    elif exp_type == 'clock-position':
        test_ids = [62, 54, 119, 131, 143, 144, 189, 316, 426, 484, 697, 820, 960, 1045, 1163, 1607, 1618, 1620, 1736, 1778]
    elif exp_type == 'basic-cardinal':
        test_ids = [1, 2, 3, 9, 10, 16, 17, 18, 19, 22, 25, 27, 33, 34, 35, 42, 49, 51, 59, 66]
    elif exp_type == 'hard':
        test_ids = range(30)
    
    test_examples, test_golds = [], []
    for i in test_ids:
        # ex = [stepgame_data[str(i)]]
        ex = stepgame_data[str(i)]
        if exp_type == 'hard':
            ex['input'] = "Given the description: {}. {}".format(
                ' '.join(ex['story']), ex['question'].replace('relation', 'spatial relation, (e.g left, right, above lower-left, ..)'))
       
        else:
            ex['input'] = f"{ex['story'][0]} {ex['question']} {MC_text}"
        test_examples.append(ex['input'])
        test_golds.append(ex['label'])

    return test_examples, test_ids, test_golds


def letter_string_analogies(): 
    """
        Letter_string_analogies - Analogical Reasoing
        {
            "data_source": ["https://github.com/taylorwwebb/emergent_analogies_LLM/blob/main/letter_string/letter_string_analogies.npz"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "analogical-reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the prompt and obtain the answer from ChatGPT and compare with gold."
        }
    """
    # Load all problems
    file ='data/letter_string_analogies.npz'
    all_prob = np.load(file, allow_pickle=True)['all_prob']
    all_completions = np.load(file, allow_pickle=True)['all_prob_completion']

    test_examples,test_ids,test_golds = [],[],[]

    # problem types 4 and 5
    for p in [4, 5]:
        probs = all_prob[p][:15]
        golds = all_completions[p][:15]
        for prob, ans in zip(probs, golds):
            prompt = "Let's try to complete the pattern:\n\n" + prob
            test_examples.append(prompt)
            test_golds.append(ans)
    
    return test_examples, test_ids, test_golds


def babi(exp_type = 15, prompt_engineering=False, num_dataset=30, batching=True, output_gold=True, save_csv=False):
    """
        bAbI
        This basic induction bAbI tasks is taken from the (20) QA bAbI tasks that a set of proxy tasks that evaluate 
        reading comprehension via question answering. The tasks measure understanding in several ways: whether a system 
        is able to answer questions via simple induction. 
        The tasks are designed to be prerequisites for any system that aims to be capable of conversing with a human.
    
        {
            "data_source": ["http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "deductive-inductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    
    def make_samples(filename, exp_type, prompt_engineering=False, batching=True):
        df = pd.read_csv(filename, header=None)
    
        dataset = {}
        index = 0
        golds = {}
        for i, row in df.itertuples():
            if index not in dataset.keys():
                dataset[index] = ''
                golds[index] = ''

            if 'what' in row.lower():
                
                gold = row[row.find('\t'):]
                row = row[:row.find('\t')]
                
                if batching:
                    dataset[index] += row + '\n'
                    golds[index] += gold + '\n'
                    
                    if i+1 < df.shape[0]:
                        if prompt_engineering and \
                            'what' not in df.iloc[[i+1]].values[0][0].lower():
                            if exp_type == 15:
                                marker = 'deductive' 
                            elif exp_type == 16:
                                marker = 'inductive' 
                              
                            allprev = ''.join([i for i in dataset[index].replace('\n', '') if not i.isdigit()]).strip()
                            context = allprev[:allprev.lower().find('what')]
                            question = allprev[allprev.lower().find('what'):]

                            dataset[index] = 'Given facts: ' + context +\
                                             '\n\nThe most recent fact is the correct fact.\n\nBased on the given facts above, do a reasonable inference on this question using '+marker+' reasoning: ' + question
                        
                    if i+1 < df.shape[0]:
                        if 'what' not in df.iloc[[i+1]].values[0][0].lower():
                            index += 1
                else:
                    if i+1 < df.shape[0]:
                        golds[index] = gold
                        if 'what' not in df.iloc[[i-1]].values[0][0].lower():
                            context = dataset[index]
                            
                        if not prompt_engineering:
                            dataset[index] = context + row
                        else:
                            if exp_type == 15:
                                marker = 'deductive' 
                            elif exp_type == 16:
                                marker = 'inductive' 
                                
                            dataset[index] = 'Given facts: ' +\
                                             ''.join([i for i in context.replace('\n', '') if not i.isdigit()]).strip() +\
                                             '\n\nThe most recent fact is the correct fact.\n\nBased on the given facts above, do a reasonable inference on this question using '+marker+' reasoning: ' +\
                                             alphaonly.sub('', row).strip()
                            
                        index += 1
                    
            else:
                dataset[index] += row + '\n'
                
        ids = dataset.keys()
        dataset_in_list = [dataset[idx] for idx in ids]
        gold_in_list = [golds[idx] for idx in ids]
        
        return dataset_in_list, gold_in_list
    
    if exp_type == 15:
        filename = 'tasks_1-20_v1-2/en/qa15_basic-deduction_test.txt'
    elif exp_type == 16:
        filename = 'tasks_1-20_v1-2/en/qa16_basic-induction_test.txt'
    else:
        raise NotImplementedError('task_id: {} is not yet implemented'.\
                                            format(exp_type))
        
    filename = 'data/' + filename
    if not os.path.isfile(filename):
        os.system('wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
        os.system('tar -xf tasks_1-20_v1-2.tar.gz')
        os.system('rm -rf tasks_1-20_v1-2.tar.gz')
        os.system('mv tasks_1-20_v1-2 data/')
    
    prompts, golds = make_samples(filename, exp_type, prompt_engineering, batching)
    
    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts[:num_dataset], 'Golds':golds[:num_dataset]})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('babi_task_'+str(exp_type)+'.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def entailmentbank(num_dataset=30, output_gold=True, save_csv=False):
    """
        EntailmentBank
        ENTAILMENTBANK, the first dataset of multistep entailment trees for QA, to support entailment-based explanation. 
        ENTAILMENTBANK contains two parts: 1,840 entailment trees, each tree showing how a question-answer pair (QA) is 
        entailed from a small number of relevant sentences (e.g., Figure 1); and a general corpus C, containing those and 
        other sentences of domain-specific and general knowledge relevant to the QA domain.
    
        {
            "data_source": ["https://drive.google.com/drive/folders/1SmnCw-Dfad3a68AmZZtG4jBz6UUKFkZk", 
                            "https://allenai.org/data/entailmentbank",
                            "https://github.com/allenai/entailment_bank/"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "deductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"        }
    """
    if not os.path.isfile("data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl"):
        url = "https://drive.google.com/drive/folders/1SmnCw-Dfad3a68AmZZtG4jBz6UUKFkZk"
        gdown.download_folder(url, quiet=True, use_cookies=False)

        os.system('unzip v3_May6_2022/entailment_trees_emnlp2021_data_v3.zip')
        os.system('rm -rf v3_May6_2022/')
        os.system('mv entailment_trees_emnlp2021_data_v3 data/')
    
    entailmentbank = load_dataset("json", data_files="data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl")
    
    prompts = []
    golds = []
    for dataset_id in range(num_dataset):
        data = entailmentbank['train'][dataset_id]
        context = ". ".join([sent[3:].strip() for sent in data['context'].split('sent')[1:]])
        question = data['question']
        gold = data['answer']

        prompt = context + '. ' + question

        prompts.append(prompt)
        golds.append(gold)
    
    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('entailmentbank.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def alpha_nli(num_dataset=30, output_gold=True, save_csv=False):
    """
        αNLI
        αbductive Natural Language Inference (αNLI) is a new commonsense benchmark dataset designed to test 
        an AI system’s capability to apply abductive reasoning and common sense to form possible explanations for 
        a given set of observations. Formulated as a binary-classification task, the goal is to pick the most 
        plausible explanatory hypothesis given two observations from narrative contexts.
    
        {
            "data_source": ["https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip",
                            "http://abductivecommonsense.xyz/"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "abductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    
    if not os.path.isfile('anli/test.jsonl'):
        os.system('wget https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/anli.zip')
        os.system('/usr/bin/unzip anli.zip')
        os.system('rm -rf anli.zip')
        os.system('mv anli data/')
    
    anli_dataset = load_dataset('json', data_files='data/anli/test.jsonl')
    lines = loadtxt('data/anli/test-labels.lst', comments="#", delimiter=",", unpack=False)
    labels = [int(line) for line in lines]
    
    prompts = []
    golds = []
    for dataset_id in range(num_dataset):
        data = anli_dataset['train'][dataset_id]

        prompt = 'Given: ' + data['obs1'] + ' Then: '+ data['obs2'] + \
                ' Select the most plausible explanation (hypothesis): A. ' + data['hyp1'] + \
                ' B. ' + data['hyp2']

        prompts.append(prompt)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':labels[:num_dataset]})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('anli.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def clutrr(num_dataset=30, output_gold=True, save_csv=False):
    """
        CLUTRR
        CLUTRR (Compositional Language Understanding and Text-based Relational Reasoning), 
        a diagnostic benchmark suite, is first introduced in (https://arxiv.org/abs/1908.06177) 
        to test the systematic generalization and inductive reasoning capabilities of NLU systems. 
        The CLUTRR benchmark allows us to test a model’s ability for systematic generalization by 
        testing on stories that contain unseen combinations of logical rules, and test for the 
        various forms of model robustness by adding different kinds of superfluous noise facts to the stories.
    
        {
            "data_source": ["https://huggingface.co/datasets/CLUTRR/v1"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "inductive reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    clutrr_dataset = load_dataset("CLUTRR/v1", "gen_train23_test2to10")
    
    prompts = []
    golds = []
    for dataset_id in range(num_dataset):
        data = clutrr_dataset['test'][dataset_id]
        first_p = eval(data['query'])[1]
        second_p = eval(data['query'])[0]

        prompt = data['clean_story'] + '. Who is ' + first_p + ' to ' + second_p + '?'
        gold = data['target_text']

        prompts.append(prompt)
        golds.append(gold)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('clutrr.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


def commonsenseqa(num_dataset=30, output_gold=True, save_csv=False):
    """
        CommonsenseQA
        CommonsenseQA is a new multiple-choice question answering dataset that requires different types of 
        commonsense knowledge to predict the correct answers . It contains 12,102 questions with one correct 
        answer and four distractor answers. The dataset is provided in two major training/validation/testing 
        set splits: "Random split" which is the main evaluation split, and "Question token split", see paper for details.
    
        {
            "data_source": ["https://huggingface.co/datasets/commonsense_qa"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "commonsense reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    commonsenseqa_dataset = load_dataset("commonsense_qa")
    
    prompts = []
    golds = []
    for i in range(num_dataset):

        gold = commonsenseqa_dataset['validation'][i]['answerKey']
        prompt = commonsenseqa_dataset['validation'][i]['question']

        choice_str = ''
        for choice_id, choice in enumerate(commonsenseqa_dataset['validation'][i]['choices']['label']):
            choice_str += choice + '. ' + commonsenseqa_dataset['validation'][i]['choices']['text'][choice_id] + ', '
        prompt += ' ' + choice_str[:-2]
        prompt

        prompts.append(prompt)
        golds.append(gold)
    
    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('commonsense_qa.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids
    

def piqa(num_dataset=30, output_gold=True, save_csv=False):
    """
        PIQA
        To apply eyeshadow without a brush, should I use a cotton swab or a toothpick? Questions requiring this 
        kind of physical commonsense pose a challenge to state-of-the-art natural language understanding systems. 
        The PIQA dataset introduces the task of physical commonsense reasoning and a corresponding benchmark 
        dataset Physical Interaction: Question Answering or PIQA. Physical commonsense knowledge is a major challenge 
        on the road to true AI-completeness, including robots that interact with the world and understand natural language. 
        PIQA focuses on everyday situations with a preference for atypical solutions. The dataset is inspired by 
        instructables.com, which provides users with instructions on how to build, craft, bake, or manipulate objects
        using everyday materials.
    
        {
            "data_source": ["https://huggingface.co/datasets/piqa"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "commonsense reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    piqa_dataset = load_dataset("piqa")
    
    prompts = []
    golds = []
    for i in range(num_dataset):

        gold = piqa_dataset['validation'][i]['label']

        if piqa_dataset['validation'][i]['goal'][-1]=='.':
            goal = piqa_dataset['validation'][i]['goal'][:-1]
        else:
            goal = piqa_dataset['validation'][i]['goal']

        prompt = 'Pick from option 0 or 1 to achieve this goal:' + goal + ' 0: "' +\
                 piqa_dataset['validation'][i]['sol1'] + '" 1: "' +\
                 piqa_dataset['validation'][i]['sol2'] + '"'

        prompts.append(prompt)
        golds.append(gold)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds})
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('piqa.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids
    

def ecare(num_dataset=30, output_gold=True, save_csv=False):
    """
        E-Care
        Understanding causality has vital importance for various Natural Language Processing (NLP) applications. 
        Beyond the labeled instances, conceptual explanations of the causality can provide a deep understanding 
        of the causal fact to facilitate the causal reasoning process. We present a human-annotated explainable 
        CAusal REasoning dataset (e-CARE), which contains over 20K causal reasoning questions, together with 
        natural language formed explanations of the causal questions.
    
        {
            "data_source": ["https://huggingface.co/datasets/12ml/e-CARE"],
            "evaluation-method": "human-evaluation",
            "evaluation-aspect": "causal reasoning",
            "answer-type":"",
            "evaluation-details": "Feed the 'input' of the examples to the model and take generated answer for evaluation.\
                                The generated answer is evaluated by a human to obtain accuracy, based on gold labels from original data.\"
        }
    """
    ecare_dataset = load_dataset("12ml/e-CARE")
    
    prompts = []
    golds = []
    gold_explanations = []

    for i in range(num_dataset):

        gold = ecare_dataset['validation'][i]['label']
        gold_explanation = ecare_dataset['validation'][i]['conceptual_explanation']

        if ecare_dataset['validation'][i]['question'] == 'cause':
            prompt = 'Choices: Choice1: ' + \
                        ecare_dataset['validation'][i]['choice1'] + ' Choice2: ' + \
                        ecare_dataset['validation'][i]['choice2'] + ' Which one of the choices are causing the sentence: ' + \
                        ecare_dataset['validation'][i]['premise']
        elif ecare_dataset['validation'][i]['question'] == 'effect':
            prompt = 'If ' + ecare_dataset['validation'][i]['premise'] + ' Which one of the choices are caused by that? Choices: Choice1: ' + \
                        ecare_dataset['validation'][i]['choice1'] + ' Choice2: ' + \
                        ecare_dataset['validation'][i]['choice2']
        prompts.append(prompt)
        golds.append(gold)
        gold_explanations.append(gold_explanation)

    df = pd.DataFrame({'Ids':list(range(num_dataset)), 'Prompts':prompts, 'Golds':golds, 'Gold_explanations':gold_explanations})
    
    test_examples = df.Prompts.tolist()
    test_ids = df.Ids.tolist()
    test_golds = df.Golds.tolist()
    if save_csv:
        df.to_csv('ecare.csv', index=False)
    
    if output_gold:
        return test_examples, test_ids, test_golds
    else:
        return test_examples, test_ids


if __name__ == "__main__":
    print("choose the testset you want")

    # Spatial Reasoning
    # test_examples, test_ids = step_game('basic') # exp_type = ['hard', 'basic', 'clock-position', 'basic-cardinal', 'diagonal']
    # print(test_examples[0], test_ids[0])
