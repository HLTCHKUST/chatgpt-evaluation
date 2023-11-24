import json
import os
import random

def retrieve_slots_and_values(services, schema):
    text = ""
    text += "Intents: Request, Inform, general-thank, general-bye\n"
    for service in services:
        for service_data in schema:
            if service_data["service_name"] == service:
                L = len(service_data["service_name"])
                for slot in service_data["slots"]:
                    possible_values = slot.get("possible_values", "[?]")
                    text += f"""Domain: {service}, Slots: {slot["name"][L+1:]}, Possible values: {possible_values}\n"""    
    return text

def multiwoz22(exp_type): #["DST", "response_gen"]
    """
    MultiWOZ 2.2 - Task-Oriented Dialogue 
    Dialogue State Tracking and Response Generation using Oracle Actions
    {
        "data-source": "https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2",
        "evaluation-method": ["human-evaluation", "BLEU"],
        "answer-type": "",
        "evaluation-details": "The prompt format used is given in the Appendix section of the paper.
                              We average the scores across all turns and all samples.
                              For DST:
                              The generated belief state is manually evaluated and compared to gold belief states to compute JGA (ratio of turns where slot-value are correct)
                              For RG:
                              The generated response is evaluated with BLEU and Inform rate, where we take the ratio of correctly used action slots provided in the prompt for the generation.
                              If the generation does not answer the USER's question, this score is 0 for the given turn."  
        "notes": "Set the exp_type to DST for dialogue state tracking, and response_gen for response generation"
    }
    
    """

    DST_EXAMPLES = ['MUL0484.json',
                    'MUL2155.json',
                    'PMUL0815.json',
                    'SNG0423.json',
                    'PMUL0079.json',
                    'SNG0840.json']

    RG_EXAMPLES = ['MUL0484.json',
                   'MUL2155.json',
                   'PMUL0815.json',
                   'SNG0423.json',
                   'PMUL0079.json',
                   'SNG0840.json']

    multiwoz_path = os.path.join(os.getcwd(), "data/MultiWOZ_2.2")

    dialog_acts_path = os.path.join(multiwoz_path, "dialog_acts.json")
    with open(dialog_acts_path, "r") as f:
        dialog_acts = json.load(f)

    schema_path = os.path.join(multiwoz_path, "schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)

    dialogue_path = os.path.join(multiwoz_path, "test/dialogues_001.json")
    with open(dialogue_path, "r") as f:
        dialogue = json.load(f)

    test_examples, test_ids = [], []

    if exp_type == "DST":
        samples = [dialog for dialog in dialogue if dialog["dialogue_id"] in DST_EXAMPLES]
        instruction = "Give the dialogue state of the last utterance in the following dialogue in the form of 'STATE: {Domain-Intent: [Slot, Possible value], ...} (for example: STATE: {Hotel-Inform: ['area', 'centre']}) by using the following pre-defined slots and possible values:\n\n"

        for sample in samples:
            prompt = instruction
            services = sample["services"]
            prompt += retrieve_slots_and_values(services, schema) + "\n"
            dialogue_id = sample["dialogue_id"]
            turns = sample["turns"]

            for turn in turns:
                prompt += f"""{turn["speaker"]}: {turn["utterance"]}\n"""
                if turn["speaker"] == "USER":
                    test_examples.append(prompt)
                    test_ids.append({"dialogue_id": dialogue_id, 
                                    "turn_id": turn["turn_id"]})
                    prompt += f"""STATE : {dialog_acts[dialogue_id][turn["turn_id"]]["dialog_act"]}\n"""
                

        return test_examples, test_ids


    elif exp_type == "response_gen":
        samples = [dialog for dialog in dialogue if dialog["dialogue_id"] in RG_EXAMPLES]
        instruction = "Continue the dialogue as a task-oriented dialogue system called SYSTEM. The answer of SYSTEM should follow the ACTION provided next while answering the USER's last utterance:\n\n"

        for sample in samples:
            conv = ""
            dialogue_id = sample["dialogue_id"]
            turns = sample["turns"]
            
            for turn in turns:
                next_turn_id = str(int(turn["turn_id"])+1)
                conv += f"""{turn["speaker"]}: {turn["utterance"]}\n"""
                if turn["speaker"] == "USER":
                    
                    action = f"""ACTION: {dialog_acts[dialogue_id][next_turn_id]["dialog_act"]}\n\n"""
                    prompt = instruction + action + conv
                    test_examples.append(prompt)
                    test_ids.append({"dialogue_id": dialogue_id, 
                                    "turn_id": next_turn_id})
        
        return test_examples, test_ids


    else:
        raise ValueError("exp_type should be either 'DST' or 'response_gen'")
    


def opendialkg(): 
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
    # Task oriented dialogue
    # test_examples, test_ids = multiwoz22('') #["DST", "response_gen"]

    #Open domain KG dialogue
    # test_examples, test_ids = opendialkg()
