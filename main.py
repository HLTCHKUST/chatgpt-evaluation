from src.hallucination import *
from src.machine_translation import *
from src.reasoning_qa import *
from src.sentiment_analysis import *
from src.dialogue import *
from src.summarization import *
import argparse

'''
Please refer to the dict datasetname_to_func to look for the dataset you want to get the test data
    then trace back to the original script in `src`, e.g truthful_qa -> ./src/hullucination.py
    and read all meta data including our notes, function signature (to pass the flag --subset properly)
    and download the dataset to ./data
    
    finally, run this file.
    Please double check the output format, because the output format of all datasets is not uniform
'''

def main():
    datasetname_to_func = {
    # src/hallucination.py
        'TruthfulQA': truthful_qa,
        'Covid-factchecking': covid_factchecking, #['scientific', 'social']

    # src/machine_translation.py
        'FLoRes-200': flores_mt,
        'WMT22': wmt22_mt_ape,

    # src/sentiment_analysis.py
        'NusaX': nusax_sentiment,

    # src/dialogue.py
        'MultiWOZ2.2': multiwoz22, #["DST", "response_gen"]
        'OpenDialKG': opendialkg,

    # src/summarization.py
        'CNNDM' : CNNDM,
        'SAMSum' : SAMSum,

    # src/reasoning_qa.py
        # Deductive
        'EntailmentBank': entailmentbank,
        'bAbI': babi, #[15, 16] #inductive
        # Inductive
        'CLUTRR': clutrr,
        # Abductive
        'AlphaNLI': alpha_nli,
        # Mathematical 
        'Math': math,
        # Temporal
        'TimeDial': timedial,
        # Spatial
        'StepGame': step_game, #['hard', 'basic', 'clock-position', 'basic-cardinal', 'diagonal']
        'SpaRTQA': spart_qa, #['1-reasoning-type', '2-reasoning-type']
        # Commonsense
        'Pep-3k': pep_3k,        
        'CommonsenseQA': commonsenseqa,
        'PIQA': piqa,
        # Causal
        'E-Care': ecare,
        # Multi-hop
        'HotpotQA': hotpot_qa,
        # Analogical
        'Letter_String_Analogy' : letter_string_analogies
   }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str, default='pep_3k', help='name of the dataset that you want to get test data',
        choices=list(datasetname_to_func.keys())
    )
    parser.add_argument(
        '--subset', 
        type=str, default='basic', help='for dataset having multiple categories/splits, please pass the name here'
     )
    args = parser.parse_args()

    data_loader = datasetname_to_func[args.dataset]
    if args.dataset in ['Covid-factchecking', 'bAbI', 'StepGame', 'SpaRTQA', 'MultiWOZ2.2']:
        output = data_loader(exp_type=args.subset)
    else:
        output = data_loader()

    if len(output) == 3:
        test_examples, test_ids, test_golds = output
        print(test_examples[0], test_ids[0], test_golds[0])
    elif len(output) == 2:
        test_examples, test_ids = output
        print(test_examples[0], test_ids[0])


if __name__ == '__main__':
    main()