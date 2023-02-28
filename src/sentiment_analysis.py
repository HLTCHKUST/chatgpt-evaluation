import pandas as pd
from nusacrowd import NusantaraConfigHelper

TEXT_CLASSIFICATION_TASKS = [
    'nusax_senti_eng_nusantara_text',
    'nusax_senti_ind_nusantara_text',
    'nusax_senti_jav_nusantara_text',
    'nusax_senti_bug_nusantara_text'
]

lang_map = {
    'nusax_senti_eng_nusantara_text': 'English',
    'nusax_senti_ind_nusantara_text': 'Indonesian',
    'nusax_senti_jav_nusantara_text': 'Javanese',
    'nusax_senti_bug_nusantara_text': 'Buginese'
}

def nusax_sentiment():
    conhelps = NusantaraConfigHelper()
    nlu_datasets = {
        lang_map[helper.config.name]: helper.load_dataset() for helper in conhelps.filtered(
            lambda x: x.config.name in TEXT_CLASSIFICATION_TASKS
        )
    }
    
    dfs = []
    for key, dset in nlu_datasets.items():
        labels = nlu_datasets[key]['test'].features['label']
        df = nlu_datasets[key]['test'].to_pandas()[:50]
        df['label'] = df['label'].apply(lambda x: labels.int2str(x))
        df['lang'] = key
        df['id'] = df['id'].apply(lambda x: f'{key.lower()[:3]}-{x}')
        df = df[['id', 'lang', 'text', 'label']]
        dfs.append(df)
    df = pd.concat(dfs)
    
    return df.to_dict(orient='records'), df['id'].tolist()

if __name__ == "__main__":
    print("choose the testset you want")    
    test_examples, test_ids = nusax_sentiment()
