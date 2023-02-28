from datasets import load_dataset
import pandas as pd
from datasets import DownloadManager

def flores_mt():
    configs = [
        'eng_Latn-zho_Hans',
        'eng_Latn-kor_Hang',
        'eng_Latn-jpn_Jpan',
        'eng_Latn-fra_Latn',
        'eng_Latn-spa_Latn',
        'eng_Latn-ind_Latn',
        'eng_Latn-vie_Latn',
        'eng_Latn-arb_Arab',
        'eng_Latn-jav_Latn',
        'eng_Latn-sun_Latn',
        'eng_Latn-bug_Latn'
    ]

    dsets = {}
    for config in configs:
        dsets[config] = load_dataset("facebook/flores", name=config)

    lang_map = {
        'zho_Hans': 'Chinese',
        'kor_Hang': 'Korean',
        'jpn_Jpan': 'Japanese',
        'fra_Latn': 'French',
        'spa_Latn': 'Spanish',
        'ind_Latn': 'Indonesian',
        'vie_Latn': 'Vietnamese',
        'arb_Arab': 'Arabic',
        'jav_Latn': 'Javanese',
        'sun_Latn': 'Sundanese',
        'bug_Latn': 'Buginese',
    }

    dfs = []
    for config in configs:
        lang1, lang2 = config.split('-')
        df = pd.DataFrame(dsets[config].shuffle(seed=42)['devtest'][:30])
        df = df.rename(columns={f"sentence_{lang1}": "eng", f"sentence_{lang2}": "oth"})
        df['lang_pair'] = config
        df['to_eng'] = df.apply(lambda row: f"What is the English translation of the following sentence?\n\n{row['oth']}", axis='columns')
        df['to_oth'] = df.apply(lambda row: f"What is the {lang_map[lang2]} translation of the following sentence?\n\n{row['eng']}", axis='columns')
        df['to_eng_ref'] = df.apply(lambda row: f"Could you perform a post-editing to ensure the meaning is equivalent to \"{row['oth']}\"", axis='columns')
        df['to_oth_ref'] = df.apply(lambda row: f"Could you perform a post-editing to ensure the meaning is equivalent to \"{row['eng']}\"", axis='columns')
        dfs.append(df[['id', 'lang_pair', 'domain', 'topic', 'eng', 'oth', 'to_eng', 'to_eng_ref', 'to_oth', 'to_oth_ref']])        
    df = pd.concat(dfs)
    
    return df.to_dict(orient='records'), df['id'].tolist()

def wmt22_mt_ape():
    dm = DownloadManager()
    dl_path = dm.download_and_extract('https://drive.google.com/uc?export=download&id=10eKt1kWpx9aHGvky0ROUH_Lcb9UFDJhr')
    base_path = f'{dl_path}/TrainDev'
    df = pd.DataFrame({
        'src': open(f'{base_path}/dev.src').readlines(),
        'mt': open(f'{base_path}/dev.mt').readlines(),
        'pe': open(f'{base_path}/dev.pe').readlines()
    })
    df['id'] = [i for i in range(len(df))]
    df = df[['id', 'src', 'mt', 'pe']]
    
    return df.to_dict(orient='records'), df['id'].tolist()

if __name__ == "__main__":
    print("choose the testset you want")    
    test_examples, test_ids = flores_mt()
    # test_examples, test_ids = wmt22_mt_ape()
