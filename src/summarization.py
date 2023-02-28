import jsonlines


def read_json(file_name):
    data = []
    with jsonlines.open(file_name) as reader:
        for item in reader:
            data.append(item)
    return data


def CNNDM():
    """
    The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering.
    The CNN / Daily Mail dataset version 1.0.0 is released under the Apache-2.0 License.
    
    test_examples:
    "src": source conversion
    "tgt": summary
    "prompt": chatgpt prompt
    """
    data = read_json('data/cnndm_test_50.jsonl')
    test_examples = []
    test_ids = []
    
    for i in range(len(data)):
        prompt = "Article: {}\n Summarize the above drticle.".format(data[i]['src'])
        test_examples.append({"src":data[i]['src'], "tgt":data[i]['tgt'], "prompt": prompt})
        test_ids.append(i)
    return test_examples, test_ids


def SAMSum():
    """
    The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. Linguists were asked to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger convesations. The style and register are diversified - conversations could be informal, semi-formal or formal, they may contain slang words, emoticons and typos. Then, the conversations were annotated with summaries. It was assumed that summaries should be a concise brief of what people talked about in the conversation in third person. The SAMSum dataset was prepared by Samsung R&D Institute Poland and is distributed for research purposes (non-commercial licence: CC BY-NC-ND 4.0).
    
    test_examples:
    "src": source conversion
    "tgt": summary
    "prompt": chatgpt prompt
    """
    data = read_json('data/samsum_test_50.jsonl')
    test_examples = []
    test_ids = []
    
    for i in range(len(data)):
        prompt = "Dialogue: {}\n Summarize the above dialogue.".format(data[i]['src'])
        test_examples.append({"src":data[i]['src'], "tgt":data[i]['tgt'], "prompt": prompt})
        test_ids.append(i)
    return test_examples, test_ids
    
if __name__ == "__main__":
    print("choose the testset you want")
    # test_examples, test_ids = SAMSum()
    test_examples, test_ids = CNNDM()
    print(test_examples[0].keys())
    print(test_ids)
    # test_examples, test_ids = covid_factchecking('social')