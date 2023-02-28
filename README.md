# A Multitask, Multilingual, Multimodal Evaluation Datasets for ChatGPT

<img align="right" src="imgs/HKUST.jpg" width="12%">This respository contains the code for extracting the test samples we used in our paper:
**A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity**. Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quyet V. Do, Yan Xu, Pascale Fung [[Arxiv]](https://arxiv.org/abs/2302.04023)

We evaluated ChatGPT on subset of **23** different publicly available datasets. We share the exact test samples we utilized in our paper for reproductibility. 

## How to Use
1. Check the dataset that you want to extract from main.py. There will be a source path and function name that leads you to find original data url.

	**e.g.**, Target dataset: 'NusaX' 
	 - path = 'src/sentiment_analysis.py' 
 	- fnc_name = nusax_sentiment()

2. Download the original data (if needed) and place it under 'data' folder of this repo.

3. Run the code directly from the source path (e.g., src/sentiment_analysis.py) or main.py.

## Citation
If you find this paper and code useful, please cite our paper.

```
@article{bang2023multitask,
  title={A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity},
  author={Bang, Yejin and Cahyawijaya, Samuel and Lee, Nayeon and Dai, Wenliang and Su, Dan and Wilie, Bryan and Lovenia, Holy and Ji, Ziwei and Yu, Tiezheng and Chung, Willy and Do, Quyet V.  and Xu, Yan and Fung, Pascale},
  journal={arXiv preprint arXiv:2302.04023},
  year={2023}
}
```
#### Contact
* Pascale Fung: pascale[at]ece[dot]ust[dot]hk
* Yejin Bang: yjbang[at]connect[dot]ust[dot]hk

## Acknowledgement
Our work utilized publicly available dataset and each function includes the exact source of data. For more details for each dataset, please check **Table 16** in our paper. 