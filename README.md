### Overview
The fine-tuning pipeline is largely based on the [LlamaFactory library](https://github.com/hiyouga/LlamaFactory). Activation space ablation and learned vector are explored using the [Transformer Lens Library](https://github.com/TransformerLensOrg/TransformerLens). 
 <br>. 
Two datasets are publicly available:  <br>
i. ClariQ: [SCAI Workshop data](https://github.com/aliannejadi/ClariQ/tree/master/data)  <br>
ii. AmbigQA: [AmbigNQ Disambiguation annotated](https://github.com/shmsw25/AmbigQA/blob/main/evidence.md)  <br>

Baselines: <br>
i. MuSIc: [System Initiative Prediction](https://github.com/ChuanMeng/SIP) <br>
ii. Bert Based: [ClariQ Baselines](https://github.com/aliannejadi/ClariQ)<br>
iii. Unsupervised QPP: [Common QPP methods](https://github.com/ChuanMeng/QPP4CS/blob/main/unsupervisedQPP/pre_retrieval.py)<br>
### Quickstart

Use the following 3 commands to run LoRA **fine-tuning**, **inference** and **merging** of the Llama3-8B-Instruct model, respectively.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

## References

If this work is helpful, please kindly cite as:

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Neel Nanda and Joseph Bloom},
    year = {2022},
    howpublished = {\url{https://github.com/TransformerLensOrg/TransformerLens}},
}
@inproceedings{aliannejadi2021building,
    title={Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions},
    author={Mohammad Aliannejadi and Julia Kiseleva and Aleksandr Chuklin and Jeff Dalton and Mikhail Burtsev},
    year={2021},
    booktitle={{EMNLP}}	 
}
@article{ karpukhin2020dense,
    title={ Dense Passage Retrieval for Open-domain Question Answering },
    author={ Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau },
    journal={ arXiv preprint arXiv:2004.04906 },
    year={2020}
}
@inproceedings{meng2023system,
  title={System Initiative Prediction for Multi-turn Conversational Information Seeking},
  author={Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={1807--1817},
  year={2023}
}
```
