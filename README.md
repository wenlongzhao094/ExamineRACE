# ExamineRACE

This repository contains pytorch implementation of [BERT](https://arxiv.org/abs/1810.04805) and [DCMN](https://arxiv.org/abs/1901.09381), and the finetuning runners for both models on [RACE](http://www.qizhexie.com/data/RACE_leaderboard). This repository also contains reasoning skills proposed for RACE and annotation of 300 randomly sampled test questions.

- **pytorch_pretrained_bert** contains [the Huggingface implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT). BERT followed by DCMN is added to the implementation. 
- **run_bert_on_race.py** and **run_dcmn_on_race.py** are the finetune runners.
- **run.sh** runs the finetune runners.
- **race_skills.csv** contains a list of reasoning skills required by questions in RACE.
- **middle_categorization.csv** and **high_categorization.csv** each contains annotation of 150 questions, and are respectively for middle and high school multiple choice questions.
- **Stats.ipynb** is a notebook for calculating model test accuracy on questions that require different skills.
