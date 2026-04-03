## Overview
This project consists of two modules: EDU-graphRAG and DLRL. You need to run the content of these two parts separately.

## Prerequisites
To run this project, the following software and libraries are required:

- Python 3.9.21
- PyTorch 2.4.1
- Gym 0.23.0
- graphRAG 0.2.2

## Usage
1. To run the EDU-graphRAG module:

   - Open the `settings.yaml` file and add the following entries:
     ```yaml
     openai_chat: YOUR_OPENAI_CHAT_API_KEY
     openai_embedding: YOUR_OPENAI_API_BASE
     ```

   - Generate the knowledge graph using the following command:

     ```bash
     python -m graphrag.index --root ./ragtest
     ```
     
2. To run the KnowLP module:
   
   Navigate to the KnowLP folder and execute the following command:

   ```bash
   python DLELP.py
   ```
## Citation
If you find our work helpful, please kindly cite our research paper:

[1]Xinghe Cheng, Zihan Zhang, Jiapu Wang, Liangda Fang, Chaobo He, Quanlong Guan, Shirui Pan, Weiqi Luo. GraphRAG-induced dual knowledge structure graphs for personalized learning path recommendation. AAAI. 2026

```
@article{Cheng2026, title={GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation},
volume={40},
DOI={10.1609/aaai.v40i17.38479},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Cheng, Xinghe and Zhang, Zihan and Wang, Jiapu and Fang, Liangda and He, Chaobo and Guan, Quanlong and Pan, Shirui and Luo, Weiqi},
year={2026}, month={Mar.}, pages={14610-14620} }

```
