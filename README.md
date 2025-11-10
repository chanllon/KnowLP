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

