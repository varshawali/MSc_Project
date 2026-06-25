
# Project Title

Performance Analysis and enhancement of DeepSC

# Abstract
This Project focuses on enhancing the existing DeepSC model, which integrates deep learning techniques for semantic communication. The goal is to improve how information is encoded, transmitted, and decoded over noisy channels, with a particular focus on preserving semantic accuracy. By exploring and refining deep learning-enabled approaches to communication, it seeks to enhance both sentence similarity and error rates while addressing the complexities of noisy environments.
The DeepSC architecture has been improved through the inclusion of techniques such as mutual information optimization, which aims to enhance the model's robustness and ensure efficient transmission of semantic information. By prioritizing the meaning of the transmitted data, the model reduces the loss of critical information, especially in challenging transmission environments. It also examines deep learning's capacity to optimize end-to-end communication systems, demonstrating significant improvements in communication efficiency and accuracy.


## Requirements
See the `requirements.txt` for the required python packages and run `pip install -r requirements.txt` to install them.
## Preprocess
Download Europarl dataset and use English texts.

python preprocess_text.py
## Train
python main.py 
## Evaluation
python performance.py

python sentencesimilarity.py
## Notes
This Project is Enhancement of DeepSC. The original work is available at https://github.com/13274086/DeepSC
