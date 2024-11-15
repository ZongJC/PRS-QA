1.Dependencies:
======================
python == 3.7 <br>
torch == 1.8.1 + cu111 <br>
torch-geometric == 1.7.0 <br>
torch-scatter == 2.0.7 <br>
torch-sparse = 0.6.9 <br>
transformers == 3.4. <br>
Run the following commands to create a conda environment: <br>
CUDA Version: 12.2 <br>
conda create -n PRS-QA python=3.7 <br>
conda activate PRS-QA <br>

2.Dataset download link：
=======================
| CommonsenseQA | OpenBookQA | medqa-usmle |  
| :--- | :--- | :--- |  
| https://www.tau-nlp.org/commonsenseqa | https://huggingface.co/datasets/allenai/openbookqa | https://github.com/jind11/MedQA |  


Download all the raw data -- ConceptNet, CommonsenseQA, OpenBookQA -- by <br>
```./download_raw_data.sh```
```python preprocess.py -p <num_processes>```


3.Pre-trained language model download link:
==========================================
| Roberta-large | aristo-roberta | SapBERT |  
| :--- | :---| :--- |  
| https://huggingface.co/FacebookAI/roberta-large | https://huggingface.co/LIAMF-USP/aristo-roberta | https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext |  
