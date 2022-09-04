# COMP7404 
Sentiment Classification on COVID-19 based on BERT and LSTM model
## Dataset
First, you need to prepare COVID-19 comments data which are publicly available. Format used here is one review per line, with 10000 lines.(11 labels per tweet) you can simply download dataset on comp7404/BERT/senwave_preprocess.
## Bert
BERT is state-of-the-art natural language processing model from Google. Using its latent space, it can be repurpossed for various NLP tasks, such as sentiment analysis.
## LSTM
Long short-term memory (LSTM) is an artificial neural network used in the fields of artificial intelligence. Unlike standard feedforward neural networks, LSTM has feedback connections.
## Requirements and Installation 
In order to run the code,you'll need the following libraries.
- torch==1.10.2
- torchvision==0.11.3
- numpy
- pandas

For Bert model we'll use the transformers library, which can be installed via:
`pip install transformers`  
## References
- > SenWave: Monitoring the Global Sentiments under the COVID-19 Pandemic, Yang, Qiang and Alamro, Hind and Albaradei, Somayah and Salhi, Adil and Lv, Xiaoting and Ma, Changsheng and Alshehri, Manal and Jaber, Inji and Tifratene, Faroug and Wang, Wei and others [https://arxiv.org/pdf/2006.10842.pdf] (Note: If you want to use the labled tweets, please mail to qiang.yang[AT]kaust[dot]edu[dot]sa to get the pwd for the zip filefolder.)

- > Tweets originating from India during Covid-19 Lockdowns 1, 2, 3, 4 - [https://ieee-dataport.org/open-access/tweets-originating-india-during-covid-19-lockdowns-1-2-3-4]
## Demo Vedio
- [https://connecthkuhk-my.sharepoint.com:/g/personal/u3591502_connect_hku_hk/ETjpleDP5sdMmfqTcjIK6oABId29X6B8z1q1LxeG5MwYHw?e=gpmkQH]
- [https://drive.google.com/file/d/19ec2FzbM8e3WLzVt0DETDdGzxSyhzcdG/view?usp=sharing]
