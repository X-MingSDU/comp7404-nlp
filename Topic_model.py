
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


from gensim import corpora
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
import gensim


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


class MyData(Dataset):
    def __init__(self, vec):
        self.vec = vec

    def __getitem__(self, idx):
        res = torch.from_numpy(self.vec[idx]).float()
        return res

    def __len__(self):
        return self.vec.shape[0]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded


class Topic:
    def __init__(self, texts, common_texts, class_name, method='TFIDF', k=3):
        """
        k: Number of Topics
        texts:
        """
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = corpora.Dictionary(common_texts)
        self.common_corpus = [self.dictionary.doc2bow(text) for text in common_texts]
        self.cluster_model = None
        self.ldamodel = None
        self.vec_dict = {'TFIDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.lab_dict = {'TFIDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.gamma = 15
        self.method = method
        self.texts = texts
        self.common_texts = common_texts
        self.class_name = class_name


    def vectorize(self, method=None):
        if method == None:
            method = self.method

        if method == 'LDA':
            lda_model = gensim.models.ldamodel.LdaModel(self.common_corpus, num_topics=self.k, alpha='auto',
                                                        id2word=self.dictionary, passes=20)
            n_comments = len(self.common_corpus)
            vec_lda = np.zeros((n_comments, self.k))

            for i in range(n_comments):
                for topic, prob in lda_model.get_document_topics(self.common_corpus[i]):
                    vec_lda[i, topic] = prob
            vec = vec_lda

        elif method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            tfidf_vec = tfidf.fit_transform(self.texts)
            print('Getting vector representations for TF-IDF. Done!')
            vec = tfidf_vec

        elif method == 'BERT':
            model = SentenceTransformer('all-MiniLM-L12-v2')
            vec_bert = np.array(model.encode(self.texts, show_progress_bar=True))

            vec = vec_bert

        elif method == 'LDA_BERT':
            vec_lda = self.vec_dict['LDA']
            vec_bert = self.vec_dict['BERT']
            if type(vec_lda) != np.ndarray or type(vec_bert) != np.ndarray:
                raise Exception('please vectorize and fit LDA/BERT first!')
            vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]

            latent_dim = 32
            epochs = 200
            lr = 0.008
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            X = vec_ldabert
            input_dim = X.shape[1]
            trainset = MyData(X)
            train = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)

            model = Autoencoder(input_dim, latent_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_function = nn.MSELoss()

            for epoch in range(epochs):
                for inputs in train:
                    # Forward
                    inputs = inputs.to(device)
                    codes, decoded = model(inputs)

                    # Backward
                    optimizer.zero_grad()
                    loss = loss_function(decoded, inputs)
                    loss.backward()
                    optimizer.step()

            vec, _ = model(torch.from_numpy(X).float().to(device))
            vec = vec.data.cpu().numpy()
        self.vec_dict[method] = vec
        return vec

    def fit(self, method=None, m_clustering=KMeans):
        if method == None:
            method = self.method
        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.common_corpus, num_topics=self.k, alpha='auto',
                                                                id2word=self.dictionary, passes=20)
                print('Fitting LDA Done!')

        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            testmodel = self.cluster_model.fit(self.vec_dict[method])
            print('Clustering embeddings. Done!')
            lbs = testmodel.labels_
            self.lab_dict[method] = lbs
