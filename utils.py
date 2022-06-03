import numpy as np
import _pickle as pickle
from math import *
import torch 

def convert_index_to_word(cls_ids, classes):
    words = []
    for ids in cls_ids:
        b_words = []
        for idx in ids:
            cls = classes[idx]
            b_words.append(cls)
        words.append(b_words)
    words = np.array(words)
    return words


def read_glove_vecs(glove_file, dictionary_file):
    d = pickle.load(open(dictionary_file, 'rb'))
    word_to_vec_map = np.load(glove_file)
    words_to_index = d[0]
    index_to_words = d[1]
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].replace('-', ' ')
        sentence_words = (X[i].lower()).split()
        sentence_words = sentence_words[:max_len]
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]
            except:
                X_indices[i, j] = 0
            # Increment j to j + 1
            j = j + 1
    return X_indices

def class_embedding(sentence, word2vec, emb_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, num_class, max_words = sentence.shape
    rnn_size = 1536
    sentence = torch.reshape(sentence, [batch*num_class, max_words])
    sentence = sentence.to(torch.int32)
    # create word embedding
    embed_ques_W = word2vec
    # create LSTM
    lstm_1 = torch.nn.LSTMCell(emb_dim, rnn_size, device=device)
    lstm_dropout_1 = torch.nn.Sequential(lstm_1, torch.nn.Dropout(0.2))
    lstm_2 = torch.nn.LSTMCell(rnn_size, rnn_size)
    lstm_dropout_2 = torch.nn.Sequential(lstm_2, torch.nn.Dropout(0.2))
    stacked_lstm = torch.nn.Sequential(lstm_dropout_1, lstm_2)
    hx = torch.zeros(batch*num_class, rnn_size, device=device)
    cx = torch.zeros(batch*num_class, rnn_size, device=device)
    for i in range(max_words):
        cls_emb_linear = torch.nn.functional.embedding(sentence[:, i],embed_ques_W)
        # embedded_users = torch.unsqueeze(embedded_user, -1)
        # cls_emb_linear = tf.nn.embedding_lookup(embed_ques_W, sentence[:, i])
        cls_emb_drop = torch.nn.Dropout(0.2)(cls_emb_linear)
        cls_emb = torch.nn.Tanh()(cls_emb_drop)
        hx, cx = lstm_1(cls_emb, (hx, cx))

    output = hx
    output = torch.reshape(output, [batch, num_class, rnn_size])
    return output