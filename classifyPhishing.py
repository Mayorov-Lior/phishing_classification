import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from nltk import word_tokenize
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
from gensim.models import FastText
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

def parse_url(url: str) -> pd.Series:
    """ Parses given string URL into URL scheme specifier, network location part, Hierarchical path"""
    url = urlparse(url)
    url_components = [url.scheme, url.netloc, url.path] # port only have 4 appearances, fragement-5, hostname seems like netloc minus port
    for index in range(len(url_components)): # handle missing components
        if url_components[index] is None:
            url_components[index] = ""
    return pd.Series(url_components)  

def embed_line(model_embd: FastText, line: str) -> pd.Series:
    """ Tokenizes given text using nltk and embeds it using given a FastText model
        note: the given text will be netloc and path components of the URL
        It's important to use a FastText model that handles OOV words, since the text will definately
        have OOV words like deliberated typos, connected words, random characters etc...
    """
    words = word_tokenize(re.sub('\.|-|/|:',' ', line))
    if len(words) == 0:
        words = ['']
    
    embeddings = [model_embd.wv[word] for word in words]
    # we use both the sum and the mean of the embeddings
    sum_embeddings = np.sum(embeddings, axis=0)
    mean_embeddings = sum_embeddings/len(embeddings)
    
    return pd.Series(np.concatenate((sum_embeddings, mean_embeddings)))

def add_embedding_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Receives a df with 'netloc' and 'path' columns and adds columns with their sum + mean 
        embeddings using embed_line function 
    """
    cap_path = datapath("cp852_fasttext.bin") # cp852_fasttext (2), lee_fasttext/_new (10), non_ascii_fasttext(2)
    model_embd = load_facebook_model(cap_path)
    
    embd_vector_length = 2
    netloc_embd_cols = [f'netloc_embd_sum_{i}' for i in range(embd_vector_length)] + [f'netloc_embd_mean_{i}' for i in range(embd_vector_length)]
    path_embd_cols = [f'path_embd_sum_{i}' for i in range(embd_vector_length)] + [f'path_embd_mean_{i}' for i in range(embd_vector_length)]

    df[netloc_embd_cols] = df['netloc'].apply(lambda s: embed_line(model_embd, s))
    df[path_embd_cols] = df['path'].apply(lambda s: embed_line(model_embd, s))
    return df

def preprocess_data(df: pd.DataFrame, length_mean: pd.Series = None, length_std: pd.Series = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """ Receives a dataframe and possibly standartization data if it's for the test set and
        calculates relevant features for classifying phishing and returns the df 
    """
    df[['scheme', 'netloc', 'path']] = df['url'].apply(parse_url)
    df['scheme'] = df['scheme'].apply(lambda s: 1 if s == 'https' else 0) # there's only http and https
    df['url_length'] = df['url'].apply(len)
    if length_mean == None or length_std == None: # if df is train data
        length_mean = df['url_length'].mean()
        length_std = df['url_length'].std()
    df['url_length'] = (df['url_length'] - length_mean)/length_std # standartization
    
    # as seen in https://www.expressvpn.com/blog/what-is-url-phishing/ section 5:
    # one has to be careful when the URL leads to subfolders. So here we add the depth of path (no subfolder, depth=1)
    df['path_depth'] = df['path'].apply(lambda s: len(s.split('/'))) 
    df['is_com'] = df['netloc'].apply(lambda s: int(s.split('.')[-1] == 'com')) # with more data, maybe should use more domain-indicator features
    
    df = add_embedding_features(df)
    df.drop(columns=['netloc', 'path', 'url'], inplace=True)
    
    return df, length_mean, length_std

if __name__ == "__main__":
    
    ''' Reading the data and splitting to train-test
        note: with so little data, we splitted the data only to train and test. 
              but usually we should include a val/dev set 
    '''

    df = pd.read_excel("dataset_phishing.xlsx")
    df['status'] = df['status'].apply(lambda s: 1 if s == 'phishing' else 0)

    train_size = 0.8
    train_df = df.iloc[:int(train_size*len(df))]
    test_df = df.iloc[int(train_size*len(df)):]

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    # preprocess train and test data + feature engineering
    train_df, length_mean, length_std = preprocess_data(train_df)
    test_df, _, _ = preprocess_data(test_df, length_mean, length_std)
    
    """
    Model Training:

    Since we only have 1000 examples overall, we can't train a deep learning model.
    We'll stick to simpler machine learning models.

    We'll go with an ensemble classification model for best outcomes.
    After some maual experiments and little manual hpt, the best model was found to be random forest, 
    beating xgboost and others. But with so little data, it could be overfitting

    """

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=80, max_depth=10, random_state=0).fit(
        train_df.loc[:, train_df.columns != 'status'].values, train_df['status'].values)

    pred = clf.predict_proba(test_df.loc[:, test_df.columns != 'status'].values)
    pred_round = np.argmax(pred,axis=1)

    # evaluate predictions:
    
    print ('accuracy: ', np.round(sklearn.metrics.accuracy_score(test_df['status'].values, pred_round),4))
    print ('recall: ', np.round(sklearn.metrics.recall_score(test_df['status'].values, pred_round),4))
    print ('precision: ', np.round(sklearn.metrics.precision_score(test_df['status'].values, pred_round),4))
    print ('f1: ', np.round(sklearn.metrics.f1_score(test_df['status'].values, pred_round),4))
    
    """
    We went over some of the basic metrics.
    The accuracy shows the overall score, how many of the URLs we classified correctly, phishing or legitimate.
    The recall shows how many of the phishing examples we succesfully identified.
    The precision shows how many of what we identified as phishing actually is phishing
    
    It looks like we did okay and about the same in the four metrics.

    while it's always good to view over a few metrics, choosing the most appropriate metric 
    depends on the objective and what you want to achieve.
    If falsely alerting phishing is severe, we should especially make sure our precision is high.
    If missing out phishing URLs is unacceptable, we should especially make sure our recall is high.

    If both is equally important, we should check out F1.
    And in that case, it's also important to check the auc roc score, that maps the relationship 
    between tpr and fpr of the model across different thresholds. It mostly shines when the classes 
    are imbalanced and that's not the case here at all but let's watch it anyway:
    """
    
    fpr, tpr, _ = sklearn.metrics.roc_curve(test_df['status'].values, pred[:,1])
    auc = np.round(sklearn.metrics.roc_auc_score(test_df['status'].values, pred[:,1]),4)
    plt.plot(fpr,tpr)
    plt.title(f'AUC = {auc}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    """
    We see that the auc is quite high (0.87), meaning the separability of the problem seems feasable and 
    quite successful, considering the little data and simple training.
    
    How to improve:
    1. Increase dataset
    2. Fine tune the fasttext model with URL data (not limited to our dataset, no need for labels)
    3. Hyper parameter tuning of the architecture, try different classification models, different feature groups
    4. Add data outside of the URL, like how was it received, was it attached to an image, etc..
    5. With larger dataset this isn't needed but for this maybe check similarity of URL to top k popular website
        for example, substitute numbers in similiar letters (1 -> l, 0 -> O etc) and see if it creates a valid word (g00g1e...). 
        probablly going too far but could also do it with computer vision, perhaps with character-pretrained auto-encoder and use 
        the latent-space similarity as a feature
    
    """