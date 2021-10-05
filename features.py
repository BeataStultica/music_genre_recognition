import os
import time
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import librosa
import ast
import warnings
import pickle

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')


# функція для загрузки, очищення і представлення csv файлів датасету в потрібному вигляді
def load(filepath):
    filename = os.path.basename(filepath)
    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

# функція яка обчислює різні показники аудіозаписів і записує їх в csv файл, далі вони використовуються для
# класифікації музики по жанрах
def mfcc_write():
    tracks = load('tracks.csv')
    medium = tracks[tracks['set', 'subset'] <= 'medium']
    print(medium.head().to_string())
    indexs = medium.index
    mfcc_features = list()
    for i in indexs[0:10]:
        id_str = '{:06d}'.format(i)
        filepath = os.path.join('fma_medium', id_str[:3], id_str + '.mp3')
        y, sr = librosa.load(filepath, sr=None, mono=True)
        mfcc_features.append(librosa.feature.mfcc(y=y, sr=sr))
    mfcc_features = np.array(mfcc_features)
    print(mfcc_features.shape)
def features_write():
    tracks = load('tracks.csv')
    medium = tracks[tracks['set', 'subset'] <= 'medium']
    print(medium.head().to_string())
    indexs = medium.index
    features = pd.DataFrame(index=indexs, columns=['mfcc', 'chroma_cens', 'tonnetz', 'spectral_contrast',
                                                   'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                                                   'rmse', 'zcr'])
    for i in indexs:
        id_str = '{:06d}'.format(i)
        filepath = os.path.join('fma_medium',id_str[:3], id_str + '.mp3')
        print(filepath)
        try:
            x, sr = librosa.load(filepath, sr=None, mono=True)
            cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                     n_bins=7 * 12, tuning=None))
            stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
            mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
            mfcc = np.mean(librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))
            chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
            tonnetz = np.mean(librosa.feature.tonnetz(chroma=chroma_cens))
            chroma_cens = np.mean(chroma_cens)
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, n_bands=6))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=stft))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=stft))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=stft))
            rmse = np.mean(librosa.feature.rms(S=stft))
            zcr = np.mean(librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
        except:
            mfcc = None
            chroma_cens = None
            tonnetz = None
            spectral_contrast = None
            spectral_centroid = None
            spectral_bandwidth = None
            spectral_rolloff = None
            rmse = None
            zcr = None
        features.loc[i] = [mfcc, chroma_cens, tonnetz, spectral_contrast,
                                                   spectral_centroid, spectral_bandwidth, spectral_rolloff,
                                                   rmse, zcr]
    features.to_csv('features_short.csv', sep=',')
    print(features.head().to_string())
def data_prepare():
    tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])
    # tracks = load('tracks.csv')
    genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
    print('Top genres ({}): {}'.format(len(genres), genres))
    medium = tracks[tracks['set', 'subset'] <= 'medium']
    features = pd.read_csv('features_short.csv', index_col=0)
    # features = features.drop('tonnetz',1)
    scaler = StandardScaler(copy=False)
    print(tracks.head().to_string())
    print(medium.head().to_string())

    train = medium[medium['set', 'split'] != 'test']
    test = medium[medium['set', 'split'] == 'test']
    y_train = train['track']['genre_top']
    y_test = test['track']['genre_top']
    print('------')
    print(y_test.head(10).to_string())
    print(y_train.head(10).to_string())
    X_train = features.loc[features.index.isin(train.index)]
    # видаляємо записи з NaN, тобто ті, де аудіозапис був пошкоджений
    X_train.dropna(inplace=True)
    print(X_train.head().to_string())
    X_test = features.loc[features.index.isin(test.index)]
    X_test.dropna(inplace=True)
    # видаляємо відповідні записи з y наборів
    y_train = y_train.loc[y_train.index.isin(X_train.index)]
    y_test = y_test.loc[y_test.index.isin(X_test.index)]
    print(y_test.head().to_string())
    print(y_train.head().to_string())
    print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
    print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

    # нормалізуємо дані
    scaler.fit_transform(X_train)
    scaler.fit_transform(X_test)
    #print((X_train[:10]))
    return X_train,X_test,y_train,y_test
def genre_classifier():
    X_train, X_test, y_train, y_test = data_prepare()
    classifiers = {
        'LR': LogisticRegression(),
        'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5),
        'kNN': KNeighborsClassifier(n_neighbors=200),
        'SVCrbf': SVC(kernel='rbf'),
        'SVCpoly1': SVC(kernel='poly'),
        'linSVC1': SVC(kernel="linear"),
        'linSVC2': LinearSVC(),
        'DT-5_gini': DecisionTreeClassifier(max_depth=5, criterion='gini'),
        'DTFull_gini': DecisionTreeClassifier(criterion='gini'),
        'DT-5_entropy': DecisionTreeClassifier(max_depth=5, criterion='entropy'),
        'DTFull_entropy': DecisionTreeClassifier(criterion='entropy'),
        'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
        'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
        'QDA': QuadraticDiscriminantAnalysis(),
    }
    # навчаємо модель дерева класифікації і оцінюємо її точність
    scoresdict = {}
    timesdict = {}
    for clf_name, clf in classifiers.items():
        print('\n')
        print(clf_name)
        t = time.process_time()
        clf.fit(X_train, y_train)
        #score = clf.score(X_test, y_test)
        predicts = clf.predict(X_test)
        with open(clf_name+'.pkl', 'wb') as f:
            pickle.dump(clf, f)
        cm = confusion_matrix(y_test, predicts)
        cmd = ConfusionMatrixDisplay(cm, display_labels=['Blues', 'Classical', 'Country', 'Easy Listening',
                                                         'Electronic','Experimental', 'Folk', 'Hip-Hop','Instrumental', 'International',
                                                            'Jazz', 'Old-Time/Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken'])
        print(cm)
        cmd.plot(xticks_rotation='vertical' )
        plt.show()
        print(classification_report(y_test, predicts, zero_division=0))
        #if clf_name == 'RF':
        #    importances = pd.DataFrame(
        #    {'feature': X_train.columns, 'importance': np.round(clf.feature_importances_, 3)})
        #    importances = importances.sort_values('importance', ascending=False).set_index('feature')
        #    print(importances)
        scoresdict[clf_name] = clf.score(X_test, y_test)
        timesdict[clf_name] = time.process_time() -t
        print('\n')
        for clf_n, scores in scoresdict.items():
            print('{:<15}:{:>10f}'.format(clf_n, scores))
        for clf_n, times in timesdict.items():
            print('{:<15}:{:>10f}'.format(clf_n, times))
    for clf_n, scores in scoresdict.items():
        print('{:<15}:{:>10f}'.format(clf_n, scores))
    for clf_n, times in timesdict.items():
        print('{:<15}:{:>10f}'.format(clf_n, times))




def genre_define(filepath, algor='all'):
    algoritms = ['LR',
    'RF',
    'kNN',
    'SVCrbf',
    'SVCpoly1',
    'linSVC1',
    'linSVC2',
    'DT-5_gini',
    'DT-5_entropy',
    'MLP1',
    'MLP2',
    'QDA']
    try:
        x, sr = librosa.load(filepath, sr=None, mono=True, duration=30)
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
        mfcc = np.mean(librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))
        chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        tonnetz = np.mean(librosa.feature.tonnetz(chroma=chroma_cens))
        chroma_cens = np.mean(chroma_cens)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, n_bands=6))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=stft))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=stft))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=stft))
        rmse = np.mean(librosa.feature.rms(S=stft))
        zcr = np.mean(librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
    except:
        mfcc = None
        chroma_cens = None
        tonnetz = None
        spectral_contrast = None
        spectral_centroid = None
        spectral_bandwidth = None
        spectral_rolloff = None
        rmse = None
        zcr = None
    features_music =pd.DataFrame({'mfcc':mfcc, 'chroma_cens':chroma_cens, 'tonnetz':tonnetz, 'spectral_contrast':spectral_contrast,
                                        'spectral_centroid':spectral_centroid, 'spectral_bandwidth':spectral_bandwidth, 'spectral_rolloff':spectral_rolloff,
                                        'rmse':rmse, 'zcr':zcr}, index=[1])

    if algor=='all':
        for i in algoritms:
            with open(i+'.pkl', 'rb') as f:
                clf = pickle.load(f)
            #t = time.process_time_ns()
            print(filepath + ' genre with '+i+' predict: ' + str(clf.predict(features_music)))
            #print(i + ' time: '+str(time.process_time_ns()-t))
    else:
        with open(algor + '.pkl', 'rb') as f:
            clf = pickle.load(f)
        #t = time.process_time_ns()
        print(filepath + ' genre with ' + algor + ' predict: ' + str(clf.predict(features_music)))



def create_spectogram(track_id):
    id_str = '{:06d}'.format(track_id)
    filename = os.path.join('fma_medium', id_str[:3], id_str + '.mp3')
    print(filename)
    y, sr = librosa.load(filename)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=1024)
    spect = librosa.power_to_db(spect, ref=np.max)
    return spect.T

dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4,
               'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8, 'Jazz':9,'Old-Time / Historic':10,
               'Soul-RnB':11, 'Spoken':12, 'Blues':13, 'Classical':14, 'Country':15, 'Easy Listening':16 }
tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])
keep_cols = [('set', 'split'),
                 ('set', 'subset'), ('track', 'genre_top')]



def create_array(df):

    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            spect = create_spectogram(track_id)

            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(dict_genres[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

def splitDataFrameIntoSmaller(df, chunkSize = 4750):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf


#mfcc_write()
#data_prepare()
#genre_classifier()
genre_define('MoonlightSonata.mp3', 'MLP1')
genre_define('Gorillaz.mp3', 'MLP2')
genre_define('TheProdigy.mp3', 'MLP1')
genre_define('Nirvana.mp3', 'MLP1')

