"""
**Riepilogo del progetto:**

**Pre-processing**

Prima di procedere all'addestramento, è stata implementata una funzione di  
pulizia del testo con rimozione di eventuale punteggiatura e stopwords e
portando le restanti parole alla forma base (usando lemma). Per questa funzione
è stato necessario scaricare prima dei language models da spacy.
In particolare, per l'italiano e il tedesco è stato necessario fare prima
una installazione.

In seguito, è stata fatta la suddivisione in bigrammi e creazione di Bag Of Words
attraverso vettorizzatori che restituiscono vettori di occorrenze di bigrammi.

**Addestramento**

Il dataset risulta molto piccolo (294 frasi) e perfettamente equilibrato nel
numero di frasi tra le 3 lingue presenti.

L'addestramento è stato eseguito sia con Hold Out Validation che con
Cross Validation, utilizzando diversi vettorizzatori e classificatori.

Come prevedibile, data la dimensione del dataset, con la Hold Out Validation
si ottengono performance migliori se il set di test è tra il 25% ed il 30% del
totale.

In questo caso gli indici di performance sono quasi unitari, favorendo il
sospetto di overfitting considerando anche le dimensioni ridotte del database.

Tuttavia, anche procedendo con Cross Validation, le performance rimangono
molto buone, con accuratezza appena superiore al 95% con il Multi-Layer
Perceptron (MLP) Classifier e al 97% con il Random Forest Classifier.

A proposito di Cross Validation, gli addestramenti sono stati eseguiti con 4 o
più batches fino anche a 10 batches, ottenendo valori pressoché identici in
tutti questi casi. Usare 5 batches è quindi un buon compromesso per mantenere
performance e tempi di addestramento a livelli ottimali.

**Conclusioni**

In generale, usando CountVectorizer per la vettorizzazione, i classificatori
più performanti sono il Random Forest Classifier ed il MLP, ma richiedono
anche più tempo di addestramento.

D'altro canto, il Multinomial Naive Bayes ed il Support Vector Classifier, sono
estremamente veloci, pur mantenendo indici di performance al 93-94%, quindi
molto buoni.

Se si passa all'uso di TfidfVectorizer, tutti i classificatori calano di
performance, ad eccezione del SVC che aumenta l'accuratezza al 97%.
In particolare, il MLP Classifier a parità di iperparametri ottiene un
peggioramento sensibile delle performance. Per avere performance ancora
accettabili bisogna agire sugli iperparametri, incrementando però di molto
i tempi di addestramento.

**Scelta del modello**

Comunque in tutti i casi, viste le dimensioni del dataset, i tempi di addestramento rimangono in assoluto piuttosto brevi anche per i classificatori più lenti, che  rimangono intorno ai 30 secondi. Per questo motivo opterei per un addestramento fatto con Random Forest Classifier, preceduto da Cross Validation (5 batch), e l'uso di CountVectorizer e, che nel complesso è risultata la combinazione migliore in termini di Accuracy, ma anche di altri indici di performance.

**Nota**

Naturalmente il modello non si presta bene ad identificare lingue da ogni
testo, in quanto l'addestramento è stato fatto su un argomento specifico.
"""

import pandas as pd
import string
import spacy
import nltk
nltk.download('stopwords')    #Download stopwords
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OrdinalEncoder    #Import encoder
from sklearn.metrics import mean_squared_error, r2_score, log_loss, confusion_matrix, ConfusionMatrixDisplay, classification_report, get_scorer_names     #Import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    #Import Vectorizers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict    #Import validators
from nltk import ngrams
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier                              #Import classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

"""
Download of language models needed, to use in terminal

python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python -m spacy download de_core_news_sm
"""

def ngram_f(text, n):                                     #Function that receives in input a text and the number of elements (character) for each gram to be created
    n_gram_text = ngrams(text, n)                         #Function 'ngrams' imported from nltk splits the text in grams of n elements
    return ["".join(ngr) for ngr in n_gram_text]          #Returns the merge of each ngram in n_gram_text


def data_cleaner(ds, mnlp):                               #The function receive a dataset and cleans it through some steps
  ds_to_return = []                                       #Creates an empty list to store the cleaned dataset to return

  for i in range(len(ds)):
    sentence = ds['Testo'].iloc[i].lower()                #Allocation of sentence using position indexing .iloc
    code = ds['Codice Lingua'].iloc[i]                    #Allocation of code using position indexing .iloc

    if code == "it":          #Chooses the proper language model according to the code ('Codice Lingua')
      nlp = mnlp[0]
    elif code == "en":
      nlp = mnlp[1]
    else:
      nlp = mnlp[2]

    for c in string.punctuation:
      sentence = sentence.replace(c, " ")                 #The inside cicle removes all punctuation, puts a space in its place
    document = nlp(sentence)                              #The sentence is analyzed word by word, each one as a token to be processed in next steps
    sentence = " ".join([token.lemma_ for token in document])       #Each word is reduced to "base" form
    sentence = " ".join([word for word in sentence.split() if word not in stop_words])      #Removes stopwords from each sentence
    ds_to_return.append(sentence)                         #Updates list of sentences in each iteration

  return ds_to_return


def bow_count(ds, count_vectorizer):                      #Function that receives a dataset and a class "CountVectorizer()" (from sklearn) in input
  if count_vectorizer == None:
    count_vectorizer = CountVectorizer()                  #Initializes count_vectorizer if empty
    X = count_vectorizer.fit_transform(ds)                #Converts the text in a matrix of token counts, which shows the occurrence of each word in each sentence

  else:
    X = count_vectorizer.fit_transform(ds)                #Converts the text in a matrix of token counts, which shows the occurrence of each word in each sentence

  return X.toarray(), count_vectorizer                    #Returns the matrix in a dense array format and the count_vectorizer


def bow_tfidf (ds, tfidf_vectorizer):                     #Function that receives a dataset and a class "TfidfVectorizer()" (from sklearn) in input
  if tfidf_vectorizer == None:
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(ds)

  else:
    X = tfidf_vectorizer.transform(ds)

  return X.toarray(), tfidf_vectorizer



#The next 2 functions evaluate the performance with indexes like MSE and R2, which are more suitable for regression problemsinstead of classification ones like this one
#Nevertheless, they are used to show the difference between indexes

def hold_out_regression(mode, data_set, objective_encoded, size):

  # Split the dataset into training and testing sets
  X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(bow_data_set, objective_encoded, test_size=size, random_state=42)

  mode.fit(X_train, y_train_encoded)                              #Execution of the training with training portion

  print("MSE: ", str(mean_squared_error(y_test_encoded, mode.predict(X_test))))      #Prints the Mean Squared Error using the function from sklearn.metrics
  print("R2: ", str(r2_score(y_test_encoded, mode.predict(X_test))), "\n")                 #Prints the value of R^2 calculated with the function "r2_score" from sklearn.metrics

  cm = confusion_matrix(y_test_encoded, mode.predict(X_test))     #Creates the confusion matrix with test data

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)      #Creates a heatmap of the confusion matrix
  disp.plot(cmap="Blues")                                                             #Makes the map in blue
  plt.title("Confusion matrix - Hold Out Validation")                                 #Title of the confusion matrix
  plt.show()                                                                          #Shows the plot


def cross_regression(mode, bow_data_set, objective_encoded, k):

  mode_score = cross_validate(mode, bow_data_set, objective_encoded, cv=k, scoring='r2', return_train_score=True)   #Prints report with performance indexes

  print("\nR2 for each batch:", mode_score['test_score'])        #Prints list of R2 index for each batch
  print("\nAverage R2:", mode_score['test_score'].mean(), "\n")        #Prints the average of R2 indexes

  y_pred = cross_val_predict(mode, bow_data_set, objective_encoded, cv=5)       #Predictions with cross-validation

  cm = confusion_matrix(objective_encoded, y_pred)          #Creates the confusion matrix

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)        #Creates a heatmap of the confusion matrix
  disp.plot(cmap="Blues")                                                               #Makes the map in blue
  plt.title("Confusion matrix - Cross Validation")                                      #Title of the confusion matrix
  plt.show()                                                                            #Shows the plot



#The last 2 functions evaluate the performance with indexes like accuracy, precision, recall and F1-score, more suitable for classification problems


def hold_out_classification(mode, bow_data_set, objective_encoded, size):

  # Split the dataset into training and testing sets

  X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(bow_data_set, objective_encoded, test_size=size, random_state=42)

  mode.fit(X_train, y_train_encoded)                              #Execution of the training with training portion

  score = mode.score(X_test, y_test_encoded)                      #Average accuracy on test of the trained network with test portion

  print(f"\nThe R2 score with test size of {size} is: {score}\n")      #Prints the performance using score

  print(classification_report(y_test_encoded, mode.predict(X_test)))   #Prints the classification report with accuracy, precision, recall and F1-score

  cm = confusion_matrix(y_test_encoded, mode.predict(X_test))     #Creates the confusion matrix with test data

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)        #Creates a heatmap of the confusion matrix
  disp.plot(cmap="Blues")                                                               #Makes the map in blue
  plt.title("Confusion matrix - Cross Validation")                                      #Title of the confusion matrix
  plt.show()                                                                            #Shows the plot


def cross_classification(mode, bow_data_set, objective_encoded, k):

  mode_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']    #Defines a list with the performance indexes to be shown with mode_score

  mode_score = cross_validate(mode, bow_data_set, objective_encoded, cv=k, scoring=mode_metrics, return_train_score=True)   #Prints report with performance indexes

  print("\nAccuracy for each batch:", mode_score['test_accuracy'])                #Prints list of accuracy indexes for each batch
  print("Precision for each batch:", mode_score['test_precision_macro'])        #Prints list of precision indexes for each batch
  print("Recall for each batch:", mode_score['test_recall_macro'])              #Prints list of recall indexes for each batch
  print("F1 for each batch:", mode_score['test_f1_macro'], "\n")                      #Prints list of F1-score indexes for each batch

  y_pred = cross_val_predict(mode, bow_data_set, objective_encoded, cv=5)       #Predictions with cross-validation

  print(                          #Prints the classification report with accuracy, precision, recall and F1-score
    classification_report(
        objective_encoded,        #Labels on all the dataset
        y_pred,                   #Predictions
        target_names=le.classes_  #Classes names
    )
  )

  cm = confusion_matrix(objective_encoded, y_pred)          #Creates the confusion matrix

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)        #Creates a heatmap of the confusion matrix
  disp.plot(cmap="Blues")                                                               #Makes the map in blue
  plt.title("Confusion matrix - Cross Validation")                                      #Title of the confusion matrix
  plt.show()                                                                            #Shows the plot

# Initialize spacy and stopwords here as they are used in the data_cleaner function

english_stopwords = stopwords.words('english')    #Initialize a list of str with english stopwords
italian_stopwords = stopwords.words('italian')    #Initialize a list of str with italian stopwords
german_stopwords = stopwords.words('german')      #Initialize a list of str with german stopwords

nlp_en = spacy.load("en_core_web_sm")             #Loads an English language model from spacy
nlp_it = spacy.load("it_core_news_sm")            #Loads an Italian language model from spacy
nlp_de = spacy.load("de_core_news_sm")            #Loads a German language model from spacy

nlp = [nlp_it, nlp_en, nlp_de]                    #List with all the language models in the correct order

punctuation = set(string.punctuation)             #Initialize a set composed of punctuation taken from the "string" module

stop_words_en = english_stopwords                 #English not significative words
stop_words_it = italian_stopwords                 #Italian not significative words
stop_words_de = german_stopwords                  #German not significative words

stop_words = stop_words_en + stop_words_it + stop_words_de        #List of all not significative words


#First step is to store the link in the "BASE_URL" value

BASE_URL = "https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv"

df = pd.read_csv(BASE_URL)                       #Panda function to read a csv file and store it in df (dataframe)

languages = set(df['Codice Lingua'])             #Creates a set (to avoid repetitions) with all the language codes under the column "Codice Lingua"

for item in languages:
  print(item + ": " + str(len(df[df['Codice Lingua'] == item])))        #Prints how many occurrences there are for each language code in the dataframe

objective = df['Codice Lingua']                  #Extract of the dataframe with 2 elements for each row: the number of the row and the language code

clean_data = data_cleaner(df, nlp)              #List of str type each one made with the sentence with no stopwords and punctuation (cleaned)

data_set = [ngram_f(row, 2) for row in clean_data]        #Creates a dataset which contains a list: each element is a list for each sentence splitted in bigrams
                                                          #and epurated from stopwords

data_set = [" ".join(row) for row in data_set]            #The dataset is now a list of strings (bigrams) with no stopwords


#CountVectorizer

bow_data_set, vectorizer = bow_count(data_set, None)       #Function that returns a matrix of token counts (bow_data_set) and a dictionary (CountVectorizer)
                                                           #which contains each bigram as key and their occurrences as item

#TFIFDVectorizer

# bow_data_set, vectorizer = bow_tfidf(data_set, None)     #Function that returns a matrix of token counts (bow_data_set) and a dictionary (TfidfVectorizer)
                                                           #which contains each bigram as key and their occurrences as item

print(f"\nThe Bag of Words dataset has {len(bow_data_set)} elements.")
print(f"\nThe bigrams in the vectorizer are: {len(vectorizer.vocabulary_)}")

# Encode the target variable
le = LabelEncoder()
objective_encoded = le.fit_transform(objective)


#Definition of the classifier Multi-Layer Perceptron

clf = MLPClassifier(activation = "logistic",          #Defines a classifier with activation "logistic": this activation is ideal for text classification
                    hidden_layer_sizes=(100,),        #One layer with size 100: the performance tends to decrease with more or less neurons
                    max_iter=200,                     #Max number of iterations
                    solver="adam",                    #Most used solver for NPL
                    tol=0.001,                        #Tolerance under which the code stops to iterate after 10 iterations
                    #verbose=True                     #To visualize the training
                    random_state=42                   #To make the training repeatable
                    )


#Definition of the classifier Multinomial Naive-Bayes

munb = MultinomialNB()


#Definition of the classifier Support Vector Classifier

svc = SVC()


#Definition of the classifier Random Forest Classifier

rfc = RandomForestClassifier(
    n_estimators = 400,                     #Number of trees and ax_features which give better performances
    max_features = 2,
    n_jobs = -1,                            #Number of jobs to run in parallel = -1 to reduce the time of training
    criterion = "gini",
    random_state = 42
)

#Validation

print("Overview of performance indexes and confusion matrix\n\n")
#The functions below apply the hold out validation and the cross validation to the dataset.

#They both receive in input a classifier (clf, munb, svc, rfc), the "bow_data_set" and the "objective_encoded".

cls = munb


#Hold out validation test size can be modified by changing the value of the 4th input (0 to 1)
print("Regression performance indexes for", cls, "\n")
hold_out_regression(cls, bow_data_set, objective_encoded, 0.3)

print("\nClassification performance indexes for", cls, "\n")
hold_out_classification(cls, bow_data_set, objective_encoded, 0.3)


#Cross validation number of batches can be modified by changing the value of the 4th input

print("\nRegression performance indexes for", cls, "\n")
cross_regression(cls, bow_data_set, objective_encoded, 5)

print("\nClassification performance indexes for", cls, "\n")
cross_classification(cls, bow_data_set, objective_encoded, 5)

import pickle

# Dizionario con tutto ciò che serve per la prediction
model_artifacts = {
    "model": cls,                 # il classificatore addestrato (es. rfc)
    "vectorizer": vectorizer,     # CountVectorizer o TfidfVectorizer
    "label_encoder": le           # LabelEncoder
}

# Salvataggio su file
with open("language_detector.pkl", "wb") as f:
    pickle.dump(model_artifacts, f)

print("Modello salvato correttamente")