# Training del Modello di Riconoscimento Lingue

Questa cartella contiene tutto il necessario per addestrare il modello di riconoscimento delle lingue per contenuti museali.

## 📊 Dataset

Il modello è addestrato su **294 frasi** perfettamente bilanciate tra 3 lingue:
- **Italiano (it)**: 98 frasi
- **Inglese (en)**: 98 frasi
- **Tedesco (de)**: 98 frasi

Le frasi provengono da descrizioni di opere d'arte, pannelli informativi e contenuti didattici tipici di un contesto museale.

**Fonte dati**: Il dataset viene scaricato automaticamente da GitHub durante l'esecuzione dello script.

## 🔧 Setup

### 1. Installa le dipendenze

```bash
pip install -r ../requirements.txt
```

### 2. Scarica i modelli linguistici Spacy

**IMPORTANTE**: Questi modelli sono necessari per il preprocessing del testo.

```bash
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python -m spacy download de_core_news_sm
```

### 3. Verifica l'installazione

```python
import spacy
spacy.load("en_core_web_sm")
spacy.load("it_core_news_sm")
spacy.load("de_core_news_sm")
print("✓ Tutti i modelli Spacy installati correttamente")
```

## 🚀 Esecuzione del Training

### Training con parametri di default

```bash
python train_model.py
```

Questo comando:
1. Scarica il dataset
2. Esegue il preprocessing (lemmatizzazione, rimozione stopwords, creazione bigrammi)
3. Addestra il modello con Hold Out e Cross Validation
4. Salva il modello in `../models/museum_language_detector.pkl`

### Modificare il classificatore

Nel file `train_model.py`, alla riga con `cls = munb`, puoi scegliere tra:

```python
cls = munb   # Multinomial Naive Bayes (veloce, 93-94% accuracy)
cls = svc    # Support Vector Classifier (veloce, 93-97% accuracy)
cls = clf    # MLP Classifier (lento, 95% accuracy)
cls = rfc    # Random Forest Classifier (lento, 97% accuracy) - CONSIGLIATO
```

**Raccomandazione**: `rfc` (Random Forest) offre le migliori performance complessive.

### Modificare il vectorizer

Di default usa `CountVectorizer`. Per usare `TfidfVectorizer`, commenta/decommenta queste righe:

```python
# CountVectorizer (default)
bow_data_set, vectorizer = bow_count(data_set, None)

# TfidfVectorizer (alternativa)
# bow_data_set, vectorizer = bow_tfidf(data_set, None)
```

## 📈 Pipeline di Training

### 1. Pre-processing

Il testo viene processato attraverso questi step:

1. **Lowercasing**: Tutto in minuscolo
2. **Rimozione punteggiatura**: Sostituzione con spazi
3. **Lemmatizzazione**: Riduzione parole alla forma base usando Spacy
4. **Rimozione stopwords**: Eliminazione parole non significative (NLTK)
5. **Creazione bigrammi**: Suddivisione in sequenze di 2 caratteri
6. **Vectorization**: Bag of Words con CountVectorizer

### 2. Validazione

Lo script esegue due tipi di validazione:

**Hold Out Validation (70/30 split)**:
- Training set: 70%
- Test set: 30%
- Risultati: accuracy ~99% (possibile overfitting per dataset piccolo)

**Cross Validation (5-fold)**:
- Dataset diviso in 5 batch
- Training su 4 batch, test su 1 (rotazione)
- Risultati più affidabili: accuracy 93-97% a seconda del classificatore

### 3. Metriche di Performance

Per ogni validazione vengono calcolate:

- **Accuracy**: Percentuale predizioni corrette
- **Precision**: Qualità delle predizioni positive
- **Recall**: Capacità di identificare tutte le istanze positive
- **F1-score**: Media armonica di precision e recall
- **Confusion Matrix**: Visualizzazione errori di classificazione

## 📦 Output del Training

Il modello viene salvato in formato pickle con questa struttura:

```python
{
    "model": RandomForestClassifier(),      # Il classificatore addestrato
    "vectorizer": CountVectorizer(),        # Il vectorizer per preprocessing
    "label_encoder": LabelEncoder()         # Encoder per le lingue
}
```

**File di output**: `../models/museum_language_detector.pkl`

## 🎯 Performance Attese

### Con CountVectorizer

| Classificatore | Accuracy | Velocità | Consigliato |
|----------------|----------|----------|-------------|
| Multinomial NB | 93-94% | ⚡⚡⚡ | ✓ Produzione veloce |
| SVC | 93-94% | ⚡⚡⚡ | ✓ Produzione veloce |
| MLP | ~95% | ⚡ | - |
| Random Forest | ~97% | ⚡⚡ | ✓✓ Massima accuracy |

### Con TfidfVectorizer

| Classificatore | Accuracy | Note |
|----------------|----------|------|
| Multinomial NB | ~90% | Performance inferiore |
| SVC | ~97% | Migliora con TF-IDF |
| MLP | ~93% | Richiede tuning iperparametri |
| Random Forest | ~95% | Performance inferiore |

## ⚠️ Note Importanti

### Overfitting

Con un dataset così piccolo (294 frasi), c'è rischio di overfitting. I risultati di Hold Out Validation (~99%) sono probabilmente troppo ottimistici. I risultati di Cross Validation (93-97%) sono più realistici.

### Specializzazione

Il modello è **ottimizzato per contenuti museali**. Le performance potrebbero degradare su:
- Testi tecnici
- Conversazioni informali
- Linguaggio colloquiale
- Domini molto diversi (medicina, legge, sport, ecc.)

### Lingue supportate

Il modello riconosce **solo 3 lingue**:
- Italiano (it)
- Inglese (en)
- Tedesco (de)

Testi in altre lingue produrranno comunque una classificazione in una delle tre.

## 🔄 Riaddestramento

Per riaddestrare il modello con nuovi dati:

1. Prepara un CSV con colonne: `Testo`, `Codice Lingua`
2. Modifica `BASE_URL` in `train_model.py` con il path al tuo CSV
3. Esegui `python train_model.py`

## 📊 Esempio Output

```
Caricamento dataset...

Distribribuzione lingue nel dataset:
  it: 98
  en: 98
  de: 98

Pre-processing del testo...
Vectorization con CountVectorizer...

Bag of Words: 294 elementi
Bigrammi nel vectorizer: 1247

============================================================
TRAINING DEI MODELLI
============================================================

Classificatore selezionato: RandomForestClassifier

--- HOLD OUT VALIDATION ---

Regressione:
MSE: 0.0204
R2: 0.9796

Classificazione:
Accuracy con test size 0.3: 0.9898

              precision    recall  f1-score   support
          de       0.97      1.00      0.99        30
          en       1.00      1.00      1.00        29
          it       1.00      0.97      0.98        30
    accuracy                           0.99        89

[Visualizzazione Confusion Matrix]

--- CROSS VALIDATION ---

Regressione:
R2 per ogni batch: [0.95 0.96 0.98 0.97 0.96]
R2 medio: 0.9640

Classificazione:
Accuracy per batch: [0.95 0.97 0.98 0.97 0.95]
Precision per batch: [0.95 0.97 0.98 0.97 0.96]
Recall per batch: [0.95 0.97 0.98 0.97 0.95]
F1 per batch: [0.95 0.97 0.98 0.97 0.95]

              precision    recall  f1-score   support
          de       0.97      0.97      0.97        98
          en       0.98      0.97      0.97        98
          it       0.96      0.98      0.97        98
    accuracy                           0.97       294

[Visualizzazione Confusion Matrix]

============================================================
TRAINING FINALE SU DATASET COMPLETO
============================================================

✓ Modello salvato in: museum_language_detector.pkl
  - Classificatore: RandomForestClassifier
  - Lingue: ['de', 'en', 'it']
  - Features: 1247 bigrammi
```

## 🐛 Troubleshooting

### Errore: "Can't find model 'it_core_news_sm'"

**Soluzione**: Installa il modello Spacy mancante:
```bash
python -m spacy download it_core_news_sm
```

### Errore: "No module named 'nltk.corpus'"

**Soluzione**: Il download delle stopwords NLTK è fallito. Esegui manualmente:
```python
import nltk
nltk.download('stopwords')
```

### Performance inaspettatamente basse

**Possibili cause**:
- Dataset corrotto o modificato
- Modelli Spacy non installati correttamente
- Versione scikit-learn non compatibile (richiesta 1.6.0)

## 📚 Riferimenti

- [Documentazione Spacy](https://spacy.io/)
- [Documentazione scikit-learn](https://scikit-learn.org/)
- [NLTK Corpus](https://www.nltk.org/howto/corpus.html)

---

**Autore**: Murri-Hub  
**Ultima modifica**: Gennaio 2026
