# Language Detector API

API REST per il riconoscimento automatico della lingua di testi, implementata con FastAPI.

## 📋 Descrizione

Questa API identifica automaticamente la lingua di testi in input utilizzando un modello di Machine Learning pre-addestrato. L'implementazione utilizza FastAPI per garantire alte performance e gestione nativa del formato JSON.

## ✨ Caratteristiche

- **Framework**: FastAPI (ottimizzato per API ML)
- **Server**: Uvicorn (ASGI server ad alte prestazioni)
- **Formato I/O**: JSON
- **Logging**: Sistema completo per debugging e monitoraggio
- **Profiling**: Analisi performance tramite PyInstrument
- **Validazione**: Pydantic per validazione automatica input
- **Soglia confidenza**: Risposta solo con probabilità > 75%

## 🚀 Installazione

### Prerequisiti

- Python 3.8 o superiore
- pip (package manager Python)

### Setup

1. **Clona la repository:**
```bash
git clone https://github.com/Murri-Hub/language-detector-api.git
cd language-detector-api
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

3. **Installa scikit-learn**:
```bash
pip install scikit-learn
```

### Verifica installazione

```bash
python -c "import fastapi, uvicorn, sklearn; print('✓ Tutto installato correttamente')"
```

## 📦 File richiesti

Assicurati di avere questi file nella cartella principale:

```
language-detector-api/
├── main_pro.py                  # API principale (questo file)
├── language_detector.pkl        # Modello pre-addestrato
└── requirements.txt             # Dipendenze Python
```

**IMPORTANTE**: Il file `language_detector.pkl` deve essere presente nella stessa cartella di `main_pro.py`.

## 🎮 Utilizzo

### Avvio del server

```bash
python main_pro.py
```

Il server si avvierà su `http://localhost:8000`

Vedrai un messaggio simile a:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Endpoint disponibile

**POST** `/identify-language`

Identifica la lingua di uno o più testi.

## 📝 Esempio di utilizzo

### Con Python (requests)

```python
import requests

url = "http://localhost:8000/identify-language"

# Esempio 1: Stringa singola
payload = {"text": "Questo è un testo in italiano"}
response = requests.post(url, json=payload)
print(response.json())
# Output: {"language_code": "it", "confidence": 0.98}

# Esempio 2: Lista di stringhe
payload = {"text": ["Welcome to the museum", "Bienvenue au musée"]}
response = requests.post(url, json=payload)
print(response.json())
```

## 📤 Formato Request

### Schema JSON

```json
{
  "text": "stringa o array di stringhe"
}
```

### Esempi validi

```json
{"text": "Hello world"}
```

```json
{"text": ["Prima frase", "Seconda frase"]}
```

## 📥 Formato Response

### Successo (200 OK)

```json
{
  "language_code": "it",
  "confidence": 0.95
}
```

**Campi:**
- `language_code` (string): Codice ISO della lingua identificata (es. "it", "en", "de")
- `confidence` (float): Confidenza della predizione (0-1, arrotondata a 2 decimali)

### Errori (400 Bad Request)

**Formato non valido:**
```json
{
  "detail": "Formato non valido"
}
```

**Testo mancante:**
```json
{
  "detail": "Nessun testo rilevato"
}
```

**Bassa confidenza:**
```json
{
  "detail": "Lingua non identificabile"
}
```

## ⚙️ Configurazione

### Soglia di confidenza

La soglia minima è impostata a **0.75** (75%). Per modificarla, edita nel file `main_pro.py`:

```python
CONFIDENCE_THRESHOLD = 0.75  # Modifica questo valore
```

### Porta del server

Per cambiare la porta, modifica l'ultima riga:

```python
uvicorn.run('main_pro:app', host='0.0.0.0', port=8080)  # Porta 8080
```

### Livello di logging

Per modificare il livello di logging, cambia:

```python
level=logging.DEBUG  # Per logging più dettagliato
level=logging.WARNING  # Per meno log
```

## 📊 Logging

I log vengono stampati su console con il formato:
```
[LEVEL]nome_file - YYYY-MM-DD HH-MM-SS - messaggio
```

**Esempi:**
```
[INFO]__main__ - 2025-01-03 14-30-15 - Input ricevuto: 1 stringa
[ERROR]__main__ - 2025-01-03 14-30-20 - Nessun testo rilevato
```

## 📈 Profiling delle performance

Al termine dell'esecuzione (quando fermi il server con `Ctrl+C`), viene generato il file:

```
profiling_performance.html
```

Aprilo con un browser per visualizzare:
- Tempo di esecuzione per ogni funzione
- Bottleneck delle performance
- Call stack completo

## 🐛 Troubleshooting

### Errore: "No module named 'fastapi'"

**Soluzione:**
```bash
pip install fastapi uvicorn pydantic
```

### Errore: "File not found: language_detector.pkl"

**Causa**: Il file del modello non è presente nella cartella.

**Soluzione**: 
- Se hai il file pickle, spostalo nella stessa cartella di `main_pro.py`
- Se devi generarlo, esegui lo script di training nella cartella `Training/`

### Il server non si avvia

**Verifica:**
1. La porta 8000 non sia già occupata
2. Tutti i file necessari siano presenti
3. Le dipendenze siano installate correttamente

## 🔒 Note di sicurezza

- L'API non implementa autenticazione (aggiungerla per produzione)
- Non ci sono limiti di rate limiting (implementarli per uso pubblico)
- Il server è configurato per localhost (cambia `host` per esposizione esterna)

## 🛠️ Tecnologie utilizzate

| Libreria | Versione | Scopo |
|----------|----------|-------|
| FastAPI | latest | Framework web |
| Uvicorn | latest | Server ASGI |
| Pydantic | latest | Validazione dati |
| scikit-learn | latest | Machine Learning |
| PyInstrument | latest | Profiling |

## 📚 Documentazione API interattiva

FastAPI genera automaticamente documentazione interattiva:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Aprile nel browser per testare l'API direttamente dall'interfaccia!

## 🔄 Aggiornamento del modello

Per usare un nuovo modello:

1. Addestra il modello con lo script in `Training/`
2. Sostituisci `language_detector.pkl` con il nuovo file
3. Riavvia il server

## 👤 Autore

**Murri-Hub**
- GitHub: [@Murri-Hub](https://github.com/Murri-Hub)

---

**Nota tecnica**: Questo codice utilizza una pipeline pre-addestrata salvata in formato pickle. La versione specifica di scikit-learn (1.6.0) è richiesta per garantire la compatibilità del file pickle.
