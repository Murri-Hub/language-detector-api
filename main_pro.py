"""
Il codice è stato implementato prevedendo l'uso di FastAPI, teoricamente più portato per deploy di
modelli di ML in quanto più prestazionale. Inoltre è più adatto per l'uso del formato JSON.

Nel mio caso specifico ho dovuto installare manualmente la versione 1.6.0 di scikit-learn per
l'uso del file pickle fornito e ho dovuto ripulire gli handler in modo da ottenere la formattazione
dei log voluta con il comando basicConfig.

Successivamente è stato definito il profiler con annessi parametri ed infine definito l'endpoint.

All'interno dell'endpoint è stata definita una funzione asincrona che per prima cosa verifica
che il formato in input sia un JSON, restituendo un messaggio di formato non valido e registrando
l'eccezione nel log. Volendo si può togliere il cancelletto dalla riga dedicata per avere anche
il traceback, ma in questo caso l'ho ritenuto più pesante del dovuto.

A questo punto il testo in input viene inserito in una lista di stringhe pronta per essere classificata
dalla pipeline estratta dal file pickle.

Se la lista è vuota, viene registrato un errore nel log e si restituisce in output un messaggio
di "testo mancante".

Le righe di codice successive definiscono gli item necessari per ottenere l'output come da consegna
del progetto.

Se la probabilità di previsione corretta è sotto una certa soglia, che arbitrariamente ho stabilito
a 0.75, viene restituito un messaggio di "lingua non rilevabile" e registrato a log un errore.

Il codice di tutti gli errori è il generico 400.

Visto l'uso di FastAPI, l'applicazione viene lanciata tramite server Uvicorn.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi_profiler import PyInstrumentProfilerMiddleware
from pydantic import BaseModel
from typing import Union, List
import pickle
import uvicorn
import logging
import json

# Carica il file
# Per l'uso del file pickle è stato necessario utilizzare la versione 1.6.0 di scikit-learn
filename = 'language_detector.pkl'
loaded_pipeline = pickle.load(open(filename, 'rb'))

class TextInput(BaseModel):
    text: Union[str, List[str]]

# Prima dell'uso di basicConfig è stato necessario ripulire gli handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Formattazione dei log con nome del livello di eccezione, nome file di provenienza, data e ora, messaggio
logging.basicConfig(format='[%(levelname)s]%(name)s - %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H-%M-%S',
                    level=logging.INFO)    #Visualizzare messaggi di log di livello INFO e superiori

# Aggancia al file "main_pro" il messaggio di log per tracciare da dove è stato generato
logger = logging.getLogger(__name__)

# Utilizzo di FastAPI
app = FastAPI()

# Profiler
app.add_middleware(PyInstrumentProfilerMiddleware,
                   server_app = app,                                # Salva i log una volta fermata l'app
                   profiler_output_type = "html",                   # Salvataggio info in html
                   is_print_each_request = False,                   # Per limitare il contenuto dei log
                   open_in_browser = False,                         # Per evitare che apra il file al termine
                   html_file_name = "profiling_performance.html")   # Dove salva il file

# Definizione endpoint di tipo POST
@app.post('/identify-language')
async def string_in(payload: TextInput):   # Funzione asincrona con FastAPI

    # Verifica ingresso in formato JSON
    try:
        data = await request.json()  # data è un dizionario python con il JSON inviato

    # Se l'input non è in formato JSON, restituisce e logga un errore 400
    except json.JSONDecodeError:
        logger.error("Body non JSON")               # Log senza traceback
        #logger.exception("Body non JSON")           # Log dell'eccezione con traceback
        raise HTTPException(                        # Raise dell'eccezione specifica a FastAPI
            status_code=400,
            detail="Formato non valido"
        )

    # Estraiamo il testo dal JSON con chiave "text"
    text_list = payload.text


    # In caso di testo assente, restituisce e logga un errore 400
    if not text_list:
        logger.error(f"Nessun testo rilevato")      # Log dell'errore
        raise HTTPException(
            status_code=400,
            detail="Nessun testo rilevato"
        )

    # In caso di stringa singola, si inserisce il testo estratto nel formato lista atteso
    if isinstance(text_list, str):
        text_list = [text_list]

    # Previsione linguaggio
    predicted_language = loaded_pipeline.predict(text_list)

    # Item con le probabilità associate alle lingue
    probabilities = loaded_pipeline.predict_proba(text_list)

    # Item che contiene le classi nell'ordine corrispondente alle colonne di "probabilities"
    classes = loaded_pipeline.named_steps['multinomialnb'].classes_

    # Item con l’indice della classe predetta
    predicted_index = list(classes).index(predicted_language[0])

    # Item con la probabilità corrispondente
    predicted_prob = probabilities[0][predicted_index]

    CONFIDENCE_THRESHOLD = 0.75

    #Se la probabilità è bassa, restituisce e logga un errore
    if predicted_prob <= 0.75:
        logger.error(f"Item predicted_prob = {predicted_prob} < CONFIDENCE_THRESHOLD")  # Log dell'errore
        raise HTTPException(
        status_code=400,
        detail="Lingua non identificabile"
        )

    # Restituisce in output la previsione di linguaggio e la probabilità
    return {
    "language_code": predicted_language[0],
    "confidence": round(predicted_prob, 2)
    }

if __name__ == '__main__':
    # Run dell'app tramite server ASGI Uvicorn
    uvicorn.run('main_pro:app')
