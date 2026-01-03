# Language Detection API

API REST per l'identificazione automatica della lingua di un testo
basata su modelli di Machine Learning.

## Tecnologie
- FastAPI
- scikit-learn
- Multinomial Naive Bayes
- Bag of Words + n-grams
- Uvicorn ASGI server

## Endpoint

### POST /identify-language

**Request**
```json
{
  "text": "Questo è un testo di esempio"
}
