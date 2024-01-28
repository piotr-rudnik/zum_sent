


## Zbiór danych
W projekcie został użyty zbiór danych z Kaggle - ChatGPT sentiment analysis
https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis/

Zbiór danych zawiera ok. 217 tysięcy rekordów. 
Każdy tweet reprezentuje jedną opinię na temat ChatGPT, która została sklasyfikowana jako pozytywna, neutralna lub negatywna.

Podział klas:
- 49% - bad 
- 25% - neutral
- 26% - good

Dane zawarte są w pliku `data.csv` w folderze głównym

## Notebook 
Cała treść kodu i analizy znajduje się w pliku NLP_praca_domowa.ipynb

## Modele 
1. LSTM - lstm.pth - model LSTM z warstwą embedding - wynik ok. 78%
2. CNN - cnn.pth - wynik ok. 74.5%
3. Glove 50d - glovo.pth - predefiniowany embedding - 76%
4. Roberta (https://huggingface.co/docs/transformers/model_doc/roberta) - 89%
    
!! UWAGA - model Roberta nie jest w repozytorium, ponieważ jest zbyt duży, okolice 500mb,
co przekracza granice githuba. Model można wyuczyć ponownie, jednak jest to kilka godzin działania GPU.

Pierwsze 3 modele znajdują się w folderze `models`.


## Użycie 
W projekcie jest zawarty plik glovo50d.zip który posiada zkompresowany embedding Glove 50d. Należy go rozpakować w folderze głównym.
```
unzip glovo50d.zip
```

Do zarządzania zależnościami użyte zostało narzędzie poetry (https://python-poetry.org/)

Aby zainstalować zależności należy wykonać polecenie:
```bash 
poetry install
```

Ze względu na problem z zależnościami i CUDA przy uczeniu modelu Roberta, wersja pytorch była zdowngredowana do 2.0.0
reszta modeli korzysta z 2.0.1

Aby użyc modelu należy uruchomić plik `main.py`:
```poetry run python3 main.py```
I następnie podać zdanie:
```
INPUT:  chatgpt is a bad negative tool wasted time
negative
good
bad
```
