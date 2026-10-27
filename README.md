# CinePredict AI

CinePredict predicts a movie genre from a plot summary using a TF-IDF and
Logistic Regression pipeline.

## Requirements

- Python 3
- The packages listed in `requirements.txt`
- The training dataset at `dataset/wiki_movie_plots_deduped.csv`

The CSV must include `Plot` and `Genre` columns. The training script uses the
10 most common valid genres in the dataset.

## Setup

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the model

With the dataset in place, run:

```powershell
python train_model.py
```

This evaluates the model and saves the trained pipeline as
`movie_genre_model.pkl` in the project directory. The Streamlit app and
command-line predictor load this file.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Enter a plot summary or choose an example to see the predicted genre and the
model's top three genre scores.

## Run the command-line predictor

```powershell
python predict.py
```

Enter a plot summary when prompted. Type `exit` to quit.
