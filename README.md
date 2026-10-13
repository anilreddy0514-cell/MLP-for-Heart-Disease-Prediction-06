# Heart Disease Prediction with MLPs

This project compares two Multi-Layer Perceptron (MLP) classifiers on the Heart Disease dataset. The goal is to evaluate how different hidden-layer configurations perform and select the stronger architecture using test-set metrics.

## Project Files

- `heart.csv`: Heart Disease dataset. The `target` column is the prediction target; the remaining columns are model inputs.
- `main.ipynb`: Data preparation, model training, metric comparison, and conclusion.
- `problemstatement.txt`: Assignment requirements.

## Method

The notebook removes duplicate records, then creates a stratified 80/20 train/test split using a fixed random seed. Each model is trained in a pipeline that standardizes features before fitting the MLP, keeping scaling fit to the training data. Both networks use the same split and training settings:

- MLP with hidden layers `(32, 16)`
- MLP with hidden layers `(64, 32, 16)`

The models are compared by accuracy, precision, recall, and F1 score. The notebook ranks them by F1 score, using accuracy as a tie-breaker.

## Run

Open and run `main.ipynb` in Jupyter or VS Code. The notebook uses Python with pandas and scikit-learn; install them if needed:

```bash
python -m pip install pandas scikit-learn
```

## Current Result

In the notebook's recorded run, the `(32, 16)` MLP ranked first with accuracy `0.754`, precision `0.765`, recall `0.788`, and F1 score `0.776`. This conclusion reflects the fixed split and settings in the notebook; rerunning with different data or settings may change the result.