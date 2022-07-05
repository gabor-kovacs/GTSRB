# GTSRB assignment

96.5% accuracy on test data using CNN in pytorch

## Train

```sh
python main.py
```

## Test using best model

```sh
python test.py
```

## Structure

```sh
📦GTSRB
 ┣ 📂data
 ┃ ┣ 📂Meta
 ┃ ┣ 📂Test
 ┃ ┣ 📂Train
 ┃ ┣ 📜Meta.csv
 ┃ ┣ 📜Test.csv
 ┃ ┗ 📜Train.csv
 ┣ 📂out
 ┃ ┣ 📜confusion_matrix.png
 ┃ ┗ 📜model_best.pth
 ┣ 📜.DS_Store
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜dataset.py
 ┣ 📜loss.py
 ┣ 📜main.py
 ┣ 📜model.py
 ┣ 📜notes.txt
 ┣ 📜requirements.txt
 ┣ 📜signnames.csv
 ┗ 📜test.py
```
