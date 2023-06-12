# Introduction

This repository implements models from the following two papers:

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**  

> **Variational Autoencoders for Collaborative Filtering (Liang et al.)**  

and lets you train them on MovieLens-1m and MovieLens-20m.

# Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

On running `main.py`, it asks you whether to train on MovieLens-1m or MovieLens-20m. (Enter 1 or 20)

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## BERT4Rec

```bash
python main.py --template train_bert
```

## Examples

1. Train BERT4Rec on ML-20m and run test set inference after training

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```
