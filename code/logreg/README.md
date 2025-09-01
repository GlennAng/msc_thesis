We provide code for automatically computing and evaluating papers embeddings for the following text embedding models:
- **specter2_base:** Model Size 0.1B parameters, Dimension 768.
- **gte-base-en-v1.5:** Model Size 0.1B parameters, Dimension 768.
- **gte-large-en-v1.5:** Model Size 0.4B parameters, Dimension 1024.
- **Qwen3-Embedding-0.6B:** Model Size 0.6B parameters, Dimension 1024.
- **Qwen3-Embedding-4B:** Model Size 4B parameters, Dimension 2560.
- **Qwen3-Embedding-8B:** Model Size 7B parameters, Dimension 4096.

Here is an overview of their results (256-dimensional with 100-dimensional category embeddings attached):

## Session-based Evaluation (no filter)

| Model Name | Bal. Acc. | NDCG | MRR | InfoNCE |
|------------|------------------|------|-----|---------|
| tf-idf | 64.07 | 73.81 | 65.27 | 1.8260 |
| specter2_base | 71.74 | 80.54 | 74.09 | 1.1775 |
| gte-base-en-v1.5 | 70.98 | 79.80 |73.09 | 1.2266 |
| gte-large-en-v1.5 | 71.60 | 80.55 | 74.09 | 1.1924 |
| Qwen3-Embedding-0.6B | 71.59 | 80.55 | 74.09 | 1.1793 |
| Qwen3-Embedding-4B | 73.80 | 82.88 | 77.18 | 1.0680 |
| Qwen3-Embedding-8B | **74.07** | **83.20** | **77.59** | **1.0389** |

## Cross-Validation

| Model Name | Bal. Acc. | NDCG | MRR | InfoNCE |
|------------|------------------|------|-----|---------|
| tf-idf | 74.14 | 82.32 | 76.40 | 1.3369 |
| specter2_base | 77.30 | 84.80 | 79.69 | 0.9444 |
| gte-base-en-v1.5 | 77.58 | 84.97 | 79.91 | 0.9441 |
| gte-large-en-v1.5 | 78.07 | 85.32 | 80.38 | 0.9240 |
| Qwen3-Embedding-0.6B | 77.86 | 85.37 | 80.44 | 0.9150 |
| Qwen3-Embedding-4B | **79.42** | **86.85** | **82.41** | **0.8381** |
| Qwen3-Embedding-8B | 79.34 | 86.82 | 82.37 | 0.8382 |


## Fine-tuning Session-based Evaluation NDCG (still no filter but on Test Users only)

| Model Name | Total | CS Users | Non-CS Users
|------------|------------------|------|-----|
| gte-large-en-v1.5 | 80.32 | 80.21 | 82.54 |
| Qwen3-Embedding-8B | 83.05 | 82.95 | **85.03** |
| gte-large-en-v1.5 fine-tune no cat loss | 83.13 | 83.16 | 82.55 |
| gte-large-en-v1.5 fine-tune cat loss | **83.23** | **83.19** | 83.99 |

## Cosine Similarity Changes between Physics and other Categories
| Category | before Fine-tuning | after Fine-tuning w/o Cat Loss | after Fine-tuning w/ Cat Loss |
|------------|------------------|------|-----|
| Computer Science | 20.44 | 13.28 | 14.54
| Medicine | 25.19 | 50.09 | 31.53
| Linguistics | 29.54 | 65.38 | 36.56
| Psychology | 33.09 | 57.94 | 36.61
| Biology | 39.36 | 63.51 | 39.47
| Astronomy | 39.52 | 71.77 | 43.53
| Physics | 57.18 | 85.96 | 64.07

## Session-based Evaluation NDCG (filtering, predicting whole Validation Set at once vs. Sliding Window)
*First/Last Sess:* The first/last Validation Session with at least one Upvote in it  
*HiSess:* The 100 Users who have the largest number of Validation Sessions with at least one Upvote in it

| Model Name | Total | First Sess | Last Sess | HiSess Total | HiSess First Sess | HiSess Last Sess 
|------------|------------------|------|-----|-----|-----|-----|
| LogReg w/o Sliding Window | 80.39 | 80.82 | 80.29 | 82.27 | 84.57 | 81.69
| LogReg w/ Sliding Window | 81.02 | 80.83 | 81.67 | 83.33 | 84.79 | 84.04
| MeanPos w/o Sliding Window | 77.46 | 78.19 | 77.39 | 77.28 | 77.34 | 75.88
| MeanPos w/ Sliding Window | 77.77 | 78.19 | 78.02 | 77.97 | 77.34 | 77.81


## LogReg Sliding Window NDCG (drop ratings which are too old (but at least 10 Train Positives))
We are looking for justification to use the entire context, otherwise one can reduce training time at same performance.  

| Max. Number of Days | Total | First Sess | Last Sess | HiSess Total | HiSess First Sess | HiSess Last Sess 
|------------|------------------|------|-----|-----|-----|-----|
| 30 | 80.89 | 80.69 | 81.65 | 82.64 | 83.49 | 82.56
| 60 | 80.98 | 80.57 | 81.83 | 83.13 | 83.15 | 83.96
| 100 | 81.10 | 80.67 | **82.08** | 83.58 | 84.31 | **84.85**
| 250 | **81.19** | 80.90 | 82.00 | **83.72** | 84.77 | 84.72
| 500 | 81.13 | **80.91** | 81.80 | 83.48 | **85.02** | 84.31
| Infinity | 81.02 | 80.83 | 81.67 | 83.33 | 84.79 | 84.04

## MeanPos Sliding Window NDCG (drop ratings which are too old (but at least 1 Train Positive))
| Max. Number of Sessions | Total | First Sess | Last Sess
|------------|------------------|------|-----
| 1 | 74.62 | 76.45 | 74.19
| 3 | 78.09 | 78.42 | 78.47
| 5 | 78.29 | **78.49** | **79.16**
| 10 | **78.30** | 78.44 | 78.73
| 20 | 78.13 | 78.40 | 78.64
| 50 | 77.77 | 78.18 | 78.07
| Infinity | 77.77 | 78.19 | 78.02

## Strong Users Group: The 100 Users with the largest Sliding Validation Positive Session Cosine Similarities
- NDCG: 86.08 (Average 81.02)
- NDCG First Val Session: 87.34 (Average 80.83)
- NDCG Last Val Session: 85.51 (Average 81.67)
- Training Positives: 51.8 (Average 78.3)
- Validation Positive Sessions: 5.1 (Average 6.0)
- Validation Positive Days: 55.8 (Average 50.1)
- Train Cosine Sliding Window: 46.8 (Average 34.31)
- Validation Cosine Sliding Window: 53.29 (Average 34.27)

## High Sessions vs High Votes
HiSess: 19.7 Val Pos Sessions, 32.9 Val Pos Ratings, 138.4 Val Pos Days, 33.67 Val Cosine  
HiVotes: 6.9 Val Pos Sessions, 58.4 Val Pos Ratings, 50.1 Val Pos Days, 31.41 Val Cosine

MeanPos:  
HiSess (NDCG): Total 80.28, First S 79.57, Last S 82.53, Random S 82.15  
HiVotes (NDCG): Total 78.29, First S 83.08, Last S 79.96, Random S 79.27

LogReg:  
HiSess (NDCG): Total 83.39, First S 83.48, Last S 84.17, Random S: 86.62  
HiVotes (NDCG): Total 81.02, First S 84.24, Last S 83.09, Random S: 83.21


# II. Experiment Setup
To evaluate the recommender performance with specialized settings, create an examplary config file by running (from the root directory):
```bash
python -m code.scripts.create_example_configs
```
You may then inspect and alter the settings in `code/logreg/experiments/example_configs/example_config.json`.
Here is an overview of the possible settings:
#### 1) Users Selection & Evaluation
- **users_selection:** String default: "finetuning_test".
Select a subset of users on which the experiments are performed. The default "finetuning_test" automatically chooses 500 fixed users with sufficiently many votes. Other options are "finetuning_val", "random" or explicitly entering a list of IDs such as [9, 13, 21]. 
But if `--all_papers` was NOT set while computing the paper embeddings, this value has to remain at "finetuning_test" (other users might have voted on papers for which no embeddings were computed).
- **take_complement_of_users:** Bool default: False.
Among all users meeting the requirements (see below), select exactly those that were NOT specified.
- **evaluation:** String default: "cross_validation".
The default "cross_validation" performs a random shuffle of the user voting data with 5-fold cross-validation. Other options are "train_test_split" (another random shuffle but only for a single split) and "session_based" (a temporal split so that training is done on the oldest voting data and evaluation on the most recent ratings).
- **test_size:** Float default: 0.2.
Percentage of user votes selected for validation. When selecting evaluation: "session_based", the algorithm performs the split at the session which brings the validation share closest to this value.
- **stratified:** Bool default: True.
Decides whether training and validation set should have the same positive/negative ratio when selection evaluation: "cross_validation" or "train_test_split". Without it, there is a risk of users not having enough positive or negative labels and being skipped (for both training and evaluation).

Next are multiple options used to specify minimum requirements for users to be considered when choosing users_selection: "random" (otherwise ignored).
- **max_users:** Integer default: 500.
An upper bound for the number of users to be selected.
- **min_n_posrated / min_n_negrated:** Integer default: 20.
These refer to the required numbers of upvotes and downvotes when specifying evaluation: "cross_validation" or "train_test_split". Ignored for "session_based".
- **min_posrated_train / min_n_negrated_train:** Integer default: 16.
These refer to the required numbers of upvotes and downvotes in the training set when entering evaluation: "session_based". Since temporal splits are based on sessions, this is needed to avoid having users for which all ratings were performed during just a single session (so the training or validation set would end up too small).
- **min_posrated_val / min_n_negrated_val:** Integer default: 5.
Analogously but refering to the sample size in the validation split.

#### 2) Miscellanous
- **users_random_state:** Integer default: 42.
The seed when randomly drawing a subset of users by specifying users_selection: "random". Ignored otherwise.
- **model_random_state:** Integer default: 42.
The seed passed to sklearn's LogisticRegression. It is also used for generating train/test splits (for a single split and during cross-validation). If `--all_papers`was NOT set while computing embeddings, this value has to lie in the list
[1, 2, 25, 26, 42, 75, 76, 100, 101, 150, 151].
- **cache_random_state:** Integer default: 42.
The seed used to randomly select a subset of paper IDs from the total 100K cache for training negatives. Same restrictions as for model_random_state.
- **ranking_random_state:** Integer default: 42.
The seed used to randomly select 4 actual negatives and additional simple negatives to evaluate ranking performance. The simple negatives are drawn from research categories untypical for the Scholar Inbox users. Same restrictions as for model_random_state.
- **save_users_predictions:** Bool default: false.
When set to True, the program will store all the model predictions made during the run in .json files. This is generally not necessary and costs additional storage, but enables more detailed visualizations.
- **save_users_coefs:** Bool default: false.
When set to True, the program will store all the trained logistic regression coefficients in .npy files. This is generally not necessary and costs additional storage, but enables using them as a starting point for further processing.
- **load_users_coefs:** Bool default: false.
When set to True, the program will skip training and instead use the coefficients provided under **users_coefs_path** (String default: null) for evaluation.
-- **n_jobs:** integer default: -1.
The number of users to be processed in parallel. By default, it takes all cores.

#### 3) Hyperparameters
For the following settings, you can enter single values or lists. In case of lists, all possible combinations of hyperparameters will be tried.
- **clf_C:** Float default: 0.1.
The inverse regularization strength. A strong value for TF-IDF is 0.4.
- **weights_cache_v:** Float default: 0.9.
The importance of the actual downvotes compared to the random cache negatives. A strong value for TF-IDF is 0.9.
- **weights_neg_scale:** Float default: 5.0.
A scaling factor applied to all negative samples. A strong value for TF-IDF is 1.0.

#### 4) Embedding
Lastly, specify the path to the folder containing the embeddings matrix under **embedding_folder**.

# III. Running Experiments & Visualization
To run the experiments, simply specify the path to the config file (again all from the root directory)
```bash
python -m code.scripts.logreg_run --config_path code/logreg/experiments/example_configs/example_config.json
```
Alternatively, you may automatically run the same experiments multiple times over different random seeds and average the results. For this case, specify one specific config file and the program will then automatically copy it and change the seeds.
```bash
python -m code.scripts.logreg.logreg_average_seeds --config_path code/logreg/experiments/example_configs/example_config.json
```
This will create a pdf-file containing detailed results under `code/logreg/outputs/example_config/global_visu_bal.pdf`.
If you want more detailed visualizations for individual users, you may run (but only possible if you specified save_users_predictions: true beforehand):

```bash
python -m code.logreg.src.visualization.visualize_users --outputs_folder code/logreg/outputs/example_config --users 46 110 6792
```
Note that these visualizations require a lot of time and are therefore not done automatically for all users. Examples are shown below:


### Hyperparameter Ablation Study
<img src="images/ablation_study.png" width="1000">

### Global Performance Study Overview
<img src="images/global_visu_bal-04.png" width="1000">
<img src="images/global_visu_bal-05.png" width="1000">

### Single User Performance Study Overview
<img src="images/word_cloud.png" width="1000">
<img src="images/false_positive.png" width="1000">

