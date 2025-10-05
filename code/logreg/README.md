We provide code for automatically computing and evaluating papers embeddings for the following text embedding models:
- **specter2_base:** Model Size 0.1B parameters, Dimension 768.
- **gte-base-en-v1.5:** Model Size 0.1B parameters, Dimension 768.
- **gte-large-en-v1.5:** Model Size 0.4B parameters, Dimension 1024.
- **Qwen3-Embedding-0.6B:** Model Size 0.6B parameters, Dimension 1024.
- **Qwen3-Embedding-4B:** Model Size 4B parameters, Dimension 2560.
- **Qwen3-Embedding-8B:** Model Size 7B parameters, Dimension 4096.

Here is an overview of their results (256-dimensional with 100-dimensional category embeddings attached):

## Session-based Evaluation (1070 Users, no causal Mask for Ranking Negatives)

| Model Name | Bal. Acc. | NDCG | MRR | InfoNCE |
|------------|------------------|------|-----|---------|
| tf-idf | 64.07 | 73.81 | 65.27 | 1.8260 |
| specter2_base | 71.74 | 80.54 | 74.09 | 1.1775 |
| gte-base-en-v1.5 | 70.98 | 79.80 |73.09 | 1.2266 |
| gte-large-en-v1.5 | 71.60 | 80.55 | 74.09 | 1.1924 |
| Qwen3-Embedding-0.6B | 71.59 | 80.55 | 74.09 | 1.1793 |
| Qwen3-Embedding-4B | 73.80 | 82.88 | 77.18 | 1.0680 |
| Qwen3-Embedding-8B | **74.07** | **83.20** | **77.59** | **1.0389** |

## Cross-Validation (1070 Users, Ranking Negatives sampled randomly)

| Model Name | Bal. Acc. | NDCG | MRR | InfoNCE |
|------------|------------------|------|-----|---------|
| tf-idf | 74.14 | 82.32 | 76.40 | 1.3369 |
| specter2_base | 77.30 | 84.80 | 79.69 | 0.9444 |
| gte-base-en-v1.5 | 77.58 | 84.97 | 79.91 | 0.9441 |
| gte-large-en-v1.5 | 78.07 | 85.32 | 80.38 | 0.9240 |
| Qwen3-Embedding-0.6B | 77.86 | 85.37 | 80.44 | 0.9150 |
| Qwen3-Embedding-4B | **79.42** | **86.85** | **82.41** | **0.8381** |
| Qwen3-Embedding-8B | 79.34 | 86.82 | 82.37 | 0.8382 |

## Session-based Evaluation NDCG (like above but only on the 500 Test Users)

| Model Name | Total | CS Users | Non-CS Users
|------------|------------------|------|-----|
| gte-large-en-v1.5 | 80.32 | 80.21 | 82.54 |
| Qwen3-Embedding-8B | 83.05 | 82.95 | **85.03** |
| gte-large-en-v1.5 fine-tune no cat loss | 83.21 | 83.27 | 82.02 |
| gte-large-en-v1.5 fine-tune cat loss | **83.33** | **83.28** | 84.20 |

## Cosine Similarity Changes between Physics and other Categories
| Category | before Fine-tuning | after Fine-tuning w/o Cat Loss | after Fine-tuning w/ Cat Loss |
|------------|------------------|------|-----|
| Computer Science | 20.44 | 16.24 | 17.21
| Medicine | 25.19 | 50.95 | 31.98
| Linguistics | 29.54 | 65.80 | 38.28
| Psychology | 33.09 | 58.73 | 37.81
| Biology | 39.36 | 65.00 | 41.08
| Astronomy | 39.52 | 72.81 | 45.42
| Physics | 57.18 | 86.96 | 64.53

## Sliding Window Evaluation NDCG (666 Users)
*HiVotes:* The 75 Users who have the largest number of Validation Upvotes

| Model Name | Total | First 25% Sess | Last 25% Sess | HiVotes Total | HiVotes First 25% Sess | HiVotes Last 25% Sess 
|------------|------------------|------|-----|-----|-----|-----|
| LogReg w/ Sliding Window | 79.83 | 79.39 | 81.39 | 82.26 | 81.70 | 83.27
| LogReg w/o Sliding Window | 77.27 | 78.63 | 77.46 | 77.18 | 78.93 | 76.40
| MaxPos w/ Sliding Window | 77.41 | 77.26 | 78.80 | 77.89 | 78.01 | 79.28
| MaxPos w/o Sliding Window | 75.61 | 76.70 | 75.79 | 76.88 | 76.32 | 74.02
| MeanPos w/ Sliding Window | 77.30 | 77.40 | 78.03 | 78.37 | 78.72 | 78.55
| MeanPos w/o Sliding Window | 75.97 | 76.93 | 75.94 | 76.07 | 77.25 | 75.59
| TF-IDF w/ Sliding Window | 73.19 | 72.86 | 74.51 | 75.76 | 76.42 | 76.80
| TF-IDF w/o Sliding Window | 70.27 | 72.03 | 70.03 | 69.86 | 72.95 | 68.94


## Temporal Decay Evaluation NDCG (666 Users)
*HiTime:* The 75 Users who have the largest time span between Upvotes in their Validation Set
| Model Name | Total | First 25% Time | Last 25% Time | HiTime Total | HiTime First 25% Sess | HiTime Last 25% Sess 
|------------|------------------|------|-----|-----|-----|-----|
| No Decay | 79.83 | 79.52 | 81.30 | 78.20 | 77.48 | 79.83
| No Decay, 200 Days Max | 79.91 | 79.54 | 81.30 | 78.77 | 77.53 | 79.89
| Exponential Decay, Joint Normalization, Param 0.01 | 80.10 | 79.54 | 81.73 | 78.93 | 77.68 | 80.85

## Neural Network Evaluation (150 Test Users)
| Model Name | Total | First 25% Time | Last 25% Time |
| LogReg | 79.40 | 80.76 | 79.83
| LogReg after fine-tuning | 82.33 | 83.62 | 83.21
| LogReg Exponential Decay | 79.74 | 80.94 | 80.81
| LogReg Exponential Decay after fine-tuning | 82.53 | 83.52 | 83.91
| MeanPos | 76.74 | 79.51 | 76.88
| MeanPos after fine-tuning | 79.50 | 80.75 | 79.38
| Neural after fine-tuning | 78.80 | 81.48 | 80.00
| Neural after fine-tuning | 81.27 | 82.74 | 81.67




# II. Experiment Setup
To evaluate the recommender performance with specialized settings, create an examplary config file by running (from the root directory):
```bash
python -m code.scripts.create_example_configs
```
You may then inspect and alter the settings in `code/logreg/experiments/example_configs/example_config.json`.
Here is an overview of the possible settings:
#### 1) Users Selection & Evaluation
- **evaluation:** String default: "cross_validation".
The default "cross_validation" performs a random shuffle of the user voting data with 5-fold cross-validation. Other options are "train_test_split" (another random shuffle but only for a single split) and "session_based" (a temporal split so that training is done on the oldest voting data and evaluation on the most recent ratings).


#### 2) Miscellanous
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
-- **n_jobs:** integer default: 1.
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
Note that these visualizations require a lot of time and are therefore not done automatically for all users. 
You will need to set `save_users_coefs`to True during training.

 Examples are shown below:

### Hyperparameter Ablation Study
<img src="images/ablation_study.png" width="1000">

### Global Performance Study Overview
<img src="images/global_visu_bal-04.png" width="1000">
<img src="images/global_visu_bal-05.png" width="1000">

### Single User Performance Study Overview
<img src="images/word_cloud.png" width="1000">
<img src="images/false_positive.png" width="1000">

