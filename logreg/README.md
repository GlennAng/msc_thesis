# I. Embedding Computation
In order to run our experiments, you need a directory that contains the two files `abs_X.npy` and `abs_paper_ids_to_idx.pkl`. 
The first is a numpy matrix of shape (n_papers, embedding_dim) containing the text embedding of each paper in the database whereas the second is a dictionary mapping the respective paper ids to their index in this matrix. Note that for most experiments, it is sufficient to only embed the papers stored in `relevant_papers_ids.pkl`.

You may use your own embeddings or compute them for one of the provided models with the code below:

- **gte-base-en-v1.5:** Model Size 0.1B parameters, Dimension 768.
- **gte-large-en-v1.5:** Model Size 0.4B parameters, Dimension 1024.
- **specter2_base:** Model Size 0.1B parameters, Dimension 768.
- **Qwen3-Embedding-0.6B:** Model Size 0.6B parameters, Dimension 1024.
- **Qwen3-Embedding-4B:** Model Size 4B parameters, Dimension 2560.
- **Qwen3-Embedding-8B:** Model Size 7B parameters, Dimension 4096.

For example, then run the following to compute embeddings only for the relevant papers.
Instead, you could compute embeddings for the entire database by further appending `--all_papers`:
```bash
python run_embed.py --model_name gte-large-en-v1.5 --batch_size 500
```
Note that the outputs will be stored in the directory `embeddings/before_pca/gte_large`.
### PCA Dimensionality Reduction
Empirical results have shown strong downstream performance even after vastly lowering the embedding dimensionality via PCA. To perform this step, run:
```bash
python src/embeddings/apply_pca.py --embeddings_input_folder embeddings/before_pca/gte_large --pca_dim 256
```
### Attaching Scientific Category Embeddings
In order to attach GloVe word embeddings for the research categories (dimension from [50, 100, 200, 300]), run:
```bash
python src/embeddings/papers_categories.py --embeddings_input_folder embeddings/after_pca/gte_large_256 --dim 100
```
This will require you to have the GloVe word embeddings downloaded into `embeddings/glove/glove.6B.100d.txt`.

# II. Experiment Setup
To evaluate the recommender performance, create an examplary config file by running:
```bash
python create_example_configs.py
```
You may then inspect and alter the settings in `experiments/example_configs/example_config.json`. 
Here is an overview of the possible settings:
#### 1) Users Selection
- **users_selection:** String default: "500_temporal".
Select a subset of users on which the experiments are performed. The default "500_temporal" automatically chooses 500 fixed users with sufficiently many votes. Other options are "random" or explicitly entering a list of IDs such as [9, 13, 21]. 
But if `--all_papers` was NOT set while computing the paper embeddings, this value has to remain at "500_temporal" (other users might have voted on papers for which no embeddings were computed).
- **max_users:** Integer default: null.
An upper bound for the number of users to be selected when entering users_selection: "random". Has to be null otherwise.

Next are multiple options to set minimum requirements for users to be considered during the random selection. When explicitly entering users whose requirements are not met, an error will occur.
- **min_n_posrated / min_n_negrated:** Integer default: 20.
These refer to the required numbers of upvotes and downvotes when entering evaluation: "cross_validation" or "train_test_split". They should be null for "session_based".
- **min_posrated_train / min_n_negrated_train:** Integer default: 16.
These refer to the required numbers of upvotes and downvotes in the training set when entering evaluation: "session_based". Since temporal splits are based on sessions, this is needed to avoid having users for which all ratings were performed during just a single session (so the training or validation set would end up too small).
- **min_posrated_val / min_n_negrated_val:** Integer default: 5.
- **take_complement_of_users:** Bool default: False.
Among all users meeting the requirements, select exactly those that were NOT specified.

#### 2) Evaluation
- **evaluation:** String default: "cross-validation".
The default performs a random shuffle of the user voting data with 5-fold cross-validation. Other options are "train_test_split" (another random shuffle but only for a single split) and "session_based" (a temporal split so that training is done on the oldest voting data and evaluation on the most recent ratings).
- **test_size:** Float default: 0.2.
Percentage of user votes selected for validation. When entering evaluation: "session_based", the algorithm performs the split at the session which brings the validation share closest to this value.
- **stratified:** Bool default: True.
Decides whether training and validation set should have the same positive/negative ratio when entering evaluation: "cross_validation" or "train_test_split". Without it, there is a risk of users not having enough positive or negative labels (for both training and evaluation).

#### 3) Miscellanous
- **users_random_state:** Integer default: null.
The seed when randomly drawing a subset of users by entering users_selection: "random". Has to be null otherwise.
- **model_random_state:** Integer default: 42.
The seed passed to sklearn's LogisticRegression. It is also used for generating train/test splits (for a single split and during cross-validation). If `--all_papers`was NOT set while computing embeddings, this value has to lie in the list
[1, 2, 25, 26, 42, 75, 76, 100, 101, 150, 151].
- **cache_random_state:** Integer default: 42.
The seed used to randomly select a subset of paper IDs from the total 100K cache for training negatives. Same restrictions as for model_random_state.
- **ranking_random_state:** Integer default: 42.
The seed used to randomly select 4 actual negatives and additional simple negatives to evaluate ranking performance. The simple negatives are drawn from research categories untypical for the Scholar Inbox users. Same restrictions as for model_random_state.
- **save_users_predictions:** Bool default: false.
When set to True, the program will store all the model predictions made during the run in .json files. This is generally not necessary and costs additional memory, but enables more detailed visualizations.
- **save_users_coefs:** Bool default: false.
When set to True, the program will store all the trained logistic regression coefficients in .npy files. This is generally not necessary and costs additional memory, but enables using them as a starting point for further processing.
- **load_users_coefs:** Bool default: false.
When set to True, the program will skip training and instead use the coefficients provided under **users_coefs_path** (String default: null) for evaluation.

#### 4) Hyperparameters
For the following settings, you can enter single values or lists. In case of lists, all possible combinations of hyperparameters will be tried.
- **clf_C:** Float default: 0.1.
The inverse regularization strength. A strong value for TF-IDF is 0.4.
- **weights_cache_v:** Float default: 0.8.
The importance of the actual downvotes compared to the random cache negatives. A strong value for TF-IDF is 0.9.
- **weights_neg_scale:** Float default: 8.0.
A scaling factor applied to all negative samples. A strong value for TF-IDF is 1.0.

#### 5) Embedding
Lastly, specify the path to the folder containing the embeddings matrix. 

# III. Running Experiments & Visualization
