import pandas as pd
from ....src.load_files import load_papers, load_users_ratings, load_finetuning_users

def get_l1_distribution(papers_df: pd.DataFrame) -> pd.Series:
    distribution = papers_df["l1"].value_counts()
    distribution = distribution / distribution.sum()
    return distribution

papers = load_papers()
users_ratings = load_users_ratings()
test_users = load_finetuning_users(selection="test")
users_ratings = users_ratings[users_ratings["user_id"].isin(test_users)]
users_ratings = users_ratings[users_ratings["rating"] == 1]
users_ratings = users_ratings.merge(
    papers[["paper_id", "l1"]],
    on="paper_id",
    how="left")

for user_id in test_users:
    user_ratings = users_ratings[users_ratings["user_id"] == user_id]
    if user_ratings.empty:
        continue
    distribution = get_l1_distribution(user_ratings)
    s = f"User {user_id}, {distribution.index[0]}: {distribution.values[0]:.2f}"
    if distribution.values[0] < 1:
        s += f", {distribution.index[1]}: {distribution.values[1]:.2f}"
    if distribution.values[0] < 1 and distribution.values[0] + distribution.values[1] < 1:
        s += f", {distribution.index[2]}: {distribution.values[2]:.2f}"
    print(s)


# 10830 Biology 0.61, Medicine 0.25
# 9079 Psychology: 0.42, Computer Science 0.19
# User 7260, Biology: 0.85, Psychology: 0.07
# User 4871, Biology: 0.39, Physics: 0.27
#User 5254, Physics: 0.52, Biology: 0.21