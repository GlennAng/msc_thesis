import pandas as pd

from ....src.project_paths import ProjectPaths

mind_path = ProjectPaths.sequence_data_mind_path()


def count_history(history_str: str) -> int:
    if pd.isna(history_str):
        return 0
    return len(history_str.split(" "))


def count_upvotes(votes_str: str) -> int:
    if pd.isna(votes_str):
        return 0
    votes = votes_str.split(" ")
    return len([v for v in votes if int(v.split("-")[1]) == 1])


def count_downvotes(votes_str: str) -> int:
    if pd.isna(votes_str):
        return 0
    votes = votes_str.split(" ")
    return len([v for v in votes if int(v.split("-")[1]) == 0])


subdirs = sorted([d for d in mind_path.iterdir() if d.is_dir()])
subdirs = [d for d in subdirs if d.name != "mind_large_test"]
behaviors_columns = ["session_id_global", "user_id", "time", "history", "votes"]

for subdir in subdirs:
    name = subdir.name
    print(f"Inspecting {name}...")
    behaviors_path = subdir / "behaviors.tsv"
    behaviors = pd.read_csv(behaviors_path, sep="\t", names=behaviors_columns)
    print(f"Number of Sessions: {len(behaviors)}")
    print(f"Number of distinct Users: {behaviors['user_id'].nunique()}")
    print(f"Min Number of Sessions per User: {behaviors['user_id'].value_counts().min()}")
    print(f"Max Number of Sessions per User: {behaviors['user_id'].value_counts().max()}")
    print(f"Mean Number of Sessions per User: {behaviors['user_id'].value_counts().mean()}")
    behaviors["history_n"] = behaviors["history"].apply(count_history)
    print(f"Min History Length: {behaviors['history_n'].min()}")
    print(f"Max History Length: {behaviors['history_n'].max()}")
    print(f"Mean History Length: {behaviors['history_n'].mean()}")
    behaviors["upvotes_n"] = behaviors["votes"].apply(count_upvotes)
    print(f"Min Upvotes: {behaviors['upvotes_n'].min()}")
    print(f"Max Upvotes: {behaviors['upvotes_n'].max()}")
    print(f"Mean Upvotes: {behaviors['upvotes_n'].mean()}")
    print(f"Total Upvotes: {behaviors['upvotes_n'].sum()}")
    behaviors["downvotes_n"] = behaviors["votes"].apply(count_downvotes)
    print(f"Total Downvotes: {behaviors['downvotes_n'].sum()}")
    print("_______________________\n")


# merged
behaviors_list = []
for subdir in subdirs:
    behaviors_path = subdir / "behaviors.tsv"
    behaviors = pd.read_csv(behaviors_path, sep="\t", names=behaviors_columns)
    behaviors_list.append(behaviors)
merged_behaviors = pd.concat(behaviors_list, ignore_index=True)
print(f"Inspecting Merged Data...")
print(f"Number of Sessions: {len(merged_behaviors)}")
print(f"Number of distinct Users: {merged_behaviors['user_id'].nunique()}")
print(f"Min Number of Sessions per User: {merged_behaviors['user_id'].value_counts().min()}")
print(f"Max Number of Sessions per User: {merged_behaviors['user_id'].value_counts().max()}")
print(f"Mean Number of Sessions per User: {merged_behaviors['user_id'].value_counts().mean()}")
merged_behaviors["history_n"] = merged_behaviors["history"].apply(count_history)
print(f"Min History Length: {merged_behaviors['history_n'].min()}")
print(f"Max History Length: {merged_behaviors['history_n'].max()}")
print(f"Mean History Length: {merged_behaviors['history_n'].mean()}")
merged_behaviors["upvotes_n"] = merged_behaviors["votes"].apply(count_upvotes)
print(f"Min Upvotes: {merged_behaviors['upvotes_n'].min()}")
print(f"Max Upvotes: {merged_behaviors['upvotes_n'].max()}")
print(f"Mean Upvotes: {merged_behaviors['upvotes_n'].mean()}")
print(f"Total Upvotes: {merged_behaviors['upvotes_n'].sum()}")
merged_behaviors["downvotes_n"] = merged_behaviors["votes"].apply(count_downvotes)
print(f"Min Downvotes: {merged_behaviors['downvotes_n'].min()}")
print(f"Max Downvotes: {merged_behaviors['downvotes_n'].max()}")
print(f"Mean Downvotes: {merged_behaviors['downvotes_n'].mean()}")
print(f"Total Downvotes: {merged_behaviors['downvotes_n'].sum()}")
merged_behaviors["click_through_rate"] = merged_behaviors["upvotes_n"] / (
    merged_behaviors["upvotes_n"] + merged_behaviors["downvotes_n"]
)
print(f"Mean Click Through Rate: {merged_behaviors['click_through_rate'].mean()}")
print(f"Mean Votes per Session: {(merged_behaviors['upvotes_n'] + merged_behaviors['downvotes_n']).mean()}")
print("_______________________\n")
