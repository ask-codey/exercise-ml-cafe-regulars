# Runs one round of k-means by hand, then groups the Cozy Corner Café's 600 members with your four functions from
# cafe_groups.py, and prints what each step gave, so you can see them at work. Ready to use.
# Run it with: python report.py
import pandas as pd

from cafe_groups import assign_to_nearest, cluster_members, describe_groups, move_centers

six = [(18, 5), (16, 6), (20, 4), (4, 20), (2, 22), (6, 18)]
centers = [(18, 5), (16, 6)]
assignments = assign_to_nearest(six, centers)
print("1) six members, nearest of two starting centers:", assignments)
print("2) the centers moved to:", move_centers(six, assignments, 2))

members = pd.read_csv("cafe_members.csv")
model = cluster_members(members, 4, 0)
labels = model.named_steps["kmeans"].labels_
print("3) members:", len(members))
print("4) labels:", len(labels))
summary = describe_groups(members, labels)
print("5) the four groups:")
print(summary.round(2).to_string())
