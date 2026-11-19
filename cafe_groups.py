"""Cozy Corner Café: four functions for you to finish.

The docstring under each function says what it should do, with examples.
Replace each raise NotImplementedError("Not built yet.") line with your own code.
Return each answer: the tests read what comes back, not what's printed.

The first two functions are plain Python: one round of k-means, by hand.
points and centers are lists of tuples, one number per feature, such as
(visits, spend).

The last two use scikit-learn and pandas. members is a pandas table (a DataFrame)
with one row per member and the columns member_id, visits_per_month, avg_spend,
share_before_9am and share_with_food, like the one pd.read_csv("cafe_members.csv")
gives you. The imports below are everything the four functions need.
"""
import math

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The columns k-means groups the members by (its features), in this order. Ready to use.
FEATURES = ["visits_per_month", "avg_spend", "share_before_9am", "share_with_food"]


def assign_to_nearest(points, centers):
    """Return a list with the number of each point's nearest center, in the points' order.

    A center's number is its position in centers: 0 for the first. Measure with the
    straight-line distance (math.dist). If a point is equally near two centers, give
    it the one that comes first in centers.

    points [(18, 5), (16, 6), (4, 20)] and centers [(18, 5), (3, 21)] -> [0, 0, 1]
    points [(4, 10)] and centers [(6, 8), (6, 12)] (a tie)           -> [0]
    """
    raise NotImplementedError("Not built yet.")


def move_centers(points, assignments, k):
    """Return the k new centers: center number n is the mean of the points assigned to n.

    assignments[i] is the number of the center points[i] belongs to, from 0 to k - 1.
    Each new center is a tuple with one mean per feature, in the points' order of
    features, and the list holds center 0 first. Every number from 0 to k - 1 has at
    least one point.

    points [(18, 5), (20, 4), (4, 20), (2, 22)], assignments [0, 0, 1, 1] and k = 2
    -> [(19.0, 4.5), (3.0, 21.0)]
    """
    raise NotImplementedError("Not built yet.")


def cluster_members(members, k, seed):
    """Group the members into k groups, and return the fitted pipeline.

    Make a pipeline of two steps with make_pipeline: StandardScaler(), then
    KMeans(n_clusters=k, n_init=10, random_state=seed). Fit it on the FEATURES
    columns only (members[FEATURES]), so member_id and any other column are left
    out, and return it. Its KMeans step then holds each member's group number in
    labels_, one per member, from 0 to k - 1:

    model = cluster_members(members, 4, 0)
    model.named_steps["kmeans"].labels_   # 600 group numbers for the 600 members
    """
    raise NotImplementedError("Not built yet.")


def describe_groups(members, labels):
    """Return a table describing each group: its mean for every feature, and its size.

    labels holds each member's group number, in the members' order. The table has one
    row per group, the group numbers as its index from smallest to largest, and the
    columns FEATURES in order, each the group's mean, then "members", how many members
    the group has. Leave the means unrounded.

    With four members, visits_per_month 20, 16, 4 and 2, and labels [0, 0, 1, 1]:
    group 0's visits_per_month is 18.0 and its members 2; group 1's are 3.0 and 2.
    """
    raise NotImplementedError("Not built yet.")
