# Tests for cafe_groups.py. Run them with: python -m pytest (in the exercise folder, with the virtual environment
# active). Each test calls one of your functions, on the 600 members in cafe_members.csv or on a small table written
# in the test, and checks what comes back. A failing test says what it called, what it expected and what came back.
# None of them checks which members k-means puts together on the real data, which depends on its random starts: they
# check the hand round exactly, the pipeline's steps and settings, the labels' shape, that the same seed gives the
# same groups, and the group summaries' arithmetic. You don't need to change this file.
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cafe_groups import FEATURES, assign_to_nearest, cluster_members, describe_groups, move_centers

# Six members as (visits, spend): three early regulars and three big spenders, the hand example.
SIX = [(18, 5), (16, 6), (20, 4), (4, 20), (2, 22), (6, 18)]

# Eight members in two groups so far apart that any run of k-means with k = 2 splits them the same way: the first
# four (many visits, a small spend, early, little food) and the last four (few visits, a big spend, late, food).
TWO_CLEAR_GROUPS = pd.DataFrame({
    "member_id": ["CC-9001", "CC-9002", "CC-9003", "CC-9004", "CC-9005", "CC-9006", "CC-9007", "CC-9008"],
    "visits_per_month": [20, 19, 21, 20, 3, 4, 2, 3],
    "avg_spend": [5.0, 5.5, 4.5, 5.0, 22.0, 21.0, 23.0, 22.0],
    "share_before_9am": [0.90, 0.85, 0.95, 0.90, 0.10, 0.05, 0.15, 0.10],
    "share_with_food": [0.10, 0.15, 0.05, 0.10, 0.90, 0.85, 0.95, 0.90],
})


def check(call, got, expected):
    """Fails with a message that says what the call returned and what it should return."""
    assert got == expected, f"{call} should return {expected!r}, but it returned {got!r}."


def labels_of(model):
    return list(model.named_steps["kmeans"].labels_)


# assign_to_nearest

def test_1_assign_to_nearest_starting_centers_18_5_and_16_6_give_0_1_0_1_1_1():
    check("assign_to_nearest(SIX, [(18, 5), (16, 6)])", assign_to_nearest(SIX, [(18, 5), (16, 6)]),
          [0, 1, 0, 1, 1, 1])


def test_2_assign_to_nearest_centers_18_5_and_4_20_give_0_0_0_1_1_1():
    check("assign_to_nearest(SIX, [(18, 5), (4, 20)])", assign_to_nearest(SIX, [(18, 5), (4, 20)]),
          [0, 0, 0, 1, 1, 1])


def test_3_assign_to_nearest_a_tie_goes_to_the_first_center():
    check("assign_to_nearest([(4, 10), (8, 10)], [(6, 8), (6, 12)])",
          assign_to_nearest([(4, 10), (8, 10)], [(6, 8), (6, 12)]), [0, 0])


def test_4_assign_to_nearest_works_with_four_features():
    points = [(20, 5.0, 0.9, 0.1), (3, 22.0, 0.1, 0.9)]
    centers = [(3, 21.0, 0.1, 0.8), (19, 5.5, 0.8, 0.2)]
    check("assign_to_nearest(two members with four features, two centers)", assign_to_nearest(points, centers),
          [1, 0])


# move_centers

def test_5_move_centers_0_1_0_1_1_1_gives_19_4_5_and_7_16_5():
    check("move_centers(SIX, [0, 1, 0, 1, 1, 1], 2)", move_centers(SIX, [0, 1, 0, 1, 1, 1], 2),
          [(19.0, 4.5), (7.0, 16.5)])


def test_6_move_centers_0_0_0_1_1_1_gives_18_5_and_4_20():
    check("move_centers(SIX, [0, 0, 0, 1, 1, 1], 2)", move_centers(SIX, [0, 0, 0, 1, 1, 1], 2),
          [(18.0, 5.0), (4.0, 20.0)])


def test_7_move_centers_keeps_center_0_first_when_the_first_point_is_in_group_1():
    check("move_centers(SIX, [1, 1, 1, 0, 0, 0], 2)", move_centers(SIX, [1, 1, 1, 0, 0, 0], 2),
          [(4.0, 20.0), (18.0, 5.0)])


def test_8_move_centers_works_with_three_groups_and_three_features():
    points = [(2, 4, 6), (4, 6, 8), (10, 10, 10), (20, 0, 1), (22, 2, 3)]
    check("move_centers(five points with three features, [0, 0, 1, 2, 2], 3)",
          move_centers(points, [0, 0, 1, 2, 2], 3), [(3.0, 5.0, 7.0), (10.0, 10.0, 10.0), (21.0, 1.0, 2.0)])


# cluster_members

def test_9_cluster_members_returns_a_standard_scaler_then_kmeans_pipeline(members):
    model = cluster_members(members, 4, 0)
    steps = list(model.named_steps)
    check("the step names of cluster_members(members, 4, 0)", steps, ["standardscaler", "kmeans"])
    assert isinstance(model.named_steps["standardscaler"], StandardScaler), \
        f"The first step should be a StandardScaler, but it was {model.named_steps['standardscaler']!r}."
    assert isinstance(model.named_steps["kmeans"], KMeans), \
        f"The second step should be a KMeans, but it was {model.named_steps['kmeans']!r}."


def test_10_cluster_members_sets_n_clusters_n_init_and_random_state(members):
    params = cluster_members(members, 3, 7).named_steps["kmeans"].get_params()
    check("n_clusters in cluster_members(members, 3, 7)", params["n_clusters"], 3)
    check("n_init in cluster_members(members, 3, 7)", params["n_init"], 10)
    check("random_state in cluster_members(members, 3, 7)", params["random_state"], 7)


def test_11_cluster_members_gives_each_of_the_600_members_one_label_from_0_to_3(members):
    labels = labels_of(cluster_members(members, 4, 0))
    check("how many labels cluster_members(members, 4, 0) gives", len(labels), 600)
    check("the different labels cluster_members(members, 4, 0) gives", sorted(set(int(n) for n in labels)),
          [0, 1, 2, 3])


def test_12_cluster_members_learns_from_the_four_features_only(members):
    model = cluster_members(members, 4, 0)
    check("the columns the pipeline learned from", list(model.named_steps["standardscaler"].feature_names_in_),
          FEATURES)


def test_13_cluster_members_seed_0_gives_the_same_labels_as_random_state_0(members):
    expected = make_pipeline(StandardScaler(), KMeans(n_clusters=4, n_init=10, random_state=0)).fit_predict(
        members[FEATURES])
    check("the labels of cluster_members(members, 4, 0)", labels_of(cluster_members(members, 4, 0)),
          list(expected))


def test_14_cluster_members_seed_7_gives_the_same_labels_as_random_state_7(members):
    expected = make_pipeline(StandardScaler(), KMeans(n_clusters=5, n_init=10, random_state=7)).fit_predict(
        members[FEATURES])
    check("the labels of cluster_members(members, 5, 7)", labels_of(cluster_members(members, 5, 7)),
          list(expected))


def test_15_cluster_members_splits_two_clear_groups_the_same_way_for_any_seed():
    for seed in [0, 1, 7]:
        labels = labels_of(cluster_members(TWO_CLEAR_GROUPS, 2, seed))
        same = [labels[0]] * 4 + [labels[4]] * 4
        check(f"the labels of cluster_members(two clear groups, 2, {seed})", labels, same)
        assert labels[0] != labels[4], \
            f"With seed {seed}, the two clear groups should get different labels, but both got {labels[0]}."


# describe_groups

def test_16_describe_groups_gives_each_groups_means_and_size():
    summary = describe_groups(TWO_CLEAR_GROUPS, [0, 0, 0, 0, 1, 1, 1, 1])
    check("the columns of describe_groups(two clear groups, labels)", list(summary.columns),
          FEATURES + ["members"])
    check("the index of describe_groups(two clear groups, labels)", list(summary.index), [0, 1])
    check("group 0's visits_per_month", summary.loc[0, "visits_per_month"], 20.0)
    check("group 1's avg_spend", summary.loc[1, "avg_spend"], 22.0)
    check("group 0's share_with_food", summary.loc[0, "share_with_food"], pytest.approx(0.1))
    check("the members column", list(summary["members"]), [4, 4])


def test_17_describe_groups_sorts_the_groups_by_number():
    summary = describe_groups(TWO_CLEAR_GROUPS, [2, 2, 2, 0, 0, 0, 0, 2])
    check("the index of describe_groups(two clear groups, [2, 2, 2, 0, 0, 0, 0, 2])", list(summary.index), [0, 2])
    check("the members column", list(summary["members"]), [4, 4])
    check("group 2's visits_per_month, the mean of 20, 19, 21 and 3", summary.loc[2, "visits_per_month"], 15.75)


def test_18_describe_groups_matches_pandas_on_the_600_members(members):
    labels = make_pipeline(StandardScaler(), KMeans(n_clusters=4, n_init=10, random_state=0)).fit_predict(
        members[FEATURES])
    summary = describe_groups(members, labels)
    expected = members[FEATURES].groupby(labels).mean()
    check("the index of describe_groups on the 600 members", list(summary.index), [0, 1, 2, 3])
    for feature in FEATURES:
        assert list(summary[feature]) == pytest.approx(list(expected[feature])), \
            f"The {feature} means should be {list(expected[feature])!r}, but they were {list(summary[feature])!r}."
    check("the members column on the 600 members", list(summary["members"]),
          list(pd.Series(labels).value_counts().sort_index()))
    check("the total of the members column", int(summary["members"].sum()), 600)
