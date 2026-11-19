"""Makes cafe_members.csv: 600 made-up loyalty members of the Cozy Corner Café.

Ready to use; you don't need to run it. The CSV it made is already in this folder,
and the lessons, the tests and report.py all read that file.

Every number in the file is made up, and the groups are built in on purpose, so you
can check whether k-means finds them. A real dataset's groups are what you're trying
to find; these are planted. Each member comes from one of four hidden groups, and
which group is NOT in the file:

    morning commuters (180): many visits (about 18 a month), a small spend (about $5.50),
                             mostly before 9 a.m. (about 85%), rarely with food (about 15%)
    weekend brunchers (120): few visits (about 4 a month), a big spend (about $21),
                             rarely before 9 (about 10%), mostly with food (about 85%)
    afternoon regulars (150): some visits (about 10 a month), a small spend (about $7),
                              rarely before 9 (about 10%), sometimes with food (about 35%)
    occasional visitors (150): few visits (about 2.5 a month), a middling spend (about $10),
                               any time of day (about 45% before 9), food about 40% of the time

Each member's numbers wobble around their group's (a normal distribution), so the
groups' edges overlap a little. The rows are shuffled, so the groups aren't in order.

The random numbers come from a fixed seed, 501, so this script makes the same file
every time on the same version of NumPy. NumPy doesn't promise the same random
numbers from a seed across its versions (NEP 19), so the CSV in this folder, not
this script, is what keeps every learner's numbers the same.
"""
import numpy as np
import pandas as pd

SEED = 501

# name: (members, visits mean and spread, spend mean and spread, before-9 share mean and spread,
#        food share mean and spread)
GROUPS = {
    "morning commuters": (180, (18, 3.0), (5.5, 1.0), (0.85, 0.08), (0.15, 0.08)),
    "weekend brunchers": (120, (4, 1.5), (21, 3.5), (0.10, 0.07), (0.85, 0.08)),
    "afternoon regulars": (150, (10, 2.5), (7, 1.5), (0.10, 0.07), (0.35, 0.10)),
    "occasional visitors": (150, (2.5, 1.0), (10, 2.5), (0.45, 0.18), (0.40, 0.15)),
}


def make_members(seed=SEED):
    """Return the made-up café members as a DataFrame, one row per member, in shuffled order."""
    rng = np.random.default_rng(seed)
    parts = []
    for count, visits, spend, before_9, food in GROUPS.values():
        parts.append(pd.DataFrame({
            "visits_per_month": rng.normal(*visits, size=count).round().clip(1, 30),
            "avg_spend": rng.normal(*spend, size=count).clip(2.5, 40).round(2),
            "share_before_9am": rng.normal(*before_9, size=count).clip(0, 1).round(2),
            "share_with_food": rng.normal(*food, size=count).clip(0, 1).round(2),
        }))
    members = pd.concat(parts, ignore_index=True)
    members = members.iloc[rng.permutation(len(members))].reset_index(drop=True)
    members["visits_per_month"] = members["visits_per_month"].astype(int)
    members.insert(0, "member_id", [f"CC-{1001 + i}" for i in range(len(members))])
    return members


if __name__ == "__main__":
    members = make_members()
    members.to_csv("cafe_members.csv", index=False)
    print("Wrote cafe_members.csv:", len(members), "members.")
