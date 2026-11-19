# Exercise: Café Regulars

Exercise code for the lesson **Exercise: Café Regulars** in the Machine Learning with Python course on [ask-codey.com](https://www.ask-codey.com).

The Cozy Corner Café wants to know what kinds of regulars it has, and its 600 loyalty members have no column that says. You'll write four functions in `cafe_groups.py`: two that run one round of k-means by hand in plain Python, one that groups the members with scikit-learn, and one that describes each group with pandas. Tests call each function and check what it returns.

## What's here

- `cafe_members.csv`: the 600 members, our own made-up data: each member's `member_id`, `visits_per_month`, `avg_spend` (dollars a visit), `share_before_9am` and `share_with_food` (0 to 1: 0.85 means 85% of their visits). The lessons, the tests and `report.py` all read this file. Ready to use.
- `seed_data.py`: the script that made `cafe_members.csv`, so you can see how the data was built and the four groups planted in it (they aren't in the file). You don't need to run it: NumPy doesn't promise the same random numbers from a seed on every version, so the CSV, not the script, is what keeps everyone's numbers the same. Ready to use.
- `cafe_groups.py`: the four functions, `assign_to_nearest`, `move_centers`, `cluster_members` and `describe_groups`, with the `FEATURES` they use. Each function stops with `NotImplementedError: Not built yet.` for now. The docstring under each says what it should do, with examples. For you to finish.
- `explore.ipynb`: a notebook for looking at the members and trying your functions as you write them. The tests don't read it. Ready to use, and yours to change.
- `report.py`: runs one round by hand, groups the members and describes the groups with your four functions, and prints what each step gave, so you can see them at work. Ready to use.
- `tests/test_cafe_groups.py`: tests that call your functions and check what comes back. You don't need to change this file.
- `tests/conftest.py`: gives each test a fresh copy of the 600 members from `cafe_members.csv`. Ready to use.
- `requirements.txt`: the packages this exercise needs, each at an exact version: scikit-learn, pandas, NumPy, ipykernel (so VS Code can run the notebook) and pytest. Ready to use.
- `pytest.ini`: tells pytest where the tests are and where to find `cafe_groups.py`. Ready to use.
- `.gitignore`: tells Git to leave out the virtual environment and the files Python, pytest and the notebook make as they run. Ready to use.

## Before you start

The packages go in a virtual environment: a private set of packages for this folder only. In VS Code's terminal, in this folder, run these three commands.

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On a Mac (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- The first command makes the virtual environment, in a folder named `.venv`.
- The second turns it on for this terminal: the prompt starts with `(.venv)` from then on.
- The third installs the packages `requirements.txt` lists into it.

*Virtual Environments and pip* explains all three. The `.venv` folder is in `.gitignore`, so Git leaves it out: these commands make it again on any computer.

If PowerShell answers that running scripts is disabled on this system, run this once, then run the second command again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Each new terminal starts with the virtual environment off. If the prompt doesn't start with `(.venv)`, run the second command again. (VS Code often turns it on for you.)

To look at the members, open `explore.ipynb`, choose the `.venv` kernel with **Select Kernel** (top right), and run the cells.

## Check your work

```bash
python -m pytest
```

At first every test fails: that is expected, because each function stops with a `NotImplementedError` for now. Each function you finish turns its tests green. A failing test's message says what it called, what it expected and what came back, for example *assign_to_nearest(SIX, [(18, 5), (16, 6)]) should return [0, 1, 0, 1, 1, 1], but it returned [0, 0, 0, 1, 1, 1].*

No test checks which members k-means puts together in the real data: that depends on its random starts. The tests check the hand round exactly, the pipeline's steps and settings, one label per member, that the same seed gives the same groups, and the group summaries' arithmetic.

To see your functions at work on the whole file, run:

```bash
python report.py
```

## The rules

- Keep each function's name and parameters as they are, and keep `FEATURES` as it is.
- Write `assign_to_nearest` and `move_centers` in plain Python: no scikit-learn, NumPy or pandas in those two.
- Pass `seed` on as `random_state`, and set `n_init=10`, so the same seed always gives the same groups.
- Return each answer; don't only print it.
- Don't change the tests, `conftest.py`, `cafe_members.csv`, `seed_data.py` or `report.py`.

## Tips

- Try each step in `explore.ipynb` first, then move the lines that work into `cafe_groups.py`.
- `math.dist(p, q)` gives the straight-line distance between two tuples with any number of numbers in them.
- `zip(*points)` turns a list of points into one tuple per feature, which makes a mean per feature short to write. `tuple(...)` around a comprehension in round brackets, such as `tuple(x * 2 for x in (1, 2))`, gives `(2, 4)`: a new center as a tuple.
- `make_pipeline(StandardScaler(), KMeans(...))` names its steps `standardscaler` and `kmeans` for you.
- After you change `cafe_groups.py`, restart the notebook's kernel before you try it there, so the notebook imports your newest code.

## Hand it in

When it works, push your solution to a public repo of your own on GitHub, then paste that repo's address into **Submit your exercise** on the lesson page at [ask-codey.com](https://www.ask-codey.com). That's what completes the lesson. The lesson page shows the Git commands, step by step.
