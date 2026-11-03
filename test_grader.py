"""Regression cases for grader_v3. Run: python -m pytest test_grader_v3.py -q
Most cases are real answers from the LFM2.5-350M test logs, and there `current` is the verdict the
dataset grader logged. The '1,869', '1351+975+932' and 'e=4208' cases are illustrative, and their
`current` value is what the grader's design implies."""

import pytest
from grader import grade

CASES = [  # gold, answer, reward_mode, question, current, v3
    (
        "50",
        "Iris-setosa: 50, Iris-versicolor: 50, Iris-virginica: 50",
        "numeric",
        "How many samples of each species?",
        1,
        0,
    ),
    ("50", "50, 50, 50", "numeric", "How many samples of each species?", 1, 1),
    ("50", "50,50,50", "numeric", "How many samples of each species?", 0, 1),  # current: commas stripped -> 505050
    ("2006", "Wii Sports, 2006", "numeric", "Which year had the highest sales?", 1, 0),
    ("0", "Class 0", "exact_short", "Which class is the majority?", 1, 0),
    ("82", "Hospital overall rating\n3    1761\n5    82", "numeric", "How many hospitals have 5 stars?", 1, 0),
    (
        "Agglomerative Clustering Cluster 1",
        "1",
        "exact_short",
        "Which cluster ...?",
        1,
        0,
    ),  # math_verify read '1' out of the label
    ("5", "2, 3, or 5", "exact_short", "Which configuration (2, 3, or 5)?", 1, 0),
    ("0.05", "0.01", "numeric", "What is the optimal learning rate after tuning?", 1, 0),
    ("0.83", "0.8717541573048714", "numeric", "What is the correlation coefficient ...?", 1, 0),
    ("0.97", "1.0", "numeric", "What is the highest F1-score ...?", 1, 0),
    ("0.96", "100", "numeric", "What is the accuracy of the model?", 1, 0),
    ("0.999204147794", "100", "numeric", "What is the accuracy of the model?", 1, 0),
    ("74.97", "0.75", "numeric", "What is the accuracy of the model?", 1, 0),
    ("33.33", "0.333333", "numeric", "What is the accuracy of the model?", 1, 1),
    ("0.1727", "0.001727", "numeric", "What percentage of transactions are fraudulent?", 0, 1),
    ("0.1727", "0.172748563062", "numeric", "What percentage of transactions are fraudulent?", 1, 1),
    ("17.94353886271143", "17.94", "numeric", "What was the mean unemployment rate?", 1, 1),
    ("0.9993", "0.998273", "numeric", "Out-of-sample accuracy rounded to four decimals?", 1, 0),
    ("2001", "2000", "numeric", "Which year ...?", 0, 0),
    ("1,869", "1869", "numeric", "How many churned?", 1, 1),
    ("3258", "1351+975+932", "numeric", "Combined total of top 3 publishers?", 1, 0),
    ("Male", "male", "exact_short", "Which gender ...?", 1, 1),
    ("Petal", "petal", "exact_short", "Which feature set ...?", 1, 1),
    ("yes", "Yes", "exact_bool", "Is it significant?", 1, 1),
    ("e=4208, p=3916", "p=3916, e=4208", "list_csv", "Distribution?", 1, 1),
    # behaviours confirmed by running the dataset's grader.py with default tolerances
    ("5", ".5", "numeric", "How many folds?", 1, 0),  # regex drops the leading dot
    ("-0.5", "-.5", "numeric", "Skewness?", 0, 1),
    ("0.000393", "0", "numeric", "What is the p-value?", 1, 0),  # atol 1e-3 > the value itself
    ("Model 2", "Group 2", "exact_short", "Which model ...?", 1, 0),  # math_verify on labels
    ("Edible, 4208", "Poisonous, 4208", "list_csv", "Which class and how many?", 1, 0),
    ("1,5", "15", "numeric", "Value?", 1, 0),  # commas deleted before parsing
    ("yes", "True", "exact_bool", "Is it significant?", 0, 1),
    ("Male", "Male.", "exact_short", "Which gender?", 0, 1),
    (
        "0",
        "1",
        "numeric",
        "Question:\nHow many duplicates?\n\nKeep decimal precision.",
        0,
        0,
    ),  # boilerplate is not the question
    ("5", "492", "numeric", "What is the accuracy score with 5 folds?", 0, 0),  # rescaled integer needs 1 dp
]


@pytest.mark.parametrize("gold,ans,mode,q,current,want", CASES)
def test_v3(gold, ans, mode, q, current, want):
    assert grade(gold, ans, reward_mode=mode, question=q).reward == want
