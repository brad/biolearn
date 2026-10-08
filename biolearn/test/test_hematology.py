import numpy as np
import pandas as pd

from biolearn.hematology import phenotypic_age


def _example():
    return pd.DataFrame(
        [
            {
                "age": 35,
                "albumin": 46,
                "creatinine": 80,
                "glucose": 4.9,
                "c_reactive_protein": 0.08,
                "lymphocyte_percent": 33,
                "mean_cell_volume": 89,
                "red_blood_cell_distribution_width": 12.4,
                "alkaline_phosphate": 62,
                "white_blood_cell_count": 5.6,
            }
        ]
    )


def _levine_2018(row):
    xb = (
        -19.9067
        + 0.0804 * row["age"]
        - 0.0336 * row["albumin"]
        + 0.0095 * row["creatinine"]
        + 0.1953 * row["glucose"]
        + 0.0954 * np.log(row["c_reactive_protein"])
        - 0.0120 * row["lymphocyte_percent"]
        + 0.0268 * row["mean_cell_volume"]
        + 0.3306 * row["red_blood_cell_distribution_width"]
        + 0.00188 * row["alkaline_phosphate"]
        + 0.0554 * row["white_blood_cell_count"]
    )
    gamma = 0.0076927
    mortality = 1 - np.exp(-np.exp(xb) * (np.exp(120 * gamma) - 1) / gamma)
    return (
        141.50225 + np.log(-0.00553 * np.log(1.00001 - mortality)) / 0.090165
    )


def test_phenotypic_age_matches_levine_2018():
    df = _example()
    expected = _levine_2018(df.iloc[0])

    result = phenotypic_age(df)

    assert abs(result.iloc[0] - expected) < 1e-6
    assert abs(result.iloc[0] - 24.3587) < 1e-3
