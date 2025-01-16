import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
from typing import Tuple


################
# Merging Data #
################


def transform_daily_data_to_quarterly_data(data):
    """
    Transform daily data to quarterly data. We decided to use the last not null value of the quarter for each feature,
    in order to get the most recent information for each quarter.
    For the price we decided to use the average and the std of the prices of the quarter.
    """
    return (
        data.groupby(["quarter", "asset_num"])
        .agg(
            price=(
                "price",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            cap_ex=(
                "cap_ex",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            cash=(
                "cash",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            dil_shares=(
                "dil_shares",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            ebit=(
                "ebit",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            ebitda=(
                "ebitda",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            net_inc=(
                "net_inc",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            survivor=(
                "survivor",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            T_assets=(
                "T_assets",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            T_debt=(
                "T_debt",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            T_rev=(
                "T_rev",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
            sector=(
                "sector",
                lambda x: x.dropna().iloc[-1] if not x.dropna().empty else None,
            ),
        )
        .reset_index()
    )


#####################
# Transforming Data #
#####################

dict_sector = {
    "Industrial Machinery": "Industrial Machinery & Supplies & Components",
    "Paper Packaging": "Paper & Plastic Packaging Products & Materials",
    "Semiconductor Equipment": "Semiconductor Materials & Equipment",
    "Construction Machinery & Heavy Trucks": "Construction Machinery & Heavy Transportation Equipment",
    "Soft Drinks": "Soft Drinks & Non-alcoholic Beverages",
    "Personal Products": "Personal Care Products",
    "Metal & Glass Containers": "Metal, Glass & Plastic Containers",
    "Auto Parts & Equipment": "Automotive Parts & Equipment",
    "Pharmaceuticals, Biotechnology & Life Sciences": "Pharmaceuticals, Biotechnology  & Life Sciences",
    "Internet & Direct Marketing Retail": "Consumer Discretionary",
    "Department Stores": "Consumer Discretionary",
    "Thrifts & Mortgage Finance": "Commercial & Residential Mortgage Finance",
    "General Merchandise Stores": "Consumer Staples Merchandise Retail",
}


def fix_sector(df: pd.DataFrame) -> pd.DataFrame:
    df["sector"] = df["sector"].replace(dict_sector)
    return df


def get_tables():
    url = "https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table", {"class": "wikitable"})
    return tables


def get_data() -> pd.DataFrame:
    tables = get_tables()
    target_table = tables[0]
    data = []
    headers = []

    for th in target_table.find_all("th"):
        headers.append(th.text.strip())
    size = 4

    for row in target_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        text = [col.text.strip() for col in cols if not col.text.strip().isdigit()]

        while len(text) < size:
            text = [None] + text
        data.append(text)
    df = pd.DataFrame(data, columns=headers)
    df = df.ffill(axis=0)

    df.iloc[118:131, 0] = "Information Technology"
    df.iloc[118:123, 1] = "Software & Services"

    return df


def subindustry_classification(subindustry: str, df: pd.DataFrame) -> str:
    # Returns a list with the category of the subindustry
    for column in df.columns:
        if subindustry in df[column].values:
            sector = df[df[column] == subindustry]["Sector"].values[0]
            break
        else:
            sector = None
    return sector


def reduce_sector_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    data_subindustry = get_data()
    df = fix_sector(df)
    df["new_sector"] = df.apply(
        lambda x: subindustry_classification(x["sector"], data_subindustry), axis=1
    )
    return df


###################
# Training models #
###################


def calculate_mean_performance_sector(df: pd.DataFrame, features) -> pd.DataFrame:
    for feature in features:
        df[feature + "_mean_sector"] = df.groupby(["quarter", "sector"])[
            feature
        ].transform("mean")
        df[feature + "_mean_sector_diff"] = df[feature] - df[feature + "_mean_sector"]
        df.drop(columns=[feature + "_mean_sector"], inplace=True)
    return df


def prepare_train_test_data(
    dataset: pd.DataFrame,
    quarters_col: str,
    features: list[str],
    target: str,
    window_size: int = 10,
) -> list[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    data = dataset.sort_values(by=quarters_col)
    unique_quarters = data[quarters_col].unique()
    datasets = {}

    for i in range(len(unique_quarters) - window_size):
        train_quarters = unique_quarters[i : i + window_size]
        test_quarter = unique_quarters[i + window_size]

        train_data = data[data[quarters_col].isin(train_quarters)]
        test_data = data[data[quarters_col] == test_quarter]

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        datasets[str(test_quarter)] = (X_train, y_train, X_test, y_test)

    return datasets


def return_learning_curve(results: dict, set_: str) -> pd.DataFrame:
    learning_curves = pd.DataFrame()
    for quarter in results:
        learning_curve = pd.DataFrame(results[(quarter)][set_])
        learning_curve["n_trees"] = list(range(len(learning_curve)))
        learning_curve["quarter"] = quarter
        learning_curves = pd.concat([learning_curves, learning_curve])

    for column in learning_curves.columns:
        if column not in ["n_trees", "quarter"]:
            first_metric = learning_curves[[column, "quarter"]].drop_duplicates(
                ["quarter"]
            )
            first_metric = first_metric.rename(columns={column: "first_" + column})
            learning_curves = pd.merge(learning_curves, first_metric, on=["quarter"])
            learning_curves["norm_" + column] = (
                learning_curves[column] - learning_curves["first_" + column]
            ) / learning_curves["first_" + column]
            learning_curves = learning_curves.drop(columns="first_" + column)
    learning_curves["quarter"] = learning_curves["quarter"].astype(str)

    return learning_curves


def create_results_df(
    df_final,
    predictions: dict,
    quarter: str,
    sector: bool = False,
    column: str = "asset_return_diff_sp500",
) -> pd.DataFrame:
    assets = df_final[df_final["quarter"] == quarter]["asset_num"]
    df = pd.DataFrame(assets).reset_index(drop=True)
    df["prediction"] = predictions[quarter]
    df["asset_num"] = assets.values
    df["rank_pos"] = df["prediction"].rank(ascending=False)
    df["rank_neg"] = df["prediction"].rank(ascending=True)
    df = df.merge(
        df_final[df_final["quarter"] == quarter][
            ["asset_num", "asset_return_gt_sp500", column]
        ],
        on="asset_num",
    )

    if sector:
        df = df.merge(
            df_final[df_final["quarter"] == quarter][
                ["asset_num", "Baseline", "new_sector"]
            ],
            on="asset_num",
        )
    else:
        df = df.merge(
            df_final[df_final["quarter"] == quarter][["asset_num", "Baseline"]],
            on="asset_num",
        )
    return df


def equitative_return(
    df: pd.DataFrame, n: int, column: str = "asset_return_diff_sp500"
) -> pd.DataFrame:
    df = df.sort_values("rank_pos", ascending=False).head(n)
    return df[column].mean()


def acumulative_quarter_return(
    returns_df: pd.DataFrame,
    sector: bool = False,
    valor_inicial_cartera_quarter: int = 25,
) -> pd.DataFrame:
    result_df = returns_df.copy()
    if sector:
        for column in result_df.columns:
            for sector in result_df["sector"].unique():
                dict_quarter = {
                    "Q1": valor_inicial_cartera_quarter,
                    "Q2": valor_inicial_cartera_quarter,
                    "Q3": valor_inicial_cartera_quarter,
                    "Q4": valor_inicial_cartera_quarter,
                }
                if column not in ["quarter", "sector"]:
                    for index, row in result_df.iterrows():
                        if row["sector"] == sector:
                            for quarter in dict_quarter:
                                if quarter in row["quarter"]:
                                    dict_quarter[quarter] += (
                                        dict_quarter[quarter] * row[column]
                                    )
                                    result_df.loc[
                                        index, f"cumulative_return_{column}"
                                    ] = dict_quarter[quarter]

    else:
        for column in result_df.columns:
            dict_quarter = {
                "Q1": valor_inicial_cartera_quarter,
                "Q2": valor_inicial_cartera_quarter,
                "Q3": valor_inicial_cartera_quarter,
                "Q4": valor_inicial_cartera_quarter,
            }
            if column not in ["quarter", "sector"]:
                for index, row in result_df.iterrows():
                    for quarter in dict_quarter:
                        if quarter in row["quarter"]:
                            dict_quarter[quarter] += dict_quarter[quarter] * row[column]
                            result_df.loc[
                                index, f"cumulative_return_{column}"
                            ] = dict_quarter[quarter]
    return result_df


def acumulative_year_earnings(returns_df: pd.DataFrame) -> pd.DataFrame:
    result_df = acumulative_quarter_return(returns_df)
    acum_df = pd.DataFrame()
    unique_years = result_df["quarter"].str[:4].unique()
    unique_years = np.append(unique_years, str(int(unique_years[-1]) + 1))
    for column in result_df.columns:
        dict_year = {year: 0 for year in unique_years}
        if column not in ["quarter", "sector"]:
            for index, row in result_df.iterrows():
                for year in dict_year:
                    if year in row["quarter"]:
                        dict_year[year] += row[column]
            acum_df[column] = dict_year.values()

    acum_df = acum_df.loc[:, acum_df.columns.str.contains("cumulative_return")]
    acum_df.index = dict_year.keys()
    # Hardcodeada
    acum_df = acum_df.shift(1)
    acum_df.loc[unique_years[0]] = 100
    return acum_df
