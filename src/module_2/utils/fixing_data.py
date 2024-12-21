import pandas as pd

import requests
from bs4 import BeautifulSoup

dict_sector = {'Industrial Machinery' : 'Industrial Machinery & Supplies & Components',
               "Paper Packaging":"Paper & Plastic Packaging Products & Materials",
               "Semiconductor Equipment":"Semiconductor Materials & Equipment",
               "Construction Machinery & Heavy Trucks":"Construction Machinery & Heavy Transportation Equipment",
               "Soft Drinks":"Soft Drinks & Non-alcoholic Beverages",
               "Personal Products":"Personal Care Products",
               "Metal & Glass Containers":"Metal, Glass & Plastic Containers",
               "Auto Parts & Equipment":"Automotive Parts & Equipment",
               "Pharmaceuticals, Biotechnology & Life Sciences":"Pharmaceuticals, Biotechnology  & Life Sciences",
               "Internet & Direct Marketing Retail":"Consumer Discretionary",
               "Department Stores":"Consumer Discretionary",
               "Thrifts & Mortgage Finance":"Commercial & Residential Mortgage Finance",
               "General Merchandise Stores":"Consumer Staples Merchandise Retail"}

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
    
    df.iloc[118:131,0] = 'Information Technology'
    df.iloc[118:123,1] = 'Software & Services'
    
    return df


def subindustry_classification(subindustry: str, df:pd.DataFrame) -> str:
    # Returns a list with the category of the subindustry
    for column in df.columns:
        if subindustry in df[column].values:
            sector = df[df[column] == subindustry]['Sector'].values[0]
            break
        else:
            sector = None
    return sector


def reduce_sector_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    data_subindustry = get_data()
    df = fix_sector(df)
    df["new_sector"] = df.apply(lambda x: subindustry_classification(x['sector'], data_subindustry), axis=1)
    return df