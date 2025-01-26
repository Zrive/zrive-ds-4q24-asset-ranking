```python
import pandas as pd
from utils.functions import transform_daily_data_to_quarterly_data
```


```python
capital_expenditure = pd.read_csv('./data/initial_data/Reto_SP500_CapitalExpenditure.csv')
cash_and_quivalents = pd.read_csv('./data/initial_data/Reto_SP500_CashAndEquivalents.csv')
diluted_shares = pd.read_csv('./data/initial_data/Reto_SP500_DilutedShares.csv')
ebit = pd.read_csv('./data/initial_data/Reto_SP500_EBIT.csv')
ebitda = pd.read_csv('./data/initial_data/Reto_SP500_EBITDA.csv')
prices = pd.read_csv('./data/initial_data/Reto_SP500_Prices.csv')
net_income = pd.read_csv('./data/initial_data/Reto_SP500_NetIncome.csv')
sectors = pd.read_csv('./data/initial_data/Reto_SP500_Sectors.csv')
survivor = pd.read_csv('./data/initial_data/Reto_SP500_Survivor.csv')
total_assets = pd.read_csv('./data/initial_data/Reto_SP500_TotalAssets.csv')
total_debt = pd.read_csv('./data/initial_data/Reto_SP500_TotalDebt.csv')
total_revenues = pd.read_csv('data/initial_data/Reto_SP500_TotalRevenues.csv')
sp500 = pd.read_csv('data/initial_data/SP500_TR.csv')
```


```python
sp500.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>SP500</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>1784.956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>1764.302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>1758.067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>1764.626</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>1762.123</td>
    </tr>
  </tbody>
</table>
</div>



### Understanding of how the data sets works and possible modifications


```python
prices.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>12861</th>
      <th>12911</th>
      <th>12913</th>
      <th>12924</th>
      <th>12995</th>
      <th>12997</th>
      <th>13000</th>
      <th>13011</th>
      <th>13058</th>
      <th>...</th>
      <th>193496</th>
      <th>748463</th>
      <th>748466</th>
      <th>126340</th>
      <th>128536</th>
      <th>134017</th>
      <th>147902</th>
      <th>751051</th>
      <th>141326</th>
      <th>195528</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>0.954408</td>
      <td>30.839182</td>
      <td>18.75962</td>
      <td>37.410706</td>
      <td>18.081195</td>
      <td>12.335083</td>
      <td>17.00</td>
      <td>45.566068</td>
      <td>20.265429</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.420443</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>0.964210</td>
      <td>30.024336</td>
      <td>18.05964</td>
      <td>34.981959</td>
      <td>17.314278</td>
      <td>12.100408</td>
      <td>15.80</td>
      <td>44.843022</td>
      <td>19.927993</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.576765</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>0.972654</td>
      <td>29.859367</td>
      <td>17.70965</td>
      <td>35.251820</td>
      <td>17.016533</td>
      <td>11.895068</td>
      <td>15.80</td>
      <td>44.134153</td>
      <td>19.773737</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.554825</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>0.973409</td>
      <td>29.364460</td>
      <td>17.63965</td>
      <td>35.081907</td>
      <td>16.845105</td>
      <td>11.807065</td>
      <td>15.32</td>
      <td>44.545297</td>
      <td>19.725532</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.768740</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>1.044284</td>
      <td>29.384456</td>
      <td>17.49965</td>
      <td>34.282320</td>
      <td>16.845105</td>
      <td>11.843733</td>
      <td>14.72</td>
      <td>44.637450</td>
      <td>19.696609</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.648070</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 939 columns</p>
</div>




```python
prices_reshape = pd.melt(prices, id_vars=["date"], var_name="asset_num", value_name="price")
```


```python
prices_reshape.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>0.954408</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>0.964210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>0.972654</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>0.973409</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>1.044284</td>
    </tr>
  </tbody>
</table>
</div>




```python
prices_reshape.shape
```




    (4831638, 3)




```python
sectors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12911</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12913</td>
      <td>Technology Hardware &amp; Equipment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12924</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12995</td>
      <td>Semiconductors</td>
    </tr>
  </tbody>
</table>
</div>



### Now that we understand the data sets and the transformations we replicate that in all the data sets excluding "sectors" and "S&P500"


```python
capital_expenditure_reshape = pd.melt(capital_expenditure, id_vars=["date"], var_name="asset_num", value_name="cap_ex")
cash_and_quivalents_reshape = pd.melt(cash_and_quivalents, id_vars=["date"], var_name="asset_num", value_name="cash")
diluted_shares_reshape = pd.melt(diluted_shares, id_vars=["date"], var_name="asset_num", value_name="dil_shares")
ebit_reshape = pd.melt(ebit, id_vars=["date"], var_name="asset_num", value_name="ebit")
ebitda_reshape = pd.melt(ebitda, id_vars=["date"], var_name="asset_num", value_name="ebitda")
prices_reshape = pd.melt(prices, id_vars=["date"], var_name="asset_num", value_name="price")
net_income_reshape = pd.melt(net_income, id_vars=["date"], var_name="asset_num", value_name="net_inc")
survivor_reshape =pd.melt(survivor, id_vars=["date"], var_name="asset_num", value_name="survivor")
total_assets_reshape = pd.melt(total_assets, id_vars=["date"], var_name="asset_num", value_name="T_assets")
total_debt_reshape = pd.melt(total_debt, id_vars=["date"], var_name="asset_num", value_name="T_debt")
total_revenues_reshape = pd.melt(total_revenues, id_vars=["date"], var_name="asset_num", value_name="T_rev")
```


```python
print(capital_expenditure_reshape.shape,
cash_and_quivalents_reshape.shape ,
diluted_shares_reshape.shape,
ebit_reshape.shape ,
ebitda_reshape.shape ,
prices_reshape.shape ,
net_income_reshape.shape,
survivor_reshape.shape,
total_assets_reshape.shape ,
total_debt_reshape.shape ,
total_revenues_reshape.shape )
```

    (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3) (4831638, 3)
    


```python
total_revenues_reshape.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>T_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Merging all the melted data sets in one common data set called merged_data


```python
datasets = [capital_expenditure_reshape, cash_and_quivalents_reshape, diluted_shares_reshape, ebit_reshape, ebitda_reshape,net_income_reshape,survivor_reshape, total_assets_reshape,total_debt_reshape,total_revenues_reshape]


merged_data = prices_reshape

# Iteratively merge the rest
for data in datasets:
    merged_data = pd.merge(merged_data, data, on=["date","asset_num"])

merged_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>price</th>
      <th>cap_ex</th>
      <th>cash</th>
      <th>dil_shares</th>
      <th>ebit</th>
      <th>ebitda</th>
      <th>net_inc</th>
      <th>survivor</th>
      <th>T_assets</th>
      <th>T_debt</th>
      <th>T_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>0.954408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>0.964210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>0.972654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>0.973409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>1.044284</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_data.shape
```




    (4831638, 13)




```python
sectors.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12911</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12913</td>
      <td>Technology Hardware &amp; Equipment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12924</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12995</td>
      <td>Semiconductors</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_data.dtypes
```




    date           object
    asset_num      object
    price         float64
    cap_ex        float64
    cash          float64
    dil_shares    float64
    ebit          float64
    ebitda        float64
    net_inc       float64
    survivor         bool
    T_assets      float64
    T_debt        float64
    T_rev         float64
    dtype: object



We discover that "asset_num" is type object so we want to change it into type int


```python
merged_data["asset_num"] = merged_data["asset_num"].astype(int)
```

### Now that the type changed we can merge the data set with the sectors data set


```python
merged_data = pd.merge(merged_data, sectors,how="left", left_on="asset_num", right_on="id")
```


```python
merged_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>price</th>
      <th>cap_ex</th>
      <th>cash</th>
      <th>dil_shares</th>
      <th>ebit</th>
      <th>ebitda</th>
      <th>net_inc</th>
      <th>survivor</th>
      <th>T_assets</th>
      <th>T_debt</th>
      <th>T_rev</th>
      <th>id</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>0.954408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>0.964210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>0.972654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>0.973409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>1.044284</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12861</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
  </tbody>
</table>
</div>



We found that "id" is duplicated so we drop it 


```python
merged_data = merged_data.drop(columns=["id"])
```


```python
merged_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>price</th>
      <th>cap_ex</th>
      <th>cash</th>
      <th>dil_shares</th>
      <th>ebit</th>
      <th>ebitda</th>
      <th>net_inc</th>
      <th>survivor</th>
      <th>T_assets</th>
      <th>T_debt</th>
      <th>T_rev</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>0.954408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>0.964210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>0.972654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>0.973409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>1.044284</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_data["date"] = pd.to_datetime(merged_data["date"])
```


```python
merged_data.dtypes
```




    date          datetime64[ns]
    asset_num              int32
    price                float64
    cap_ex               float64
    cash                 float64
    dil_shares           float64
    ebit                 float64
    ebitda               float64
    net_inc              float64
    survivor                bool
    T_assets             float64
    T_debt               float64
    T_rev                float64
    sector                object
    dtype: object




```python
merged_data["quarter"] = merged_data["date"].dt.to_period("Q")
```

# Explanation of what we want to do with each column of the data frame:

### First of all we groupby quarter and asset

We will take always the last non null value of the quarter. 

We believe that the latest information of the quarter is the one with more added value for our model.

price

cap_ex 

cash 

dil_shares 

ebit 

ebitda 

net_inc

survivor 

T_assets 

T_debt 

T_rev 

sector 



```python
merged_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>asset_num</th>
      <th>price</th>
      <th>cap_ex</th>
      <th>cash</th>
      <th>dil_shares</th>
      <th>ebit</th>
      <th>ebitda</th>
      <th>net_inc</th>
      <th>survivor</th>
      <th>T_assets</th>
      <th>T_debt</th>
      <th>T_rev</th>
      <th>sector</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005-01-03</td>
      <td>12861</td>
      <td>0.954408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>2005Q1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005-01-04</td>
      <td>12861</td>
      <td>0.964210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>2005Q1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005-01-05</td>
      <td>12861</td>
      <td>0.972654</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>2005Q1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005-01-06</td>
      <td>12861</td>
      <td>0.973409</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>2005Q1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-07</td>
      <td>12861</td>
      <td>1.044284</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22515.239936</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
      <td>2005Q1</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_data_quarterly = transform_daily_data_to_quarterly_data(merged_data)
```


```python
merged_data_quarterly.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quarter</th>
      <th>asset_num</th>
      <th>price</th>
      <th>cap_ex</th>
      <th>cash</th>
      <th>dil_shares</th>
      <th>ebit</th>
      <th>ebitda</th>
      <th>net_inc</th>
      <th>survivor</th>
      <th>T_assets</th>
      <th>T_debt</th>
      <th>T_rev</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005Q1</td>
      <td>12861</td>
      <td>1.256760</td>
      <td>-190.000</td>
      <td>2475.000</td>
      <td>22880.814320</td>
      <td>678.000</td>
      <td>825.300</td>
      <td>508.000</td>
      <td>True</td>
      <td>9362.000</td>
      <td>0.000</td>
      <td>9763.000</td>
      <td>Technology Hardware, Storage &amp; Peripherals</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2005Q1</td>
      <td>12883</td>
      <td>15.538408</td>
      <td>-2.231</td>
      <td>58.367</td>
      <td>90.187000</td>
      <td>259.082</td>
      <td>260.558</td>
      <td>281.445</td>
      <td>False</td>
      <td>3491.427</td>
      <td>1578.374</td>
      <td>336.082</td>
      <td>Asset Management &amp; Custody Banks</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005Q1</td>
      <td>12911</td>
      <td>33.585000</td>
      <td>-60.294</td>
      <td>336.525</td>
      <td>488.400994</td>
      <td>599.244</td>
      <td>661.568</td>
      <td>479.257</td>
      <td>True</td>
      <td>2122.810</td>
      <td>0.000</td>
      <td>1716.182</td>
      <td>Application Software</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005Q1</td>
      <td>12913</td>
      <td>13.929720</td>
      <td>-12.000</td>
      <td>88.400</td>
      <td>810.770999</td>
      <td>35.500</td>
      <td>81.200</td>
      <td>71.300</td>
      <td>True</td>
      <td>1432.900</td>
      <td>400.000</td>
      <td>891.000</td>
      <td>Technology Hardware &amp; Equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005Q1</td>
      <td>12924</td>
      <td>29.760000</td>
      <td>-40.835</td>
      <td>517.654</td>
      <td>228.000000</td>
      <td>261.573</td>
      <td>313.522</td>
      <td>221.508</td>
      <td>True</td>
      <td>1142.204</td>
      <td>0.000</td>
      <td>1233.767</td>
      <td>Application Software</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_data_quarterly.to_feather("./data/merged_data_quarterly.feather")
```

### Next steps

How to deal with the missing values in the fundamentals? 
