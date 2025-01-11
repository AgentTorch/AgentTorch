import dask.dataframe as dd
import pandas as pd
data = dd.read_parquet("/home/hardik/Desktop/AgentTorch/agent_torch/populations/sample/area.parquet")  # Combines all files into one DataFrame
data = data.compute()  # Converts Dask DataFrame to Pandas DataFrame

df= pd.read_pickle("/home/hardik/Desktop/AgentTorch/agent_torch/populations/sample/area.pickle")
print(type(df))
print(data.values)