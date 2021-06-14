import statsmodels.api as sm

data = sm.datasets.scotland.load(as_pandas=False)
data.exog = sm.add_constant(data.exog)
model = sm.GLM(data.endog, data.exog, family=sm.families.Gamma())
res = model.fit()
