import pomegranate as pg
print(pg.__version__)              # you want ≥0.14
print([c for c in dir(pg) if "Bayesian" in c])
