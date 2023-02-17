import numpy as np
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":
    x = np.random.rand(30, 10)
    y = np.ones(30)
    y[:10] -= 1
    y[20:] += 1

    model = XGBClassifier(
        objective='multi:softproba'
    )
    model.fit(x, y)
    model.save_model('checkpoint.model')

    y0 = model.predict_proba(x[0].reshape(1, -1))
    # x[0] = [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864,
    #        0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258]
    y1 = model.predict_proba(x[10].reshape(1, -1))
    # x[10] = [0.03142919, 0.63641041, 0.31435598, 0.50857069, 0.90756647,
    #        0.24929223, 0.41038292, 0.75555114, 0.22879817, 0.07697991]
    y2 = model.predict_proba(x[20].reshape(1, -1))
    # x[20] = [0.64203165, 0.08413996, 0.16162871, 0.89855419, 0.60642906,
    #        0.00919705, 0.10147154, 0.66350177, 0.00506158, 0.16080805]
    print(y0, y1, y2)


