import pickle

with open("points_val.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
