import pandas as pd

def stats_df(category, target, name):
    rows = []
    for cat in sorted(set(category)):
        rows.append({
            "split": name,
            "type": "category",
            "value": cat,
            "share": round((category == cat).sum() / len(category), 3),
        })
    for cat in sorted(set(target)):
        rows.append({
            "split": name,
            "type": "target",
            "value": cat,
            "share": round((target == cat).sum() / len(target), 3),
        })
    return pd.DataFrame(rows)