import numpy as np
import pandas as pd

def dict_recur_operate(d1, d2, f):
    if isinstance(d1, dict) and isinstance(d2, dict):
        return {k: dict_recur_operate(d1[k], d2[k], f) for k in d1}
    else:
        return f(d1, d2)  
    
def dict_recur_extract(d, f, path=()):
    if isinstance(d, dict):
        return {k: dict_recur_extract(v, f, path + (k,)) for k, v in d.items()}
    else:
        return f(d)
    
def dict_recur_value(d):
    vals = []
    for v in d.values():
        if isinstance(v, dict):
            vals.extend(dict_recur_value(v))
        else:
            vals.append(v)
    return np.array(vals)

def dict_recur_paths(d, path=()):
    paths = []
    for k, v in d.items():
        if isinstance(v, dict):
            paths.extend(dict_recur_paths(v, path + (k,)))
        else:
            paths.append(path + (k,))
    return paths

def update_view(num, ax):
    if num < 90:
        ax.view_init(elev=0, azim=90 + num)
    else:
        ax.view_init(elev=0 + (num - 90), azim=180)
    return ax

def build_valid_row_mask(df, fbg_cols, threshold=9000, half_window=5):
    """
    对一个 csv 构造“有效行”掩码。

    规则：
    - 对于某一行 i、某一列 FBGk，检查 [i-half_window, i+half_window] 范围内其他行 j。
    - 如果 |value[i] - value[j]| > threshold，
      则删除 abs(value) 更大的那一行。
    - 只要某行在任意列、任意一次比较中被判为“较大者”，该行即记为无效。
    - NaN 不参与比较。
    """
    n_rows = len(df)
    valid_row_mask = np.ones(n_rows, dtype=bool)

    col_names = [col for _, col in fbg_cols]
    values = df[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    for i in range(n_rows):
        left = max(0, i - half_window)
        right = min(n_rows, i + half_window + 1)

        for j in range(left, right):
            if j == i:
                continue

            for c in range(values.shape[1]):
                vi = values[i, c]
                vj = values[j, c]

                if np.isnan(vi) or np.isnan(vj):
                    continue

                if abs(vi - vj) > threshold:
                    # 删除 magnitude 更大的那一行
                    if abs(vi) > abs(vj):
                        valid_row_mask[i] = False
                    elif abs(vj) > abs(vi):
                        valid_row_mask[j] = False
                    else:
                        # 如果绝对值完全一样，保守起见两个都删
                        valid_row_mask[i] = False
                        valid_row_mask[j] = False

    return valid_row_mask