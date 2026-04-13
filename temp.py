import pandas as pd
import numpy as np

# Tạo dữ liệu mẫu
data = {
    "student_id": [1, 2, 3, 4, 5, 6, 7],
    "gender": ["M", "F", None, "F", "M", None, "F"],
    "age": [20, 21, np.nan, 19, -2, 22, 99],
    "math_score": [8.5, 7.0, np.nan, 9.0, 50.0, 6.5, -1.0],
    "study_hours": [2.5, 3.0, 2.0, np.nan, 4.0, -3.0, 2.5]
}

df = pd.DataFrame(data)

print("=== Dữ liệu ban đầu ===")
print(df)
print("\n=== Số lượng giá trị thiếu ===")
print(df.isnull().sum())

# Thay missing values
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
df["age"] = df["age"].fillna(df["age"].median())
df["math_score"] = df["math_score"].fillna(df["math_score"].median())
df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())

# Hàm thay giá trị bất hợp lý bằng median hợp lệ
def replace_invalid_with_median(series, valid_condition):
    valid_values = series[valid_condition(series)]
    median_val = valid_values.median()
    return series.apply(lambda x: x if valid_condition(pd.Series([x])).iloc[0] else median_val)

df["age"] = replace_invalid_with_median(df["age"], lambda s: (s >= 16) & (s <= 60))
df["math_score"] = replace_invalid_with_median(df["math_score"], lambda s: (s >= 0) & (s <= 10))
df["study_hours"] = replace_invalid_with_median(df["study_hours"], lambda s: s >= 0)

print("\n=== Dữ liệu sau làm sạch ===")
print(df)