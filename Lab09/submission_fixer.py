import pandas as pd

input_filename = "submission.csv"
output_filename = "fixed_submission.csv"

df = pd.read_csv(input_filename)
df["ID"] = df["ID"] - 1
df.to_csv(output_filename, index=False)

print(f"File saved as '{output_filename}' with updated indexing.")
