import pandas as pd

# Load the dataset
file_path = "student-mat.csv" 
data = pd.read_csv(file_path)

# Perform Label Encoding
label_encoded_data = data.copy()
for col in label_encoded_data.select_dtypes(include=['object']).columns:
    label_encoded_data[col] = label_encoded_data[col].astype('category').cat.codes

# Save the Label Encoded Data to a new file
label_encoded_data.to_csv("/mnt/data/label_encoded_data.csv", index=False)

# Display the first few rows to verify
print("Label Encoded Data:")
print(label_encoded_data.head())
