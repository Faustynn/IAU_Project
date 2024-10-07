import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

os.makedirs('datasets/024-ver1', exist_ok=True)
os.makedirs('datasets/024-ver2', exist_ok=True)



# 1.1
def A1b(data):
    for name, df in data:
        print(f'"{name}" dataset has {df.shape[0]} rows and {df.shape[1]} columns')

        print("\nDataFrame info:")
        df.info()

        print("\nDataFrame describe:")
        print(df.describe())

        missing_values = df.isnull().sum()
        print("\nMissing data in attributes:")
        print(missing_values[missing_values > 0])

        print("##################################################")
def B1b(data, atributes_arr):
    for name, df in data:
        print(f'Visualisation "{name}" dataset')

        if not atributes_arr:
            continue

        selected_columns = [col for col in atributes_arr if col in df.columns]
        if not selected_columns:
            print("Error: specified attributes in the dataset!")
            continue

        for col in selected_columns:
            print(f"\nAnalysis for {col}")
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], bins=30, kde=True)

            median = df[col].median()
            mode = df[col].mode()[0]
            mean = df[col].mean()
            plt.axvline(median, color='r', linestyle=':', label=f'Median: {median}')
            plt.axvline(mode, color='b', linestyle='-.', label=f'Mode: {mode}')
            plt.axvline(mean, color='y', linestyle='-', label=f'Mean: {mean}')

            plt.title(f'Histogram Plot for {col} in {name}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid()
            plt.savefig(f'plots/B/histplot_{name}_{col}.png')

            plt.show()
        print("##################################################")
def C1b(data):
    for name, df in data:
        print(f'Pairing Analysis for "{name}" dataset')

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns if 'imei' not in col]

        if numeric_columns:
            corr_matrix = df[numeric_columns].corr()
            if len(numeric_columns) > 1:
                plt.figure(figsize=(25, 18))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.axhline(0, color='black', linewidth=1)
                plt.axvline(0, color='black', linewidth=1)
                plt.title(f'Pearson Correlation Heatmap for {name}')
                plt.savefig(f'plots/C/pearson_heat_map_{name}.png')
                plt.show()
        else:
            print(f"No numeric columns to analyze for {name}")

        #  TO DO
        # нам нужен анализ для обектов (текстовых данных)?

        print("##################################################")
def D1b(predicant, predictor, data):
    for name, df in data:
        print(f'"{name}" dataset: Correlation Analysis for {predicant} and {predictor}')

        if predicant not in df.columns or predictor not in df.columns:
            continue

        # сorr calculate
        correlation = df[[predicant, predictor]].corr().iloc[0, 1]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=predicant, y=predictor, s=50, alpha=0.5, edgecolor=None)
        sns.regplot(data=df, x=predicant, y=predictor, scatter=False, color='red', line_kws={"lw":2})
        plt.title(f'Scatter Plot and Regression Line for {predicant} and {predictor} in {name}')
        plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes,fontsize=12, color='red', verticalalignment='top')
        plt.grid(True)
        plt.savefig(f'plots/D/scatterplot_{name}_{predicant}_{predictor}.png')
        plt.show()

        print("##################################################")
def E1b():
    print("В ходе работы над проектом я заметим связь между некоторыми атрибутами в датасетах. Например, в датасете Connections есть сильная корреляция между c.dogalize и c.android.gm, а также между c.android.gm и c.katana. В датасете Processes есть сильная корреляция между p.system и p.android.gm. Эти корреляции могут быть использованы для дальнейшего анализа данных и построения моделей.")
    print("В Processes датасете есть атрибуты, которые имеют сильную корреляцию, например, p.system и p.android.gm. Это может быть связано с тем, что эти атрибуты взаимосвязаны и могут быть использованы для прогнозирования друг друга.")
    print("В Devices и Profiles только object & string атрибуты, поэтому корреляция не может быть рассчитана, также устроиства никак не взаимосвязаны между собой.")
    print("Прогнозирумая переменая зависит от таких атрибов как c.dogalize, c.android.gm, c.katana, p.system, p.android.gm потому что они имеют сильную корреляцию с прогнозируемой переменной. ( часть D1b)")
    print("На мое мнение обединение датасетов иммет смысл так как все они имеют общий атрибут imei, который может быть использован для объединения данных. Это позволит нам получить более полную картину и провести более детальный анализ данных.")

# 1.2
# method reformat_file i see in this site delftstack.com
def reformat_file(input_file, output_file):
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.truncate(0)

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        headers = lines[0].strip().split('\t')
        column_widths = [len(header) for header in headers]

        for line in lines[1:]:
            row = line.strip().split('\t')
            for i, col in enumerate(row):
                column_widths[i] = max(column_widths[i], len(col))

        formatted_header = "\t".join([headers[i].ljust(column_widths[i]) for i in range(len(headers))])
        outfile.write(formatted_header + '\n')

        for line in lines[1:]:
            row = line.strip().split('\t')
            formatted_row = "\t".join([row[i].ljust(column_widths[i]) for i in range(len(row))])
            outfile.write(formatted_row + '\n')

def clean_data_ver1(new_data_file, valid_format):
    df = pd.read_csv(new_data_file, sep='\t')

    # delete Nan values
    df.dropna(axis=0, how='any', inplace=True)
    df.dropna(axis=1, how='any', inplace=True)

    # validate format
    for col, col_type in valid_format.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(col_type)
            except ValueError:
                print(f"Error converting column {col} to {col_type}. Dropping the column.")
                df = df.drop(columns=[col])
            else:
                invalid_rows = pd.to_numeric(df[col], errors='coerce').isna()
                if invalid_rows.any():
                    print(f"Removing rows with invalid format in column {col}.")
                    df = df[~invalid_rows]

    # delete duplicates
    df = df.drop_duplicates()

    # remove outliers use IQR
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # fill missing values
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Operations with categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)

    # Normalization
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df.to_csv(new_data_file, sep='\t', index=True)
    print(f"Cleaned data has been saved to {new_data_file}.")
def clean_data_ver2(new_data_file, valid_format):
    df = pd.read_csv(new_data_file, sep='\t')

    # validate format
    for col, col_type in valid_format.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(col_type)
            except ValueError:
                print(f"Error converting column {col} to {col_type}. Dropping the column.")
                df = df.drop(columns=[col])
            else:
                invalid_rows = pd.to_numeric(df[col], errors='coerce').isna()
                if invalid_rows.any():
                    print(f"Removing rows with invalid format in column {col}.")
                    df = df[~invalid_rows]

    # delete duplicates
    df = df.drop_duplicates()

    # KNN to fill in missing datas
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # remove outliers using quantiles
    for col in numeric_columns:
        lower_quantile = df[col].quantile(0.05)
        upper_quantile = df[col].quantile(0.95)
        df[col] = np.where(df[col] < lower_quantile, lower_quantile, df[col])
        df[col] = np.where(df[col] > upper_quantile, upper_quantile, df[col])

    # Operations with categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)

    # Normalization
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df.to_csv(new_data_file, sep='\t', index=False)
    print(f"Cleaned data has been saved to {new_data_file}.")
def clean_all_data(dataset_title):
    if 'Connections' in dataset_title:
        valid_format = {
            'column1': 'object', 'column2': 'int64', 'column3': 'float64', 'column4': 'float64',
            'column5': 'float64', 'column6': 'float64', 'column7': 'float64', 'column8': 'float64',
            'column9': 'float64', 'column10': 'float64', 'column11': 'float64', 'column12': 'float64',
            'column13': 'float64'
        }
    elif 'Devices' in dataset_title:
        valid_format = {
            'column1': 'float64', 'column2': 'float64', 'column3': 'object', 'column4': 'object',
            'column5': 'object', 'column6': 'object', 'column7': 'int64'
        }
    elif 'Processes' in dataset_title:
        valid_format = {
            'column1': 'object', 'column2': 'int64', 'column3': 'float64', 'column4': 'float64',
            'column5': 'float64', 'column6': 'float64', 'column7': 'float64', 'column8': 'float64',
            'column9': 'float64', 'column10': 'float64', 'column11': 'float64', 'column12': 'float64',
            'column13': 'float64', 'column14': 'float64', 'column15': 'float64', 'column16': 'float64',
            'column17': 'float64', 'column18': 'float64', 'column19': 'float64', 'column20': 'float64',
            'column21': 'float64', 'column22': 'float64', 'column23': 'float64'
        }
    elif 'Profiles' in dataset_title:
        valid_format = {
            'column1': 'object', 'column2': 'object', 'column3': 'object', 'column4': 'object',
            'column5': 'object', 'column6': 'object', 'column7': 'int64', 'column8': 'int64',
            'column9': 'object', 'column10': 'object', 'column11': 'object', 'column12': 'object'
        }
    else:
        print("Unknown dataset")
        return

    file_path_ver1 = 'datasets/024-ver1/' + dataset_title.lower() + '.csv'
    file_path_ver2 = 'datasets/024-ver2/' + dataset_title.lower() + '.csv'
    original_file_path = '024/' + dataset_title.lower() + '.csv'

    if os.path.exists(original_file_path) and os.path.getsize(original_file_path) > 0:
        reformat_file(original_file_path, file_path_ver1)

        clean_data_ver1(file_path_ver1, valid_format)
        reformat_file(original_file_path, file_path_ver1)

        reformat_file(original_file_path, file_path_ver2)

        clean_data_ver2(file_path_ver2, valid_format)
        reformat_file(original_file_path, file_path_ver2)

        print(f"Reformation {dataset_title} was successful!")
    else:
        print(f"Error: {original_file_path} is empty or does not exist.")
    print("##################################################")

# 1.3