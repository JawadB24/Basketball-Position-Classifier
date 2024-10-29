from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns



def latest_df():
    
    driver = webdriver.Chrome()
    
    url = "https://www.sports-reference.com/cbb/conferences/big-12/men/2024-stats.html"
    
    driver.get(url)
    
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "players_per_game_conf")))
    
        page_source = driver.page_source
    
        soup = BeautifulSoup(page_source, "html.parser")
        
    
        table = soup.find("table", id = "players_per_game_conf")
            
        headers = table.find_all("th", attrs = {"aria-label": True})	
        
        columns = []
        
        for column_header in headers:
            
            header = column_header.get("aria-label")
            
            columns.append(header)
        
        columns = columns[3:-1]
        
        rows = []
            
        column_data = table.find_all("tr")
        
        individual_row = []
        
        for row in column_data:
            
            row_data = row.find_all("td")
            
            for data in row_data:
                            
                individual_row.append(data.get_text())
                            
            rows.append(individual_row[2:-1])
            
            individual_row = []
        
        rows = rows[1:-1]
        
    
        df = pd.DataFrame(rows, columns = columns)
            
        
    finally:
        driver.quit()
        
    return df
        
def older_df():
    
    driver = webdriver.Chrome()
    
    url = "https://www.sports-reference.com/cbb/conferences/big-12/men/2023-stats.html"
    
    driver.get(url)
    
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "players_per_game_conf")))
    
        page_source = driver.page_source
    
        soup = BeautifulSoup(page_source, "html.parser")
    
        table = soup.find("table", id = "players_per_game_conf")
            
        headers = table.find_all("th", attrs = {"aria-label": True})	
        
        columns = []
        
        for column_header in headers:
            
            header = column_header.get("aria-label")
            
            columns.append(header)
        
        columns = columns[3:-1]
        
        rows = []
            
        column_data = table.find_all("tr")
        
        individual_row = []
        
        for row in column_data:
            
            row_data = row.find_all("td")
            
            for data in row_data:
                            
                individual_row.append(data.get_text())
                            
            rows.append(individual_row[2:-1])
            
            individual_row = []
        
        rows = rows[1:-1]
        
    
        df = pd.DataFrame(rows, columns = columns)
        
        
    finally:
        driver.quit()
    
    return df


def oldest_df():
    
    driver = webdriver.Chrome()
    
    url = "https://www.sports-reference.com/cbb/conferences/big-12/men/2022-stats.html"
    
    driver.get(url)
    
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "players_per_game_conf")))
    
        page_source = driver.page_source
    
        soup = BeautifulSoup(page_source, "html.parser")
    
        table = soup.find("table", id = "players_per_game_conf")
            
        headers = table.find_all("th", attrs = {"aria-label": True})	
        
        columns = []
        
        for column_header in headers:
            
            header = column_header.get("aria-label")
            
            columns.append(header)
        
        columns = columns[3:-1]
        
        rows = []
            
        column_data = table.find_all("tr")
        
        individual_row = []
        
        for row in column_data:
            
            row_data = row.find_all("td")
            
            for data in row_data:
                            
                individual_row.append(data.get_text())
                            
            rows.append(individual_row[2:-1])
            
            individual_row = []
        
        rows = rows[1:-1]
        
    
        df = pd.DataFrame(rows, columns = columns)
                    
        
    finally:
        driver.quit()
        
    return df
        
def complete_dataframe() -> str:
    
    df1 = latest_df()
    df2 = older_df()
    df3 = oldest_df()
    
    complete_df = pd.concat([df1, df2, df3], ignore_index = True)
    
    ohe = OneHotEncoder(sparse_output = False).set_output(transform = "pandas")
    
    temp = complete_df[["Pos"]]
    
    ohe_transform = ohe.fit_transform(complete_df[["Pos"]])
            
    complete_df = pd.concat([complete_df.drop(columns = ["Pos"]), ohe_transform], axis = 1)
            
    complete_df = complete_df.apply(pd.to_numeric, errors = "coerce")
    
    complete_df.fillna(complete_df.mean(), inplace = True)
    
    complete_df = pd.concat([complete_df, temp], axis = 1)
    
    complete_df = complete_df[complete_df["MP"] >= 15]
    
    complete_df = complete_df.reset_index(drop = True)
    
    print(complete_df)
        
    complete_df.to_csv("basketball_data.csv")
    
    return "basketball_data.csv"

def position_classifier(csv_file: str):
    """
    
    """
    
    df = pd.read_csv(csv_file)
        
    y = df[["Pos_G", "Pos_F", "Pos_C"]]
    
    x = df.drop(["Pos", "Pos_G", "Pos_F", "Pos_C", "PF"], axis = 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
    
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    logistic_model_G = linear_model.LogisticRegression(solver = "lbfgs", max_iter = 1000)
    logistic_model_F = linear_model.LogisticRegression(solver = "lbfgs", max_iter = 1000)
    logistic_model_C = linear_model.LogisticRegression(solver = "lbfgs", max_iter = 1000)
    
    logistic_model_G.fit(x_train_scaled, y_train["Pos_G"])
    logistic_model_F.fit(x_train_scaled, y_train["Pos_F"])
    logistic_model_C.fit(x_train_scaled, y_train["Pos_C"])
    
    pred_G = logistic_model_G.predict(x_test_scaled)
    pred_F = logistic_model_F.predict(x_test_scaled)
    pred_C = logistic_model_C.predict(x_test_scaled)
    
    accuracy_G = logistic_model_G.score(x_test_scaled, y_test["Pos_G"])
    accuracy_F = logistic_model_F.score(x_test_scaled, y_test["Pos_F"])
    accuracy_C = logistic_model_C.score(x_test_scaled, y_test["Pos_C"])
    
    print("Guard Classification Accuracy: " + str(accuracy_G))
    print("Forward Classification Accuracy: " + str(accuracy_F))
    print("Center Classification Accuracy: " + str(accuracy_C))
    
    print("Classification Report for Pos_G:")
    print(classification_report(y_test['Pos_G'], pred_G))
    
    print("Classification Report for Pos_F:")
    print(classification_report(y_test['Pos_F'], pred_F))
    
    print("Classification Report for Pos_C:")
    print(classification_report(y_test['Pos_C'], pred_C))
    
    cm_G = confusion_matrix(y_test["Pos_G"], pred_G)    
    cm_F = confusion_matrix(y_test["Pos_F"], pred_F)    
    cm_C = confusion_matrix(y_test["Pos_C"], pred_C)
    
    fig, axes = plt.subplots(1, 3)
    
    sns.heatmap(cm_G, annot = True, ax = axes[0], cmap = "Blues")
    axes[0].set_title("Basketball Guard Confusion Matrix")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Actual Values")
    
    sns.heatmap(cm_F, annot = True, ax = axes[1], cmap = "Reds")
    axes[1].set_title("Basketball Forward Confusion Matrix")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Actual Values")
    
    sns.heatmap(cm_C, annot = True, ax = axes[2], cmap = "Greens")
    axes[2].set_title("Basketball Center Confusion Matrix")
    axes[2].set_xlabel("Predicted Values")
    axes[2].set_ylabel("Actual Values")     
    
    plt.tight_layout()
    plt.show()