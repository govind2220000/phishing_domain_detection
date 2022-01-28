import os
import re
import json
import shutil

class FileName_Validation:
    
    def __init__(self):
        self.regex_expression = "[Phishing]+['\_'']+[\d-]+[\d-]+[\d_]+[\d-]+[\d-]+[\d]+\.csv"
    
    def retreive_schema_values(self,path):
        with open(path, 'r') as f:
            schema_values     = json.load(f)
            splitat_          = re.split('_', schema_values['Sample_Name'])
            date_length       = schema_values['LengthOfDateStampInFile']
            time_stamp_length = schema_values['LengthOfTimeStampInFile']
            no_of_columns     = schema_values['NumberOfColumns']
        return schema_values, splitat_, date_length, time_stamp_length, no_of_columns


    

    #regex_expression = f"[{splitat_[0]}]+['\_'']+[\d-]+[\d-]+[\d_]+[\d-]+[\d-]+[\d]+\.csv"
    def filename_validation(self,date_length,time_stamp_length):
        for files in os.listdir("src\Raw_Data"):
            splitatdot = re.split(".csv", files)
            splitatunderscore = re.split("_", splitatdot[0])
            if re.match(self.regex_expression, files):
                if len(splitatunderscore[1]) == date_length and len(splitatunderscore[2]) == time_stamp_length:
                    shutil.copy(f"src\Raw_Data\{files}", f"src\After_Filename_Validation\Good_Raw_Data\{files}")
            else:
                shutil.copy(f"src\Raw_Data\{files}", f"src\After_Filename_Validation\Bad_Raw_Data\{files}")
                
if __name__ == '__main__':
    filename_validation = FileName_Validation()
    schema_dic,splitat_,date_length,time_stamp_length,no_of_columns = filename_validation.retreive_schema_values('schema_training.json')
    filename_validation.filename_validation(date_length,time_stamp_length)
