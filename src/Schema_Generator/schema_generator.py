import json
from pandas import read_csv


# for col in df.columns:
#     print(col)
#     print(df[col].dtype)
#     print()

#print(cols)
class schema_gen:
    
    def __init__(self,df) -> None:
        self.df = df
        
    def schema_generator_training(self):
        schema = {}
        schema['Sample_Name'] = 'Phishing_2022-01-20_15-56-51.csv'
        schema["LengthOfDateStampInFile"] = 10
        schema["LengthOfTimeStampInFile"] = 8
        schema["NumberOfColumns"] = len(self.df.columns) 
        schema['ColName'] = {}
        for col in self.df.columns:
            if col not in schema['ColName']:
                if self.df[col].dtype == "int64":
                    
                    schema['ColName'].update({col: self.df[col].astype(int).dtype.name})
                else:
                    schema['ColName'].update({col: self.df[col].dtype.name})
                #schema['ColName'] = {col: df[col].dtype}
            
        return schema

    def schema_generator_prediction(self):
        schema = {}
        schema['Sample_Name'] = 'Phishing_2022-01-20_15-56-51.csv'
        schema["LengthOfDateStampInFile"] = 10
        schema["LengthOfTimeStampInFile"] = 8
        schema["NumberOfColumns"] = len(self.df.columns) - 1
        schema['ColName'] = {}
        for col in self.df.columns:
            if col not in schema['ColName']:
                if self.df[col].dtype == "int64":
                    
                    schema['ColName'].update({col: self.df[col].astype(int).dtype.name})
                else:
                    schema['ColName'].update({col: self.df[col].dtype.name})
                #schema['ColName'] = {col: df[col].dtype}
        del schema['ColName']['phishing']    
        return schema
if __name__ == '__main__':
    df = read_csv('src\Raw_Data\Phishing_2022-01-20_15-56-51.csv')
    schema_gen = schema_gen(df)
    schema_train = schema_gen.schema_generator_training()
    #print (schema)
    schema_training_json = json.dumps(schema_train, indent=4)
    #print(schema_json)
    with open("schema_training.json", "w") as outfile:
        outfile.write(schema_training_json)
        
    schema_pred = schema_gen.schema_generator_prediction()
    #print (schema)
    schema_pred_json = json.dumps(schema_pred, indent=4)
    #print(schema_json)
    with open("schema_prediction.json", "w") as outfile:
        outfile.write(schema_pred_json)

