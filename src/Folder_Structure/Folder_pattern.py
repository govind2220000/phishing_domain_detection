import os
from pathlib import Path

class Folder_Structure_Generator:
    
    def folder_creator(self):

        if not os.path.exists("src/After_Filename_Validation/Good_Raw_Data"):
            os.makedirs("src/After_Filename_Validation/Good_Raw_Data")
            
        if not os.path.exists("src/After_Filename_Validation/Bad_Raw_Data"):
            os.makedirs("src/After_Filename_Validation/Bad_Raw_Data")
            
            
            
            
if __name__ == "__main__":
    folder_creator = Folder_Structure_Generator()
    folder_creator.folder_creator()