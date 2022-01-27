import boto3
import os
import configparser


class s3_Handler:
    
    def s3_client_initialize(self, access_key, secret_key):
        
        self.s3_client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key)
        return self.s3_client
    
    def s3_bucket_upload_dir(self,client_obj,file_path,bucket_name='Phishing'):
        """
        Uploads multiple file to an s3 bucket.
        """
        
        for i in os.listdir(file_path):
            print("Uploading file: " + i)
            upload_file_path = 'phishing/'+ i
            client_obj.upload_file(file_path+i, bucket_name, upload_file_path)
            print("Uploaded file: " + i + " to bucket: " + bucket_name + " with location: " + upload_file_path)
            
    def s3_bucket_upload_file(self,client_obj,file_path,file_name,bucket_name='phishing'):
        """
        Uploads a single file to an s3 bucket.
        """
        print("Uploading file: " + file_path)
        upload_file_path = 'phishing/'+ file_name
        client_obj.upload_file(file_path, bucket_name, upload_file_path)
        print("Uploaded file: " + file_path + " to bucket: " + bucket_name + " with location: " + upload_file_path)
    
            
    def list_buckets_contents(self,client_obj,bucket_name='phishing'):
        """
        Lists the contents of an s3 bucket.
        """
        response = client_obj.list_objects(Bucket=bucket_name)
        for i in response['Contents']:
            print(i['Key'])
            
    def list_folder_contents(self,client_obj,bucket_name='phishing',folder_name='phishing/'):
        """
        Lists the contents of an s3 folder.
        """
        paginator = client_obj.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_name)
        file_list = []
        for page in pages:
            for i in page['Contents']:
                file_list.append(i['Key'])
        return file_list
    
    def download_file_(self,client_obj,file_name,file_save_path,bucket_name='phishing'):
        """
        Downloads a file from an s3 bucket.
        """
        file_save_path = f"{file_save_path}/{file_name[9:]}"
        client_obj.download_file(bucket_name, file_name, file_save_path)
        
    def delete_file(self,client_obj,file_name,bucket_name='phishing'):
        """
        Deletes a file from an s3 bucket.
        """
        client_obj.delete_object(Bucket=bucket_name, Key=file_name)
        
    def delete_folder(self,client_obj,folder_name,bucket_name='phishing'):
        """
        Deletes a folder from an s3 bucket.
        """
        paginator = client_obj.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_name)
        for page in pages:
            for i in page['Contents']:
                client_obj.delete_object(Bucket=bucket_name, Key=i['Key'])
                
    def delete_bucket(self,client_obj,bucket_name='phishing'):
        """
        Deletes an s3 bucket.
        """
        client_obj.delete_bucket(Bucket=bucket_name)
        
    def create_bucket(self,client_obj,bucket_name='phishing'):
        """
        Creates an s3 bucket.
        """
        client_obj.create_bucket(Bucket=bucket_name)
        
    def list_buckets(self,client_obj):
        """
        Lists all the s3 buckets.
        """
        response = client_obj.list_buckets()
        for i in response['Buckets']:
            print(i['Name'])
                   
            
if __name__ == '__main__':
    
    
    config = configparser.ConfigParser()
    config.read('src\config.ini')
    access_key = config['s3_bucket']['access_key']
    secret_key = config['s3_bucket']['secret_key']
    cv = s3_Handler()

    client = cv.s3_client_initialize(access_key, secret_key)
    #cv.s3_bucket_upload_file(client,file_path="G:\\Phishing_Domain_Detection\\src\\S3_Bucket_Handler\\122.csv",file_name = '122.csv',bucket_name='phishingdomain')
    #cv.s3_bucket_upload(client,"G:\\Phishing_Domain_Detection\\src\\Raw_Data\\",bucket_name='phishingdomain')
    #cv.delete_file(client,'122.csv',bucket_name='phishingdomain') #needs permission
    
    #cv.list_buckets_contents(client,"phishingdomain")
    file_list = cv.list_folder_contents(client,"phishingdomain","phishing/")
    print(file_list[0])
    #cv.delete_file(client,file_list[-1],"phishingdomain")
    for i in range(len(file_list)):
        cv.download_file_(client,file_name = file_list[i],file_save_path="G:/Phishing_Domain_Detection/src/Raw_Data",bucket_name='phishingdomain')
    