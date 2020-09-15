import boto3
import xmltodict
import pandas as pd
import sys

def update_qualification(filename):
    
    workers = pd.read_csv(filename, index_col="workerId")  
    for index, worker in workers.iterrows():
        print(index)
        response = client.associate_qualification_with_worker(
            QualificationTypeId='',
            WorkerId=index,
            IntegerValue=1,
            SendNotification=False
        )

if __name__ == "__main__":
    filename = sys.argv[1]

    client = boto3.client('mturk', region_name='us-east-1')
    update_qualification(filename)