import boto3
import numpy as np
import pandas as pd
import sys

def change_workers_results(est, emptyworker = False, accept = False):
    assignments = np.unique(est.assignmentId.values)
    for a in assignments:
        ainfo = client.get_assignment(AssignmentId=a)    
        if (ainfo["Assignment"]["AssignmentStatus"] == "Submitted" and emptyworker == False):
            response = client.reject_assignment(
                AssignmentId=a,
                RequesterFeedback="You gave contradictory answers to our personal survey. Please, contact us if you feel that we should review your results again.\
                                    We'll be glad to look into the issue together with you."
            )
        elif (ainfo["Assignment"]["AssignmentStatus"] == "Submitted" and emptyworker == True):
            response = client.reject_assignment(
                AssignmentId=a,
                RequesterFeedback="Your answers to our personal survey were not complete. Please, contact us if you feel that we should review your results again.\
                                    We'll be glad to look into the issue together with you."
            )
        elif accept == True:
            response = client.approve_assignment(
                AssignmentId=a,
                RequesterFeedback="You results were reviewed and accepted. Thank you for your work!",
                OverrideRejection=True
            )
        
        print(a, client.get_assignment(AssignmentId=a)["Assignment"]["AssignmentStatus"])

if __name__ == "__main__":
    
    client = boto3.client('mturk', region_name='us-east-1')
    
    est = pd.read_csv(sys.argv[1] + "/est_all.csv", index_col = 0)
    workers = pd.read_csv(sys.argv[1] + "/workers_all.csv", index_col = "workerId")
    
    for wid, w in workers.iterrows():
        if (w.cnt_invalid_survey > 20 and w.hit_count >= 5 and 15*w.cnt_invalid_survey >= w.hit_count):
            change_workers_results(est.loc[est.workerId == wid])