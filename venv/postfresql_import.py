import psycopg2
from pandas.io.sql import read_sql_query
'''
This file is designed to fetch the relevant database 
'''
cmd_access_database = "psql -h pgpool-aky-20 -U development euur_eam \ndevelopment"
cmd_pwd_access_database = "development"
host = "web-aky-21.enschede.akyla"
username = "e.mitaros"
client_pwd = "Nkpgjmdf8!"
cmd_create_merged_dataset = "select assignment.functionname, assignment.startdate, assignment.enddate, " \
                            "assignment.active, assignment.deleted, assignment.flexworkerid," \
                            "assignmentcomponent.startdate,assignmentcomponent.enddate, assignmentcomponent.wage, timecardline.starttime," \
                            "timecardline.endtime, timecardline.resttime, timecardline.amount, timecardline.linedate," \
                            " timecard.totalhours, timecard.totalexpense, payrollcomponent.description, payrollcomponenttype.description, " \
                            "flexworkerbase.flexworkertype, flexworkerbase.active, flexworkerbase.status, " \
                            "staffingcustomer.companyname, staffingcustomer.active, staffingcustomer.region, " \
                            "timecardrepresentation.description, period.description from assignment join " \
                            "assignmentcomponent on (assignment.assignmentid = assignmentcomponent.assignmentid) join " \
                            "flexworkerbase on (assignment.flexworkerid = flexworkerbase.flexworkerid) join " \
                            "timecardline on (assignmentcomponent.assignmentcomponentid = timecardline.assignmentcomponentid) " \
                            "join timecard on (timecardline.timecardid = timecard.timecardid) join payrollcomponent on " \
                            "(assignmentcomponent.payrollcomponentid = payrollcomponent.payrollcomponentid) join " \
                            "payrollcomponenttype on (payrollcomponent.payrollcomponenttypeid = payrollcomponenttype.payrollcomponenttypeid) " \
                            "join timecardrepresentation on (timecard.timecardrepresentationid = timecardrepresentation.timecardrepresentationid) " \
                            "join period on (timecard.periodid = period.periodid) join staffingcustomer on (assignment.staffingcustomerid = staffingcustomer.staffingcustomerid) where timecardline.periodtotalline != '1'::bit;"


def fetch_postgresql_database():
    conn = psycopg2.connect(dbname='euur_eam', user='development', password='development', host="pgpool-aky-20.enschede.akyla")
    db = read_sql_query(cmd_create_merged_dataset, conn)
    conn.close()
    return db
