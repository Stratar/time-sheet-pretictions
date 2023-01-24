import psycopg2
from pandas.io.sql import read_sql_query
'''
This file is designed to fetch the relevant database 
More queries needed for specific flexworker and staffingcustomer data generation.

Get one query for the filtered database building and then a join with the active worker variable
'''


cmd_create_worker_big_dataset = "select * from timecarddata_ordered_normal_fixed_proper where timecardline_linedate > "\
                                "'2020-01-01' and assignment_flexworkerid in (select assignment_flexworkerid from " \
                                "timecarddata_ordered_normal_fixed_proper where timecardline_linedate > '2020-01-01' " \
                                "group by assignment_flexworkerid having sum(timecardline_amount)>0 order by count(*) "\
                                "desc limit 200) limit 1000000;"

# cmd_get_flexworker_list = "select view_flexworkerid from (select * from timecardline join timecard on " \
#                           "timecardline.timecardid = timecard.timecardid) as timecards where timecards.linedate > " \
#                           "'2020-01-01' group by timecards.view_flexworkerid having sum(timecards.amount) > 0 order by " \
#                           "count(*) desc limit 200;"


cmd_get_flexworker_list = "select view_flexworkerid from (select * from timecardline join timecard on " \
                          "timecardline.timecardid = timecard.timecardid) as timecards where timecards.linedate > " \
                          "'2020-01-01' group by timecards.view_flexworkerid having sum(timecards.amount) > 0 order by " \
                          "count(*) desc limit 200;"


cmd_get_staffingcustomer_list = "Select view_staffingcustomerid from (select * from timecardline join timecard on " \
                                "timecardline.timecardid = timecard.timecardid) as timecards where " \
                                "timecards.view_flexworkerid in (select view_flexworkerid from (select * from " \
                                "timecardline join timecard on timecardline.timecardid = timecard.timecardid) as " \
                                "timecards where timecards.linedate > '2020-01-01' group by timecards.view_flexworkerid " \
                                "having sum(timecards.amount) > 0 order by count(*) desc limit 200) group by " \
                                "timecards.view_staffingcustomerid;"


# cmd_get_flex_staff_db = "Select assignment_flexworkerid, staffingcustomer_staffingcustomerid from timecardinfo_avg_active where " \
#                         "assignment_flexworkerid in (select assignment_flexworkerid from timecardinfo_avg_active where " \
#                         "timecardline_linedate > '2020-01-01' group by assignment_flexworkerid having " \
#                         "(sum(timecardline_amount) > 0 and count(*) > 50) order by count(*) desc limit 200) " \
#                         "group by assignment_flexworkerid, staffingcustomer_staffingcustomerid " \
#                         "having (sum(timecardline_amount) > 0 and count(*) > 50);"

cmd_get_flex_staff_db = "Select assignment_flexworkerid, staffingcustomer_staffingcustomerid from timecardinfo_new " \
                        "group by " \
                        "assignment_flexworkerid, staffingcustomer_staffingcustomerid " \
                        "having count(*) > 64 " \
                        "order by " \
                        "count(*) desc" \
                        "limit 1000;"

# cmd_get_full_db = "select * from timecardinfo_new;"
cmd_get_full_db = "select * from timecardinfo_new where ((assignment_flexworkerid, staffingcustomer_staffingcustomerid) " \
                  "in (select assignment_flexworkerid, staffingcustomer_staffingcustomerid from timecardinfo_new " \
                  "group by assignment_flexworkerid, staffingcustomer_staffingcustomerid having count(*) > 64 " \
                  "order by count(*) desc limit 11000));"

def connect_to_postgresql_database():
    return psycopg2.connect(dbname='euur_timecards', user='development', password='development',
                            host="pgpool-aky-20.enschede.akyla")


def fetch_postgresql_database():
    conn = connect_to_postgresql_database()
    db = read_sql_query(cmd_create_worker_big_dataset, conn)
    conn.close()
    return db


def fetch_postgresql_flexworkers():
    conn = connect_to_postgresql_database()
    flexworkers = read_sql_query(cmd_get_flexworker_list, conn)
    conn.close()
    return flexworkers


def fetch_postgresql_staffingcustomers():
    conn = connect_to_postgresql_database()
    staffingcustomer = read_sql_query(cmd_get_staffingcustomer_list, conn)
    conn.close()
    return staffingcustomer


def fetch_postgresql_flex_staff_database():
    conn = connect_to_postgresql_database()
    flex_staff_db = read_sql_query(cmd_get_flex_staff_db, conn)
    conn.close()
    return flex_staff_db


def fetch_postgresql_full_database():
    conn = connect_to_postgresql_database()
    full_db = read_sql_query(cmd_get_full_db, conn)
    conn.close()
    return full_db


def fetch_postgresql_timecards(flexworkerid, staffingcustomerid):
    cmd_create_specific_dataset = f"select * from timecardinfo_new where " \
                              f"(assignment_flexworkerid = {flexworkerid} and " \
                              f"staffingcustomer_staffingcustomerid = {staffingcustomerid}) " \
                              f"order by timecardline_linedate asc;"

    conn = connect_to_postgresql_database()
    timecard = read_sql_query(cmd_create_specific_dataset, conn)
    conn.close()
    return timecard
