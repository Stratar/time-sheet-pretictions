import psycopg2
from pandas.io.sql import read_sql_query
'''
This file is designed to fetch the relevant database 
'''


cmd_create_worker_big_dataset = "select * from timecarddata_ordered_normal_fixed_proper where timecardline_linedate > "\
                                "'2020-01-01' and assignment_flexworkerid in (select assignment_flexworkerid from " \
                                "timecarddata_ordered_normal_fixed_proper where timecardline_linedate > '2020-01-01' " \
                                "group by assignment_flexworkerid having sum(timecardline_amount)>0 order by count(*) "\
                                "desc limit 200) limit 1000000;"


def fetch_postgresql_database():
    conn = psycopg2.connect(dbname='euur_timecards', user='development', password='development',
                            host="pgpool-aky-20.enschede.akyla")
    db = read_sql_query(cmd_create_worker_big_dataset, conn)
    conn.close()
    return db
