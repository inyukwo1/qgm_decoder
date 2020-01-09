import sys
import MySQLdb

HOST = 'localhost'
USER = 'root'
PASSWD = 'root'

db = sys.argv[1]
fname = sys.argv[2]

if __name__ == '__main__':
    dbconn = MySQLdb.connect(host=HOST, user=USER, passwd=PASSWD)
    try:
        cu = dbconn.cursor()
        with open(fname, 'w') as f:
            cu.execute('USE ' + db)
            cu.execute('SHOW TABLES')
            all_tables = cu.fetchall()
            for table in all_tables:
                table_name = table[0]
                cu.execute('DESC ' + table_name)
                all_columns = cu.fetchall()
                for column in all_columns:
                    column_name = column[0].upper()
                    f.write('{} {}\n'.format(table_name.upper(), column_name))
        cu.close()
    except Exception, e:
        print e
    dbconn.close()

