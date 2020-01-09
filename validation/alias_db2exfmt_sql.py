from cosette_rollback import rewrite_sql
#from parser.db2_opt_parser import alias_sql
from parser.myParser import alias_sql
import sys

if __name__ == '__main__':
    read_file = sys.argv[1]
    write_file = sys.argv[2]

    print("\n\nSTART ALIAS")
    print("READ {}".format(read_file))
    error_msg = "SELECT CLIENT APPLNAME, CLIENT ACCTNG FROM (SELECT 'Y' FROM (VALUES 1) AS Q1 ) AS Q2"
    error_cnt = 0
    with open(read_file) as f, open(write_file, 'w') as wf:
        while True:
            line = f.readline()
            if not line: break
            query = line.strip()
            postprocessed_db2exfmt_query = ''
            alias_query = ''
            if query.strip() != error_msg:
                try:
                    postprocessed_db2exfmt_query = rewrite_sql(query)
                    alias_query = alias_sql(postprocessed_db2exfmt_query)
                except Exception as e:
                    print(repr(e))
                    postprocessed_db2exfmt_query = ''
                    alias_query = ''
            #alias_query = alias_query.replace(' .', '.')
            if alias_query == '':
                error_cnt += 1
                alias_query = postprocessed_db2exfmt_query 
            wf.write('{}\n'.format(alias_query))
            print('\n{}\n  =>\n{}\n  =>\n{}\n'.format(query, postprocessed_db2exfmt_query, alias_query))
    print('ERROR: {}'.format(error_cnt))
