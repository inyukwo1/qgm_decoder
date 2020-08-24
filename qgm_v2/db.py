class DB:
    def __init__(self, db_json):
        self.db_json = db_json

    def find_parent_table_id(self, col_id):
        return self.db_json["column_names"][col_id][1]
