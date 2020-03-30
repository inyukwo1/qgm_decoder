import axios from 'axios';

export const queryTextToSQL = (db_id, model, nlq, db_obj, query_cnt) =>
  axios
    .get('http://141.223.199.148:4001/service', {
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {
        model: model,
        db_id: db_id,
        question: nlq,
        db_obj: db_obj,
        query_cnt: query_cnt,
      },
    })
    .then(response => {
      const {pred_sql, execution_result} = response.data;
      return [pred_sql, execution_result];
    });

export const getDBInstance = db_id =>
  axios
    .get('http://141.223.199.148:4001/dbinstance', {
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {
        db_id: db_id,
      },
    })
    .then(response => {
      const {db_obj, db_instance_table} = response.data;
      return [db_obj, db_instance_table];
    });

export const verifyQuery = _ =>
  axios
    .get('http://141.223.199.148:4001/verify', {
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {},
    })
    .then(response => {
      const {new_db_instance, execution_result} = response.data;
      return [new_db_instance, execution_result];
    });

export default queryTextToSQL;
