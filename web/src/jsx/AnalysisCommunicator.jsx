import axios from 'axios';

const queryAnalysis = (db_id, model, gen_sql, gold_sql, url) =>
  axios
    .get (url, {
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {
        mode: 'Analyze',
        model: model,
        db_id: db_id,
        gen_sql: gen_sql,
        gold_sql: gold_sql,
      },
    })
    .then (response => {
      const correct_systems = response.data.result;
      const diff_points = response.data.diff;
      return correct_systems, diff_points;
    });

export default queryAnalysis;
