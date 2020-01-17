import axios from 'axios';

const queryAnalysis = (db_id, model, gen_sql, gold_sql, nlq, url) =>
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
        question: nlq,
      },
    })
    .then (response => {
      const correct_systems = response.data.result;
      const diff_points = response.data.diff;
      const similarity = response.data.similarity;
      return [correct_systems, diff_points, similarity];
    });

export default queryAnalysis;
