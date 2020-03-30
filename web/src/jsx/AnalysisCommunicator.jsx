import axios from 'axios';

const queryAnalysis = (db_id, model, gen_sql, gold_sql, nlq, db_obj) =>
  axios
    .get('http://141.223.199.148:4001/service', {
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
        db_obj: db_obj,
      },
    })
    .then(response => {
      const pred_results = response.data.pred_results;
      const correct_systems = response.data.result;
      const diff_points = response.data.diff;
      const captum_results = response.data.captum_results;
      const analysisAgain = (model, pred) =>
        queryAnalysis(db_id, model, pred, gold_sql, nlq, db_obj);
      return [
        analysisAgain,
        correct_systems,
        pred_results,
        diff_points,
        captum_results,
      ];
    });

export default queryAnalysis;
