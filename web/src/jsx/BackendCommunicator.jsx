import axios from 'axios';

const queryTextToSQL = (db_id, model, nlq, url) =>
  axios
    .get (url, {
      headers: {
        'Access-Control-Allow-Origin': '*',
      },
      crossdomain: true,
      params: {
        model: model,
        db_id: db_id,
        question: nlq,
      },
    })
    .then (response => {
      const pred_sql = response.data.result;
      const plot_filename = 'plot_filename' in response.data
        ? response.data.plot_filename
        : null;
      return [pred_sql, plot_filename];
    });

export default queryTextToSQL;
