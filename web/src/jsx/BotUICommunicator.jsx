const sqlFormatter = sql => {
  const formatted_sql = sql
    .replace ('FROM', '<br />FROM')
    .replace ('WHERE', '<br />WHERE')
    .replace ('select', 'SELECT')
    .replace ('from', '<br />FROM')
    .replace ('where', '<br />WHERE');
  // .replace ('SELECT', '<br />SELECT')
  return formatted_sql;
};

const unformatsql = formatted_sql => {
  const sql = formatted_sql.replace (/<br \/>/gi, '');
  return sql;
};

String.prototype.insert = function (index, string) {
  if (index > 0)
    return (
      this.substring (0, index) + string + this.substring (index, this.length)
    );

  return string + this;
};

const sqlFormatterWithWrongParts = (formatted_sql, wrong_parts) => {
  var sql = unformatsql (formatted_sql);
  for (var [st, ed] of wrong_parts.reverse ()) {
    sql = sql.insert (ed, '</font>');
    sql = sql.insert (st, '<font color="red">');
  }
  const re_formatted_sql = sqlFormatter (sql);
  return re_formatted_sql;
};

class BotUICommunicator {
  constructor () {
    this.botui = null;
    this.state = {
      ready_to_analyze: false,
      selected_index: 0,
    };
    this.callback = () => {};
  }

  registerBotUI (botui) {
    this.botui = botui;
  }

  registerCallback (callback) {
    this.callback = callback;
  }

  sequentialHumanBotMessage (human_messages, bot_messages) {
    return human_messages
      .reduce (this._sequenceHumanMessage, Promise.resolve ())
      .then (_ =>
        bot_messages.reduce (this._sequenceBotMessage, Promise.resolve ())
      );
  }

  sequentialInsertHumanBotMessage (insert_idx, human_messages, bot_messages) {
    return human_messages
      .reduce (this._sequenceInsertHumanMessage, Promise.resolve (insert_idx))
      .then (index =>
        bot_messages.reduce (
          this._sequenceInsertBotMessage,
          Promise.resolve (index)
        )
      );
  }

  analyzeMessage (gold_sql, pred_sql, analysis_promise) {
    if (!this.state.ready_to_analyze) {
      this.oneBotMessage (
        '<b>Please click a message box which you want to analyze.</b>'
      );
      return;
    }
    this.sequentialInsertHumanBotMessage (
      this.state.selected_index,
      [sqlFormatter (gold_sql)],
      ['processing...']
    )
      .then (_ => analysis_promise)
      .then (analysis_result => {
        const [
          correct_models,
          pred_sql_wrong_parts,
          similarity,
        ] = analysis_result;
        const recommendation = similarity === 100
          ? 'Correct model list: <br /><b>' + correct_models.join (' ') + '</b>'
          : 'There is no exactly correct model, but <b>' +
              correct_models.join (' ') +
              '</b> gives the most accurate answer(<b>' +
              similarity.toFixed (2) +
              '%</b>).';
        this.botui.message
          .update (this.state.selected_index + 1, {
            content: 'Incorrectly predicted phrases are highlighted in red.',
          })
          .then (_ =>
            this.sequentialInsertHumanBotMessage (
              this.state.selected_index + 2,
              [],
              [
                sqlFormatterWithWrongParts (pred_sql, pred_sql_wrong_parts),
                recommendation,
              ]
            )
          );
        this.state.ready_to_analyze = false;
      });
  }

  exploreMessage (nlq, promise_result) {
    this.sequentialHumanBotMessage ([nlq], ['processing...'])
      .then (_ => promise_result)
      .then (pred_sql_filename => {
        const [pred_sql, plot_filename] = pred_sql_filename;
        this.botui.message
          .remove (-1)
          .then (_ =>
            this.botui.message.bot ({
              content: sqlFormatter (pred_sql),
              delay: 100,
              add_button: true,
              toggle_callback: this.callback,
            })
          )
          .then (_ => {
            if (plot_filename) {
              this.botui.message.bot ({
                type: 'embed',
                content: plot_filename,
                delay: 1000,
              });
            } else {
              return this.oneBotMessage (
                'The generated SQL query is not executable.'
              );
            }
          });
      })
      .catch (err => {
        this.oneBotMessage (err);
      });
  }

  oneBotMessage = content =>
    this.botui.message
      .bot ({
        content: content,
        delay: 1000,
      })
      .then (this.callback);

  oneBotInsertMessage = (index, content, delay = 1000) =>
    this.botui.message
      .insert (index, {
        content: content,
        delay: delay,
      })
      .then (this.callback)
      .then (_ => index + 1);

  oneHumanMessage = content =>
    this.botui.message
      .human ({
        content: content,
        delay: 500,
      })
      .then (this.callback);

  oneHumanInsertMessage = (index, content) =>
    this.botui.message
      .insert (index, {
        content: content,
        delay: 500,
        human: true,
      })
      .then (this.callback)
      .then (_ => index + 1);

  readyAnalyze = selected_index => {
    this.state.ready_to_analyze = true;
    this.state.selected_index = selected_index;
  };

  _sequenceHumanMessage = (promise, content) => {
    return new Promise (resolve => {
      resolve (promise.then (_ => this.oneHumanMessage (content)));
    });
  };

  _sequenceInsertHumanMessage = (promise, content) => {
    return new Promise (resolve => {
      resolve (
        promise.then (index => this.oneHumanInsertMessage (index, content))
      );
    });
  };

  _sequenceBotMessage = (promise, content) => {
    return new Promise (resolve => {
      resolve (promise.then (_ => this.oneBotMessage (content)));
    });
  };

  _sequenceInsertBotMessage = (promise, content) => {
    return new Promise (resolve => {
      resolve (
        promise.then (index => this.oneBotInsertMessage (index, content))
      );
    });
  };
}

export default BotUICommunicator;
