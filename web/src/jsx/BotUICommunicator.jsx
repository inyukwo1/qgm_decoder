import queryAnalysis from './AnalysisCommunicator';
import {models} from '../constants.js';

const sqlFormatter = sql => {
  const formatted_sql = sql
    .replace('FROM', '<br />FROM')
    .replace('WHERE', '<br />WHERE')
    .replace('select', 'SELECT')
    .replace('from', '<br />FROM')
    .replace('where', '<br />WHERE');
  // .replace ('SELECT', '<br />SELECT')
  return formatted_sql;
};

const unformatsql = formatted_sql => {
  const sql = formatted_sql.replace(/<br \/>/gi, '');
  return sql;
};

String.prototype.insert = function(index, string) {
  if (index > 0)
    return (
      this.substring(0, index) + string + this.substring(index, this.length)
    );

  return string + this;
};

const sqlFormatterWithWrongParts = (formatted_sql, wrong_parts) => {
  var sql = unformatsql(formatted_sql);
  for (var [st, ed] of wrong_parts.reverse()) {
    sql = sql.insert(ed, '</font>');
    sql = sql.insert(st, '<font color="red">');
  }
  const re_formatted_sql = sqlFormatter(sql);
  return re_formatted_sql;
};

class BotUICommunicator {
  constructor() {
    this.botui = null;
    this.state = {
      ready_to_analyze: false,
      selected_index: 0,
    };
    this.callback = () => {};
  }

  registerBotUI(botui) {
    this.botui = botui;
  }

  registerCallback(callback) {
    this.callback = callback;
  }

  sequentialHumanBotMessage(human_messages, bot_messages) {
    return human_messages
      .reduce(this._sequenceHumanMessage, Promise.resolve())
      .then(_ =>
        bot_messages.reduce(this._sequenceBotMessage, Promise.resolve())
      );
  }

  sequentialInsertHumanBotMessage(insert_idx, human_messages, bot_messages) {
    return human_messages
      .reduce(this._sequenceInsertHumanMessage, Promise.resolve(insert_idx))
      .then(index =>
        bot_messages.reduce(
          this._sequenceInsertBotMessage,
          Promise.resolve(index)
        )
      );
  }

  analyzeMessage(gold_sql, pred_sql, analysis_promise, redo = false) {
    if (!this.state.ready_to_analyze) {
      this.oneBotMessage(
        '<b>Please click a message box which you want to analyze.</b>'
      );
      return;
    }
    const prepare_analysis_promise = redo
      ? this.sequentialInsertHumanBotMessage(
          this.state.selected_index,
          [],
          ['processing...']
        ).then(_ => analysis_promise)
      : this.sequentialInsertHumanBotMessage(
          this.state.selected_index,
          [sqlFormatter(gold_sql)],
          ['processing...']
        ).then(_ => analysis_promise);

    prepare_analysis_promise.then(analysis_result => {
      const [
        db_id,
        nlq,
        correct_models,
        pred_results,
        pred_sql_wrong_parts,
        similarity,
        captum_results,
      ] = analysis_result;

      const selected_index = redo
        ? this.state.selected_index
        : this.state.selected_index + 1;
      const recommend_buttons = correct_models.map((model, index) => {
        const model_name = models.find(element => element.value === model)
          .label;
        const now_selected_index = selected_index + 4 + captum_results.length;
        return {
          name: model_name,
          callback: () => {
            this.state.selected_index = now_selected_index;
            this.state.ready_to_analyze = true;
            return this.OneBotInsertCheckboxTable(now_selected_index).then(_ =>
              this.analyzeMessage(
                gold_sql,
                pred_results[index],
                queryAnalysis(
                  db_id,
                  model,
                  pred_results[index],
                  gold_sql,
                  nlq,
                  'http://141.223.199.148:4001/service'
                ),
                true
              )
            );
          },
        };
      });
      console.log(selected_index);
      captum_results.reduce(
        (prev_promise, captum_result) => {
          return prev_promise.then(insert_idx => {
            console.log(captum_result);
            console.log(this.botui.message);
            const insert_obj = {
              type: 'html',
              content: captum_result,
              delay: 1000,
            };
            this.botui.message.insert(insert_idx, insert_obj);
            return insert_idx + 1;
          });
        },
        this.botui.message
          .update(selected_index, {
            content: 'Incorrectly predicted phrases are highlighted in red.',
          })
          .then(_ =>
            this.oneBotInsertMessage(
              selected_index + 1,
              sqlFormatterWithWrongParts(pred_sql, pred_sql_wrong_parts)
            )
          )
          .then(_ =>
            this.oneBotInsertMessageWithButtons(
              selected_index + 2,
              'Recommended Models: ',
              recommend_buttons
            )
          )
          .then(_ => selected_index + 3)
      );
      this.state.ready_to_analyze = false;
    });
  }

  exploreMessage(nlq, promise_result) {
    this.sequentialHumanBotMessage([nlq], ['processing...'])
      .then(_ => promise_result)
      .then(pred_sql_filename => {
        const [pred_sql, plot_filename] = pred_sql_filename;
        this.botui.message
          .remove(-1)
          .then(_ =>
            this.botui.message.bot({
              content: sqlFormatter(pred_sql),
              delay: 100,
              add_button: true,
              toggle_callback: this.callback,
            })
          )
          .then(_ => {
            if (plot_filename) {
              this.botui.message.bot({
                type: 'embed',
                content: plot_filename,
                delay: 1000,
              });
            } else {
              return this.oneBotMessage(
                'The generated SQL query is not executable.'
              );
            }
          });
      })
      .catch(err => {
        this.oneBotMessage(err);
      });
  }

  oneBotInsertMessageWithButtons = (index, content, buttons, delay = 1000) =>
    this.botui.message
      .insert_with_button(index, {
        content: content,
        delay: delay,
        buttons: buttons,
      })
      .then(this.callback)
      .then(_ => index + 1);

  oneBotMessage = content =>
    this.botui.message
      .bot({
        content: content,
        delay: 1000,
      })
      .then(this.callback);

  oneBotInsertMessage = (index, content, delay = 1000) =>
    this.botui.message
      .insert(index, {
        content: content,
        delay: delay,
      })
      .then(this.callback)
      .then(_ => index + 1);

  OneBotInsertCheckboxTable = (index, delay = 1000) =>
    new Promise((resolve, reject) =>
      this.botui.message
        .insert_with_checkbox_table(index, {
          content: 'Please select attribution analysis options: ',
          delay: delay,
          callback: () => {
            resolve(index + 1);
          },
          headers: [
            [
              {
                name: 'gold',
                colspan: 1,
              },
              {
                name: 'pred',
                colspan: 1,
              },
            ],
          ],
          category_num: 2,
          rows: [
            {
              category: [
                {
                  name: 'input',
                  rowspan: 2,
                },
                {
                  name: 'words in the sentence',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'schema',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'encoder',
                  rowspan: 2,
                },
                {
                  name: 'Transformer encoder',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'LSTM encoder',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'decoder',
                  rowspan: 2,
                },
                {
                  name: 'Transformer decoder',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'LSTM decoder',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
          ],
        })
        .then(this.callback)
    );

  oneHumanMessage = content =>
    this.botui.message
      .human({
        content: content,
        delay: 500,
      })
      .then(this.callback);

  oneHumanInsertMessage = (index, content) =>
    this.botui.message
      .insert(index, {
        content: content,
        delay: 500,
        human: true,
      })
      .then(this.callback)
      .then(_ => index + 1);

  readyAnalyze = selected_index => {
    this.state.ready_to_analyze = true;
    this.state.selected_index = selected_index;
  };

  _sequenceHumanMessage = (promise, content) => {
    return new Promise(resolve => {
      resolve(promise.then(_ => this.oneHumanMessage(content)));
    });
  };

  _sequenceInsertHumanMessage = (promise, content) => {
    return new Promise(resolve => {
      resolve(
        promise.then(index => this.oneHumanInsertMessage(index, content))
      );
    });
  };

  _sequenceBotMessage = (promise, content) => {
    return new Promise(resolve => {
      resolve(promise.then(_ => this.oneBotMessage(content)));
    });
  };

  _sequenceInsertBotMessage = (promise, content) => {
    return new Promise(resolve => {
      resolve(promise.then(index => this.oneBotInsertMessage(index, content)));
    });
  };
}

export default BotUICommunicator;
