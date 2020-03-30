import {models} from '../constants.js';
import ColWiseTable from './colwisetable';
import {verifyQuery} from './BackendCommunicator';

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

// eslint-disable-next-line no-extend-native
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
        analysisAgain,
        correct_models,
        pred_results,
        pred_sql_wrong_parts,
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
          visible: _ => true,
          callback: () => {
            this.state.selected_index = now_selected_index;
            this.state.ready_to_analyze = true;
            return this.OneBotInsertCheckboxTable(now_selected_index).then(_ =>
              this.analyzeMessage(
                gold_sql,
                pred_results[index],
                analysisAgain(model, pred_results[index]),
                true
              )
            );
          },
        };
      });
      captum_results
        .reduce(
          (prev_promise, captum_result) => {
            return prev_promise.then(insert_idx => {
              const insert_obj = {
                type: 'html',
                content: captum_result,
                delay: 1000,
              };
              this.botui.message.insert(insert_idx, insert_obj);
              return insert_idx + 1;
            });
          },
          new Promise((resolve, reject) => {
            resolve(selected_index);
          })
        )
        .then(index =>
          this.botui.message
            .update(index, {
              content: 'Incorrectly predicted phrases are highlighted in red.',
            })
            .then(_ =>
              this.oneBotInsertMessage(
                index + 1,
                sqlFormatterWithWrongParts(pred_sql, pred_sql_wrong_parts)
              )
            )
            .then(_ =>
              this.oneBotInsertMessageWithButtons(
                index + 2,
                'Correct model(s) for this question: </br> </br>',
                recommend_buttons
              )
            )
            .then(_ => index + 3)
        );
      this.state.ready_to_analyze = false;
    });
  }

  exploreMessage(isverifying, nlq, promise_result) {
    this.sequentialHumanBotMessage([nlq], ['processing...'])
      .then(_ => promise_result)
      .then(result => {
        const [pred_sql, execution_result] = result;
        this.botui.message
          .remove(-1)
          .then(_ => this.oneBotMessage('The result is: '))
          .then(_ => {
            const table = ColWiseTable(execution_result);
            return this.botui.message.insert_with_button(-1, {
              type: 'html',
              content: table,
              delay: 1000,
              add_button: true,
              toggle_callback: this.callback,
              nlq_ref_idx: -2,
              buttons: [
                {
                  name: 'Verify',
                  callback: index => {
                    this.verifyMessage(index + 3, nlq, pred_sql, verifyQuery());
                  },
                  visible: _ => isverifying(),
                },
              ],
            });
          })
          .then(_ => this.oneBotMessage('The underlying SQL query is: '))
          .then(_ =>
            this.botui.message.bot({
              type: 'html',
              content: sqlFormatter(pred_sql),
              delay: 1000,
              correct_ref_idx: -2,
              nlq_ref_idx: -4,
            })
          );
      })
      .catch(err => {
        this.oneBotMessage(err);
      });
  }

  verifyMessage(insert_idx, nlq, pred_sql, verify_promise) {
    return verify_promise.then(verify_result => {
      const [new_db_instance, execution_result] = verify_result;
      const db_instance_table = ColWiseTable(new_db_instance);
      const execution_result_table = ColWiseTable(execution_result);
      return this.oneBotInsertMessage(insert_idx, 'Verify with a new table: ')
        .then(index => this.oneBotInsertHTMLMessage(index, db_instance_table))
        .then(index => this.oneBotInsertMessage(index, 'The question is: '))
        .then(index => this.oneBotInsertMessage(index, nlq))
        .then(index => this.oneBotInsertMessage(index, 'The result is: '))
        .then(index => {
          return this.botui.message.insert(index, {
            type: 'html',
            content: execution_result_table,
            delay: 1000,
            add_button: true,
            toggle_callback: this.callback,
            nlq_ref_idx: -2,
          });
        })
        .then(index => index + 1)
        .then(index =>
          this.oneBotInsertMessage(index, 'The underlying SQL query is: ')
        )
        .then(index =>
          this.botui.message.insert(index, {
            type: 'html',
            content: sqlFormatter(pred_sql),
            delay: 1000,
            correct_ref_idx: -2,
            nlq_ref_idx: -4,
          })
        );
    });
  }

  drawDBInstance = db_instance => {
    const table = ColWiseTable(db_instance);
    const description_with_table =
      'Instance 14 was selected: </br> </br>' + table;
    return this.oneBotHTMLMessage(description_with_table).then(_ =>
      this.oneBotMessage("Done! What's yout question?")
    );
  };

  oneBotInsertMessageWithButtons = (index, content, buttons, delay = 1000) =>
    this.botui.message
      .insert_with_button(index, {
        content: content,
        delay: delay,
        buttons: buttons,
      })
      .then(this.callback)
      .then(_ => index + 1);

  oneBotHTMLMessage = content =>
    this.botui.message
      .bot({
        type: 'html',
        content: content,
        delay: 1000,
      })
      .then(this.callback);

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

  oneBotInsertHTMLMessage = (index, content, delay = 1000) =>
    this.botui.message
      .insert(index, {
        type: 'html',
        content: content,
        delay: delay,
      })
      .then(this.callback)
      .then(_ => index + 1);

  OneBotInsertCheckboxTable = (index, delay = 1000) =>
    new Promise((resolve, reject) =>
      this.botui.message
        .insert_with_checkbox_table(index, {
          content: 'Please select analysis options: </br> </br>',
          delay: delay,
          callback: () => {
            resolve(index + 1);
          },
          headers: [
            [
              {
                name: 'Ground truth',
                colspan: 1,
              },
              {
                name: 'Generated',
                colspan: 1,
              },
            ],
          ],
          category_num: 2,
          rows: [
            {
              category: [
                {
                  name: 'Input',
                  rowspan: 2,
                },
                {
                  name: 'Words in the sentence',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'Schema',
                  rowspan: 1,
                },
              ],
              row_num: 2,
            },
            {
              category: [
                {
                  name: 'Encoder',
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
                  name: 'Decoder',
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
