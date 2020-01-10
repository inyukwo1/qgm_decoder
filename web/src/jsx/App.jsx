import React from 'react';
import {propTypes, db_ids} from '../constants.js';
import TopInterfaces from './TopInterfaces';
import BottomInterfaces from './BottomInterfaces';
import MiddleInterfaces from './MiddleInterfaces';
import Examples from './Examples';
import BotUICommunicator from './BotUICommunicator';
import queryTextToSQL from './BackendCommunicator';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'botui/build/botui-theme-default.css';
import 'botui/build/botui.min.css';
import '../css/App.css';
import SpeechRecognition from 'react-speech-recognition';
import queryAnalysis from './AnalysisCommunicator';

class App extends React.Component {
  componentDidMount () {
    this.BotUICommunicator = new BotUICommunicator ();
    this.BotUICommunicator.registerBotUI (this.botui);
    this.BotUICommunicator.registerCallback (this.refreshStatusbar);
    this.BotUICommunicator.sequentialHumanBotMessage (
      [],
      [
        'Welcome to EGA!',
        'This is a demo produced by the Database & Data Mining lab @POSTECH.',
        'Please select a database..',
      ]
    );
  }
  componentWillMount () {
    const {recognition} = this.props;
    recognition.lang = 'en-US';
  }

  state = {
    original_nlq: '',
    db_id: '',
    clicked_mic: false,
    model: '',
    pred_sql: '',
  };

  handleClickedMic = clicked => {
    const {
      startListening,
      stopListening,
      resetTranscript,
      transcript,
    } = this.props;
    this.setState ({clicked_mic: clicked});

    if (clicked) {
      resetTranscript ();
      startListening ();
    } else {
      stopListening ();
      this.setState ({original_nlq: transcript});
    }
  };

  handleReset = () => {
    this.botui.message.removeAll ();
  };

  handleDBChange = val => {
    const e = db_ids.find (o => o.value === val);
    this.setState ({
      db_id: e.value,
    });
    this.BotUICommunicator.sequentialHumanBotMessage (
      [e.label],
      [
        'Selecting ' + e.value + '...',
        '![one of my article](' + e.img + ')',
        "Done! What's your question?",
      ]
    );
  };

  handleModelChange = val => {
    this.setState ({
      model: val,
    });
  };

  handleNLQChange = e => {
    this.setState ({
      original_nlq: e.target.value,
    });
  };

  refreshStatusbar = () => {
    this.botui.message.getMessageLengthCorrectPair ().then (len_msg_pair => {
      this.botui.setState ({
        len_msg_pairs: len_msg_pair,
      });
    });
  }

  handleRunClicked = e => {
    if (this.state.db_id === '') {
      this.BotUICommunicator.oneBotMessage (
        '<b>Please specify a database first.</b>'
      );
      return;
    }
    if (this.state.clicked_mic) {
      this.BotUICommunicator.oneBotMessage (
        '<b>Please turn off the mic before running.</b>'
      );
      return;
    }
    if (this.state.model === '') {
      this.BotUICommunicator.oneBotMessage ('<b>Please specify a model.</b>');
      return;
    }
    if (!this.TopInterfaces.exploreMode ()) {
      const nlq = this.state.original_nlq;
      const db_id = this.state.db_id;
      const model = this.state.model;
      const pred_sql = this.state.pred_sql;
      this.BotUICommunicator.analyzeMessage (
        nlq,
        pred_sql,
        queryAnalysis( db_id,
          model,
          pred_sql,
          nlq,
          'http://141.223.199.148:4001/service')
      );
      this._initInput ();
      return;
    }
    const nlq = this.state.original_nlq;
    const db_id = this.state.db_id;
    const model = this.state.model;
    this.BotUICommunicator.exploreMessage (
      nlq,
      queryTextToSQL (db_id, model, nlq, 'http://141.223.199.148:4001/service')
    );
    this._initInput ();
  };

  _initInput = () => {
    this.setState ({
      original_nlq: '',
    });
  };

  incorrectClickAnalyzeCallback = (next_index, pred_sql) => {
    this.BotUICommunicator
      .oneBotInsertMessage (next_index + 1, 'What is a correct query?', 100)
      .then (index => {
        this.BotUICommunicator.readyAnalyze (index);
        this.setState ({
          pred_sql: pred_sql,
        });
      });
  };

  switchExploreAnalyze = e => {
    if (this.TopInterfaces.exploreMode ()) {
      this.botui.message.enableIncorrectClick (
        this.incorrectClickAnalyzeCallback
      );
    } else {
      this.botui.message.disableIncorrectClick ();
    }
  };

  render () {
    if (!this.props.browserSupportsSpeechRecognition) {
      return null;
    }
    return (
      <div className="App">
        <div className="AppMain">
          <div className="App-header">
            <span style={{color: '#1c52a3'}}><b>EGA</b></span>
          </div>
          <TopInterfaces
            ref={cmp => {
              this.TopInterfaces = cmp;
            }}
            switchExploreAnalyze={this.switchExploreAnalyze}
            db_id={this.state.db_id}
          />
          <MiddleInterfaces
            ref={cmp => {
              this.botui = cmp;
            }}
          />
          <BottomInterfaces
            handleDBChange={this.handleDBChange}
            handleModelChange={this.handleModelChange}
            handleRunClicked={this.handleRunClicked}
            handleClickedMic={this.handleClickedMic}
            handleNLQChange={this.handleNLQChange}
            db_id={this.state.db_id}
            model={this.state.model}
            clicked_mic={this.state.clicked_mic}
            original_nlq={this.state.original_nlq}
          />
          <Examples db_id={this.state.db_id} />
        </div>
      </div>
    );
  }
}
const options = {
  autoStart: false,
};
App.propTypes = propTypes;

export default SpeechRecognition (options) (App);
