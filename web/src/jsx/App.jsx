import React from "react";
import { propTypes, db_ids } from "../constants.js";
import TopInterfaces from "./TopInterfaces";
import BottomInterfaces from "./BottomInterfaces";
import MiddleInterfaces from "./MiddleInterfaces";
import Examples from "./Examples";
import BotUICommunicator from "./BotUICommunicator";
import { queryTextToSQL, getDBInstance } from "./BackendCommunicator";
import "bootstrap/dist/css/bootstrap.min.css";
import "botui/build/botui-theme-default.css";
import "botui/build/botui.min.css";
import "../css/App.scss";
import SpeechRecognition from "react-speech-recognition";
import queryAnalysis from "./AnalysisCommunicator";

// HACK!!!
let query_cnt = 0;

class App extends React.Component {
  componentDidMount() {
    this.BotUICommunicator = new BotUICommunicator();
    this.BotUICommunicator.registerBotUI(this.botui);
    this.BotUICommunicator.registerCallback(this.refreshStatusbar);
    this.BotUICommunicator.sequentialHumanBotMessage(
      [],
      [
        "Welcome to EGA: Explore-Gather-Analyze Tool for NL to SQL!",
        "Please select a database..",
      ]
    );
  }

  componentDidUpdate() {
    const iframes = Array.from(document.getElementsByTagName("iframe"));

    iframes.forEach((iframe) => {
      iframe.style.height =
        iframe.contentWindow.document.body.offsetHeight + "px";
    });
  }

  componentWillMount() {
    const { recognition } = this.props;
    recognition.lang = "en-US";
  }

  state = {
    original_nlq: "",
    db_id: "",
    clicked_mic: false,
    model: "",
    file: "",
    mode: "Explore",
    analyze_sql: "",
    analyze_nlq: "",
    db_obj: null,
  };

  handleClickedMic = (clicked) => {
    const {
      startListening,
      stopListening,
      resetTranscript,
      transcript,
    } = this.props;
    this.setState({ clicked_mic: clicked });

    if (clicked) {
      resetTranscript();
      startListening();
    } else {
      stopListening();
      this.setState({ original_nlq: transcript });
    }
  };

  handleReset = () => {
    this.botui.message.removeAll();
  };

  handleDBChange = (val) => {
    const e = db_ids.find((o) => o.value === val);
    this.setState({
      db_id: e.value,
    });
    this.BotUICommunicator.sequentialHumanBotMessage(
      [e.label],
      ["Randomly selecting a database instance from " + e.value + "..."]
    )
      .then((_) => getDBInstance(e.value))
      .then((response) => {
        const [db_obj, db_instance_table] = response;
        this.setState({ db_obj: db_obj });
        this.BotUICommunicator.drawDBInstance(db_instance_table);
      });
  };

  handleModelChange = (val) => {
    this.setState({
      model: val,
    });
  };

  _messageInvisible = () => {
    const msgs = Array.from(document.getElementsByClassName("botui-message"));
    msgs.forEach((msg) => {
      msg.style.visibility = "hidden";
    });
    const statusbar = document.getElementsByClassName("statusbar");
    statusbar[0].style.visibility = "hidden";
  };

  _messageVisible = () => {
    const msgs = Array.from(document.getElementsByClassName("botui-message"));
    msgs.forEach((msg) => {
      msg.style.visibility = "visible";
    });
    const statusbar = document.getElementsByClassName("statusbar");
    statusbar[0].style.visibility = "visible";
    // document.getElementsByClassName ('bwTeTR')[0].innerText = 'IMDB';
    // document.getElementsByClassName ('kGtLwg')[0].innerText = 'IRNet';
  };

  handleFileChange = (val) => {
    if (val === "load") {
      this._messageInvisible();
      this.setState({
        db_id: "",
        model: "",
      });
    } else {
      this.setState({
        file: val,
      });
      this.setState({
        db_id: "imdb",
        model: "irnet",
      });
    }
    console.log(this.BotUICommunicator);
    this.BotUICommunicator.botui.message.updateMsgs();
  };

  handleNLQChange = (e) => {
    this.setState({
      original_nlq: e.target.value,
    });
  };

  refreshStatusbar = () => {
    this.botui.message.getMessageLengthCorrectPair().then((len_msg_pair) => {
      this.botui.setState({
        len_msg_pairs: len_msg_pair,
      });
    });
  };

  handleRunClicked = (e) => {
    if (this.state.db_id === "") {
      this.BotUICommunicator.oneBotMessage(
        "<b>Please specify a database first.</b>"
      );
      return;
    }
    if (this.state.clicked_mic) {
      this.BotUICommunicator.oneBotMessage(
        "<b>Please turn off the mic before running.</b>"
      );
      return;
    }
    if (this.state.model === "") {
      this.BotUICommunicator.oneBotMessage("<b>Please specify a model.</b>");
      return;
    }
    if (!this.TopInterfaces.exploreMode()) {
      const gold_sql = this.state.original_nlq;
      const db_id = this.state.db_id;
      const model = this.state.model;
      const pred_sql = this.state.analyze_sql;
      const analyze_nlq = this.state.analyze_nlq;
      this.BotUICommunicator.analyzeMessage(
        gold_sql,
        pred_sql,
        queryAnalysis(
          db_id,
          model,
          pred_sql,
          gold_sql,
          analyze_nlq,
          this.state.db_obj
        )
      );
      this._initInput();
      return;
    }
    const nlq = this.state.original_nlq;
    const db_id = this.state.db_id;
    const model = this.state.model;
    this.BotUICommunicator.exploreMessage(
      (_) => this.state.file === "john",
      nlq,
      queryTextToSQL(db_id, model, nlq, this.state.db_obj, query_cnt)
    );
    query_cnt += 1;
    this._initInput();
  };

  _initInput = () => {
    this.setState({
      original_nlq: "",
    });
  };

  incorrectClickAnalyzeCallback = (next_index, pred_sql, nlq) => {
    this.BotUICommunicator.OneBotInsertCheckboxTable(next_index)
      .then((index) =>
        this.BotUICommunicator.oneBotInsertMessage(
          index,
          "What is a correct query?",
          100
        )
      )
      .then((index) => {
        this.BotUICommunicator.readyAnalyze(index);
        this.setState({
          analyze_sql: pred_sql.replace(/<br \/>/gi, ""),
          analyze_nlq: nlq,
        });
      });
  };

  switchExploreAnalyze = (e) => {
    if (this.TopInterfaces.exploreMode()) {
      this.botui.message.enableIncorrectClick(
        this.incorrectClickAnalyzeCallback
      );
      this.setState({
        mode: "Analyze",
      });
    } else {
      this.botui.message.disableIncorrectClick();
      this.setState({
        mode: "Explore",
      });
    }
  };

  render() {
    if (!this.props.browserSupportsSpeechRecognition) {
      return null;
    }
    const root_classname =
      this.state.mode === "Explore" ? "App-explore" : "App-analyze";
    return (
      <div className={root_classname}>
        <div className="AppMain">
          <div className="App-header">
            <span>
              <b>EGA</b>: Explore-Gather-Analyze Tool for NL to SQL
            </span>
          </div>
          <TopInterfaces
            ref={(cmp) => {
              this.TopInterfaces = cmp;
            }}
            switchExploreAnalyze={this.switchExploreAnalyze}
            db_id={this.state.db_id}
          />
          <MiddleInterfaces
            ref={(cmp) => {
              this.botui = cmp;
            }}
          />
          <BottomInterfaces
            handleDBChange={this.handleDBChange}
            handleModelChange={this.handleModelChange}
            handleRunClicked={this.handleRunClicked}
            handleClickedMic={this.handleClickedMic}
            handleNLQChange={this.handleNLQChange}
            handleFileChange={this.handleFileChange}
            db_id={this.state.db_id}
            model={this.state.model}
            file={this.state.file}
            clicked_mic={this.state.clicked_mic}
            original_nlq={this.state.original_nlq}
            hide_loading={false}
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

export default SpeechRecognition(options)(App);
