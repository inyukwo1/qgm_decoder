import React from 'react';
import '../css/App.css';
import '../css/TopInterfaces.css';
import {db_ids} from '../constants.js';

class TopInterfaces extends React.Component {
  state = {
    mode: 'Explore',
    schema_showing: false,
  };
  handleSchemaClicked = e => {
    if (this.state.schema_showing) {
      this.setState ({
        schema_showing: false,
      });
    } else {
      this.setState ({
        schema_showing: true,
      });
    }
  };
  handleExploreAnalyzeButtonClicked = e => {
    this.props.switchExploreAnalyze ();
    if (this.state.mode === 'Explore') {
      this.setState ({
        mode: 'Analyze',
      });
    } else {
      this.setState ({
        mode: 'Explore',
      });
    }
  };
  exploreMode = () => {
    return this.state.mode === 'Explore';
  };
  render () {
    const classname = this.state.schema_showing
      ? 'button_pretty showing-schema'
      : 'button_pretty hiding-schema';
    const e = db_ids.find (o => o.value === this.props.db_id);
    const db_img = e ? e.img : null;
    const schema_div = this.state.schema_showing
      ? <img className="schema-img" src={db_img} alt="" />
      : <div className="schema_text">
          {' '}<b>Schema</b>{' '}
        </div>;
    return (
      <div className="top_interfaces">
        <label className="switch" >
          <input type="checkbox"></input>
          <span
            className="explore_analyze"
            onClick={this.handleExploreAnalyzeButtonClicked}
          >
            {' '}<b>{this.state.mode}</b>{' '}
          </span>
        </label>
        <button className={classname} onClick={this.handleSchemaClicked}>
          {schema_div}
        </button>
      </div>
    );
  }
}

export default TopInterfaces;
