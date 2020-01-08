import React from 'react';
import Select from 'react-styled-select';
import {db_ids, models} from '../constants.js';
import '../css/App.css';
import '../css/BottomInterfaces.css';
import Switch from 'react-switch';
import MicIcon from '../MicIcon';

const BottomInterfaces = props => {
  return (
    <div className="bottom_interfaces">
      <Select
        className="select"
        options={db_ids}
        placeholder="DB..."
        onChange={props.handleDBChange}
        value={props.db_id}
      />
      <Select
        className="select"
        options={models}
        placeholder="Model..."
        onChange={props.handleModelChange}
        value={props.model}
      />
      <div className="nlq_textbox_and_mic">
        <input
          className="nlq_textbox"
          placeholder="Type here"
          value={props.clicked_mic ? props.transcript : props.original_nlq}
          onChange={props.handleNLQChange}
        />
        <Switch
          onHandleColor="#1c52a3"
          onColor="#c1d7f7"
          offColor="#f0f2f5"
          boxShadow="0px 1px 5px rgba(0, 0, 0, 0.6)"
          activeBoxShadow="0px 0px 1px 10px rgba(0, 0, 0, 0.2)"
          height={16}
          width={32}
          checked={props.clicked_mic}
          onChange={props.handleClickedMic}
          uncheckedIcon={
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                fontSize: 15,
                color: 'orange',
                paddingRight: 2,
              }}
            >
              <MicIcon />
            </div>
          }
          id="icon-switch"
        />
      </div>
      <button className="button_pretty" onClick={props.handleRunClicked}>
        {' '}<b>RUN</b>{' '}
      </button>
    </div>
  );
};

export default BottomInterfaces;
