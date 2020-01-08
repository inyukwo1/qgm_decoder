import React from 'react';
import '../css/MiddleInterfaces.css';
import Botui from 'botui-react';

const makeStatusbarContent = (total_height, len_msg_pairs) => {
  const items = [];
  const sum_len = len_msg_pairs.reduce (
    (accumulator, currentvalue) => accumulator + currentvalue[0]
  , 0);
  for (const [len, msg] of len_msg_pairs) {
    const classname = msg === 'Incorrect' ? 'incorrect-bar' : 'correct-bar';
    const style = {
      height: total_height * len / sum_len + 'px',
    };
    items.push (<div className={classname} style={style} />);
  }
  return items;
};

class MiddleInterfaces extends React.Component {
  state = {
      len_msg_pairs: []
  };

  componentDidMount() {
      this.message = this.botui.message
  }

  render () {
    return (
      <div className="mybotui">
        <Botui ref={cmp => (this.botui = cmp)} />
        <div className="statusbar">
          <div className="padding" />
          {makeStatusbarContent (document.documentElement.clientHeight * 0.6 - 20, this.state.len_msg_pairs)}  {/* hard coded - the height of bot ui window */}
          <div className="padding" />
        </div>
      </div>
    );
  }
}

export default MiddleInterfaces;
