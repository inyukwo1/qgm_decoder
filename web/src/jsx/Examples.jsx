import React from 'react';
import ScrollArea from "react-scrollbar"
import '../css/App.scss';
import '../css/Examples.scss'
import dev_data from "../dev.json"

const getContent = (db_id) => {
    const inner = [];
    dev_data.forEach((data, idx) => {
      if (data.db_id === db_id) {
        inner.push(
          <div className="ex_item">
            <div style={{textAlign: "left"}}>
              {data.question} 
            </div>
            <div style={{textAlign: "left"}}>
              <b>SQL:</b> {data.query}
            </div>
          </div>,
        );
      }
    });
    return inner;
  }

const Examples = (props) => {
    return (
        <div className='examples'>
            <b> Examples </b>
            <ScrollArea className='scroll_examples'
              speed={0.8}
              contentClassName="example_content"
              horizontal={false}
              >
              {getContent(props.db_id)}
            </ScrollArea>
          </div>
    );
}

export default Examples
