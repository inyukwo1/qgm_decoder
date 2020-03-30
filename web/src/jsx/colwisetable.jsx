const ColWiseTable = props => {
  console.log(props);
  const {headers, rows} = props;
  const header_dom = headers
    .map(one_header => {
      const one_header_dom = one_header
        .map(elem => {
          const {name, colspan} = elem;
          return '<th colspan=' + colspan + '> ' + name + '</th>';
        })
        .reduce((prev, curr) => prev + curr, ' ');
      return '<tr>' + one_header_dom + '</tr>';
    })
    .reduce((prev, curr) => prev + curr, ' ');
  const body_dom = rows
    .map(row => {
      const one_row_dom = row
        .map(elem => {
          return '<td>' + elem + '</td>';
        })
        .reduce((prev, curr) => prev + curr, ' ');
      return '<tr>' + one_row_dom + '</tr>';
    })
    .reduce((prev, curr) => prev + curr, ' ');
  return (
    '<table> <thead>' +
    header_dom +
    '</thead> <tbody>' +
    body_dom +
    '</tbody> </table>'
  );
};

export default ColWiseTable;
