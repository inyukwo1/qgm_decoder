import PropTypes from 'prop-types';
import dog_kennels from './images/dog_kennels.png';
import flight_2 from './images/flight_2.png';
import pets_1 from './images/pets_1.png';
import concert_singer from './images/concert_singer.png';
import museum_visit from './images/museum_visit.png';
import battle_death from './images/battle_death.png';
import student_transcripts_tracking
from './images/student_transcripts_tracking.png';
import singer from './images/singer.png';
import cre_Doc_Template_Mgt from './images/cre_Doc_Template_Mgt.png';
import world_1 from './images/world_1.png';
import employee_hire_evaluation from './images/employee_hire_evaluation.png';
import network_1 from './images/network_1.png';
import poker_player from './images/poker_player.png';
import real_estate_properties from './images/real_estate_properties.png';
import course_teach from './images/course_teach.png';
import voter_1 from './images/voter_1.png';
import wta_1 from './images/wta_1.png';
import orchestra from './images/orchestra.png';
import car_1 from './images/car_1.png';
import tvshow from './images/tvshow.png';
import imdb from './images/imdb.png';

export const select_styles = {
    container: (base, state) => ({
        ...base,
        border: state.isFocused ? null : null,
        transition: 'border-color 0.2s ease, box-shadow 0.2s ease, padding 0.2s ease',
        '&:hover': {
            boxShadow: '0 2px 4px 0 rgba(41, 56, 78, 0.1)',
        },
    }),
    control: (base, state) => ({
        ...base,
        background: '#e3edfc',
        fontFamily: 'Open Sans',
        height: 'calc(4.2vmin)',
        minHeight: 'fit-content',
    }),
    valueContainer: (base, state) => ({
        ...base,
        backgroundColor: '#e3edfc',
        minHeight: 0,
        maxHeight: 'calc(2vmin)',
        padding: '0px 0.5vw 0px',
        margin: 0,
    }),
    dropdownIndicator: base => ({
        ...base,
        padding: '0px 0.2vw',
        minHeight: '0.2vw',
    }),
    option: (base, state) => ({
        ...base,
        backgroundColor: state.isSelected ?
            '#1c52a3' : state.isFocused ? '#e3edfc' : 'white',
    }),
    indicatorSeparator: base => ({
        ...base,
        display: 'none',
    }),
    placeholder: base => ({
        ...base,
        textAlign: 'center',
    }),
    input: base => ({
        ...base,
        margin: 0,
        padding: 0,
    }),
    menu: base => ({
        ...base,
        top: 'auto',
        bottom: '100%',
    }),
};

export const propTypes = {
    // Props injected by SpeechRecognition
    transcript: PropTypes.string,
    recognition: PropTypes.object,
    startListening: PropTypes.func,
    stopListening: PropTypes.func,
    resetTranscript: PropTypes.func,
    browserSupportsSpeechRecognition: PropTypes.bool,
};

export const models = [
    { label: 'IRNet', value: 'irnet' },
    { label: 'GNN', value: 'gnn' },
    { label: 'TypeSQL', value: '1' },
    { label: 'Coarse2Fine', value: '2' },
    { label: 'NSP', value: '3' },
    { label: 'SQLNet', value: '4' },
    { label: 'Seq2SQL', value: '5' },
    { label: 'PT-MAML', value: '6' },
    { label: 'IRNet + BERT', value: 'ours' },
];

export const loading_files = [
    { label: 'Load...', value: 'load' },
    { label: 'George', value: 'george' },
    { label: 'John', value: 'john' },
    { label: 'Tom', value: 'tom' },
    { label: 'Jill', value: 'jill' },
    { label: 'Bryan', value: 'bryan' },
    { label: 'Sophia', value: 'sophia' },
    { label: 'Emma', value: 'emma' },
    { label: 'Mike', value: 'mike' },
    { label: 'Mitchell', value: 'mitchell' },
];

export const db_ids = [
    { label: 'Dog kennels', value: 'dog_kennels', img: dog_kennels },
    { label: 'Flight', value: 'flight_2', img: flight_2 },
    { label: 'Pets', value: 'pets_1', img: pets_1 },
    { label: 'Concert singer', value: 'concert_singer', img: concert_singer },
    { label: 'Museum visit', value: 'museum_visit', img: museum_visit },
    { label: 'Battle death', value: 'battle_death', img: battle_death },
    {
        label: 'Student transcripts tracking',
        value: 'student_transcripts_tracking',
        img: student_transcripts_tracking,
    },
    { label: 'Singer', value: 'singer', img: singer },
    {
        label: 'Cre Doc Template Mgt',
        value: 'cre_Doc_Template_Mgt',
        img: cre_Doc_Template_Mgt,
    },
    { label: 'World', value: 'world_1', img: world_1 },
    {
        label: 'Employee hire evaluation',
        value: 'employee_hire_evaluation',
        img: employee_hire_evaluation,
    },
    { label: 'Network', value: 'network_1', img: network_1 },
    { label: 'Poker player', value: 'poker_player', img: poker_player },
    {
        label: 'Real estate properties',
        value: 'real_estate_properties',
        img: real_estate_properties,
    },
    { label: 'Course teach', value: 'course_teach', img: course_teach },
    { label: 'Voter', value: 'voter_1', img: voter_1 },
    { label: 'Wta', value: 'wta_1', img: wta_1 },
    { label: 'Orchestra', value: 'orchestra', img: orchestra },
    { label: 'Car', value: 'car_1', img: car_1 },
    { label: 'TVshow', value: 'tvshow', img: tvshow },
    { label: 'IMDB', value: 'imdb', img: imdb },
];

export const styled_db_ids = [{
        label: 'Dog kennels',
        value: { label: 'Dog kennels', value: 'dog_kennels', img: dog_kennels },
    },
    { label: 'Flight', value: { label: 'Flight', value: 'flight_2', img: flight_2 } },
    { label: 'Pets', value: { label: 'Pets', value: 'pets_1', img: pets_1 } },
    {
        label: 'Concert singer',
        value: {
            label: 'Concert singer',
            value: 'concert_singer',
            img: concert_singer,
        },
    },
    {
        label: 'Museum visit',
        value: { label: 'Museum visit', value: 'museum_visit', img: museum_visit },
    },
    {
        label: 'Battle death',
        value: { label: 'Battle death', value: 'battle_death', img: battle_death },
    },
    {
        label: 'Student transcripts tracking',
        value: {
            label: 'Student transcripts tracking',
            value: 'student_transcripts_tracking',
            img: student_transcripts_tracking,
        },
    },
    { label: 'Singer', value: { label: 'Singer', value: 'singer', img: singer } },
    {
        label: 'Cre Doc Template Mgt',
        value: {
            label: 'Cre Doc Template Mgt',
            value: 'cre_Doc_Template_Mgt',
            img: cre_Doc_Template_Mgt,
        },
    },
    { label: 'World', value: { label: 'World', value: 'world_1', img: world_1 } },
    {
        label: 'Employee hire evaluation',
        value: {
            label: 'Employee hire evaluation',
            value: 'employee_hire_evaluation',
            img: employee_hire_evaluation,
        },
    },
    {
        label: 'Network',
        value: { label: 'Network', value: 'network_1', img: network_1 },
    },
    {
        label: 'Poker player',
        value: { label: 'Poker player', value: 'poker_player', img: poker_player },
    },
    {
        label: 'Real estate properties',
        value: {
            label: 'Real estate properties',
            value: 'real_estate_properties',
            img: real_estate_properties,
        },
    },
    {
        label: 'Course teach',
        value: { label: 'Course teach', value: 'course_teach', img: course_teach },
    },
    { label: 'Voter', value: { label: 'Voter', value: 'voter_1', img: voter_1 } },
    { label: 'Wta', value: { label: 'Wta', value: 'wta_1', img: wta_1 } },
    {
        label: 'Orchestra',
        value: { label: 'Orchestra', value: 'orchestra', img: orchestra },
    },
    { label: 'Car', value: { label: 'Car', value: 'car_1', img: car_1 } },
    { label: 'TVshow', value: { label: 'TVshow', value: 'tvshow', img: tvshow } },
];