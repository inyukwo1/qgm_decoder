import torch
import torch.nn as nn
from rule.semql.semql import SemQL
from encoder.irnet import nn_utils
from encoder.irnet.pointer_net import PointerNet


log = logging.getLogger(__name__)

SKETCH_LIST = ["Root1", "Root", "Sel", "N", "Filter", "Sup", "Order"]
DETAIL_LIST = ["A", "C", "T"]


from typing import List, Dict

from rule.grammar import Action, Symbol, SymbolId
from decoder.semql_framework.state import (
    LSTMStateGold,
    LSTMStatePred,
    LSTMState,
)
from framework.sequential_monad import (
    SequentialMonad,
    WhileLogic,
    LogicUnit,
    TensorPromiseOrTensor,
    TensorPromise,
)
from framework.lazy_modules import (
    LazyLinear,
    LazyDropout,
    LazyLSTMCellDecoder,
    LazyTransformerDecoder,
    LazyCalculateSimilarity,
)


class SemQLDecoderFramework(nn.Module):
    def __init__(self, cfg):
        super(SemQLDecoderFramework, self).__init__()
        self.cfg = cfg
        self.is_cuda = cfg.cuda != -1
        is_bert = cfg.is_bert
        self.grammar = SemQL.Grammar()
        self.use_column_pointer = cfg.column_pointer

        self.new_tensor = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor

        hidden_size = cfg.hidden_size
        att_vec_size = cfg.att_vec_size
        type_embed_size = cfg.type_embed_size
        action_embed_size = cfg.action_embed_size
        input_dim = action_embed_size + att_vec_size + type_embed_size

        self.decode_max_time_step = 40
        self.action_embed_size = action_embed_size
        self.type_embed_size = type_embed_size

        # Decoder Layers
        self.encoder_Lstm = nn.LSTM(self.embed_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.decoder_cell_init = LazyLinear(hidden_size, hidden_size)
        self.lf_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)
        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, hidden_size)

        self.att_sketch_linear = LazyLinear(hidden_size, hidden_size, bias=False)
        self.att_lf_linear = LazyLinear(hidden_size, hidden_size, bias=False)
        self.sketch_att_vec_linear = LazyLinear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )
        self.lf_att_vec_linear = LazyLinear(
            hidden_size + hidden_size, att_vec_size, bias=False
        )
        self.prob_att = LazyLinear(att_vec_size, 1)
        self.prob_len = LazyLinear(1, 1)
        self.q_att = LazyLinear(hidden_size, self.embed_size)
        self.column_rnn_input = LazyLinear(
            self.embed_size, action_embed_size, bias=False
        )
        self.table_rnn_input = LazyLinear(self.embed_size, action_embed_size, bias=False)
        self.dropout = LazyDropout(cfg.dropout)
        self.query_vec_to_action_embed = LazyLinear(
            att_vec_size, action_embed_size, bias=cfg.readout == "non_linear",
        )



        # Embeddings
        self.production_embed = nn.Embedding(
            len(self.grammar.prod2id), action_embed_size
        )
        self.type_embed = nn.Embedding(len(self.grammar.type2id), type_embed_size)
        self.production_readout_b = nn.Parameter(
            torch.FloatTensor(len(self.grammar.prod2id)).zero_()
        )
        self.N_embed = nn.Embedding(
            len(SemQL.N._init_grammar()), action_embed_size
        )
        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)
        nn.init.xavier_normal_(self.N_embed.weight.data)




        self.read_out_act = (
            torch.tanh if cfg.readout == "non_linear" else nn_utils.identity
        )

        self.production_readout = lambda q: torch.nn.functional.linear(
            self.read_out_act(self.query_vec_to_action_embed(q)),
            self.production_embed.weight,
            self.production_readout_b,
        )

        self.column_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

        self.table_pointer_net = PointerNet(
            hidden_size, self.embed_size, attention_type=cfg.column_att
        )

    def action_to_embedding(self, state: LSTMState, action:Action) -> torch.Tensor:
        if action[0] == "C":
            col_idx = action[1]
            return setste

        pass

    def symbol_list_to_embedding(self, symbol_list: List[Symbol]) -> torch.Tensor:
        pass

    def one_dim_zero_tensor(self):
        return torch.zeros(self.dim).cuda()

    def forward(self, encoded_src, encoded_col, encoded_tab, col_tab_dic, golds):
        b_size = len(encoded_src)
        if golds:
            state_class = LSTMStateGold
            states = [LSTMStateGold(encoded_src[b_idx],
                                    encoded_col[b_idx],
                                    encoded_tab[b_idx],
                                    col_tab_dic[b_idx],
                                    golds[b_idx],
                                    )
                      for b_idx in range(b_size)
                      ]
        else:
            state_class = LSTMStatePred
            states = [
                LSTMStateGold(encoded_src[b_idx],
                                    encoded_col[b_idx],
                                    encoded_tab[b_idx],
                                    col_tab_dic[b_idx],
                                    self.grammar.start_symbol,
                              )
                for b_idx in range(b_size)
            ]

        def embed_history_actions(state: LSTMState) -> Dict[str, TensorPromise]:
            pass

        def embed_history_symbols(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict[str, TensorPromise]:
            pass

        def combine_symbol_action_embeddings(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict[str, TensorPromise]:
            pass

        def pass_lstm(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> Dict[str, TensorPromise]
            pass

        def pass_out_linear(state: LSTMState, prev_tensor_dict:Dict[str, TensorPromise])

        def calc_prod(state: LSTMState, prev_tensor_dict: Dict[str, TensorPromise]) -> List[TensorPromise]
            pass

        def apply_prod(state: LSTMState, prev_tensor_list: List[TensorPromise]):
            pass


        states = SequentialMonad(states)(
            WhileLogic.While(state_class.is_not_done).Do(
                LogicUnit.If(state_class.is_not_done)
                .Then(embed_history_actions)
                .Then(embed_history_symbols)
                .Then(combine_symbol_action_embeddings)
                .Then(pass_lstm)
                .Then(pass_out_linear)
                .Then(calc_prod)
                .Then(apply_prod)
            )
        ).states

        if golds:
            return LSTMStateGold.combine_loss(states)
        else:
            return LSTMStatePred.get_preds(states)
