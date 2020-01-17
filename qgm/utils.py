import torch
from qgm.ops import WHERE_OPS, BOX_OPS, AGG_OPS, IUEN, IUE
IUE_INDICES = [BOX_OPS.index(key) for key in IUE]
GROUP_IDX = BOX_OPS.index('groupBy')
ORDER_IDX = BOX_OPS.index('orderBy')


def get_limit_num(boxes):
    nums = [box['limit_num'] for box in boxes]
    return nums


def get_is_asc(boxes):
    is_asc = [int(box['is_asc']) for box in boxes]
    return is_asc


def to_tensor(item):
    return torch.tensor(item).unsqueeze(-1).cuda()


def get_head_agg_with_idx(n_idx, boxes):
    aggs = [box['head'][n_idx][0] for box in boxes]
    return aggs


def get_head_col(boxes):
    cols = []
    for b_idx in range(len(boxes)):
        tmp = [head[1] for head in boxes[b_idx]['head']]
        cols += [tmp]
    return cols


def get_head_num(boxes):
    nums = [len(box['head'])-1 for box in boxes]
    return nums


def get_local_predicate_op_with_idx(n_idx, boxes):
    ops = [box['body']['local_predicates'][n_idx][2] for box in boxes]
    return ops


def get_local_predicate_col_with_idx(n_idx, boxes):
    cols = [box['body']['local_predicates'][n_idx][1] for box in boxes]
    return cols


def get_local_predicate_agg_with_idx(n_idx, boxes):
    aggs = [box['body']['local_predicates'][n_idx][0] for box in boxes]
    return aggs


def get_local_predicate_num(boxes):
    nums = [len(box['body']['local_predicates']) if box['body']['local_predicates'] else 0 for box in boxes]
    return nums


def get_quantifier_with_type(type, boxes):
    '''
    may need to changed for type == 's'
    '''
    quantifiers = []
    for b_idx in range(len(boxes)):
        tmp = [boxes[b_idx]['body']['quantifiers'][idx] for idx in range(len(boxes[b_idx]['body']['quantifiers']))
                       if boxes[b_idx]['body']['quantifier_types'][idx] == type]
        quantifiers += [tmp]
    return quantifiers


def get_quantifier_num_with_type(type, boxes):
    nums = []
    for b_idx in range(len(boxes)):
        tmp = len([item for item in boxes[b_idx]['body']['quantifier_types'] if item == type])
        nums += [tmp]
    return nums


def get_box_with_op_type(op_type, boxes):
    group_box = []
    for b_idx in range(len(boxes)):
        tmp = [boxes[b_idx][idx] for idx in range(len(boxes[b_idx])) if boxes[b_idx][idx]['operator'] == BOX_OPS.index(op_type)]
        if tmp:
            assert len(tmp) == 1
            group_box += [tmp[0]]
    return group_box


def get_is_box_info(boxes):
    is_group_by = []
    is_order_by = []
    for b_idx in range(len(boxes)):
        ops = [box['operator'] for box in boxes[b_idx]]
        is_group_by += [int(GROUP_IDX in ops)]
        is_order_by += [int(ORDER_IDX in ops)]
    return is_group_by, is_order_by


def split_boxes(boxes):
    front_boxes = []
    rear_boxes = []
    for b_idx in range(len(boxes)):
        split_idx = [idx for idx, box in enumerate(boxes[b_idx]) if box['operator'] in IUE_INDICES]
        if split_idx:
            assert len(split_idx) == 1
            front_boxes += [boxes[b_idx][:split_idx[0]]]
            rear_boxes += [boxes[b_idx][split_idx[0]:]]
        else:
            front_boxes += [boxes[b_idx]]
    return front_boxes, rear_boxes


def get_iue_box_op(boxes):
    ops = []
    for b_idx in range(len(boxes)):
        tmp = [BOX_OPS[box['operator']] for box in boxes[b_idx] if box['operator'] if box['operator'] in IUE_INDICES]
        if tmp:
            assert len(tmp) == 1
            ops += [tmp[0]]
        else:
            ops += [[]]
    return ops




def compare_boxes(pred_qgm, gold_qgm):
    b_size = len(gold_qgm)

    total_acc = {}
    for b_idx in range(b_size):
        acc = {}
        pred_boxes = pred_qgm[b_idx]
        gold_boxes = gold_qgm[b_idx]

        # Check is group by
        pred_is_group_by = GROUP_IDX in [box['operator'] for box in pred_boxes]
        gold_is_group_by = GROUP_IDX in [box['operator'] for box in gold_boxes]
        acc['is_group_by'] = pred_is_group_by == gold_is_group_by

        # Check is order by
        pred_is_order_by = ORDER_IDX in [box['operator'] for box in pred_boxes]
        gold_is_order_by = ORDER_IDX in [box['operator'] for box in gold_boxes]
        acc['is_order_by'] = pred_is_order_by == gold_is_order_by

        # Check select box:
        # Compare head - num
        pred_head_num = len(pred_boxes[0]['head'])
        gold_head_num = len(gold_boxes[0]['head'])
        acc['head_num'] = gold_head_num == pred_head_num

        # Compare head - idx
        pass

        # Compare body - quantifier num
        pred_quantifier_num = len(pred_boxes[0]['body']['quantifier_types'])
        gold_quantifier_num = len(gold_boxes[0]['body']['quantifier_types'])
        acc['quantifier_num'] = pred_quantifier_num == gold_quantifier_num

        # Compare body - quantifiers
        pass

        # Compare body - local predicates
        pass

        # Compare operator
        pred_operators = [box['operator'] for box in pred_boxes if box['operator'] in IUE_INDICES]
        gold_operators = [box['operator'] for box in gold_boxes if box['operator'] in IUE_INDICES]
        if pred_operators and gold_operators:
            acc['operator'] = pred_operators == gold_operators
        else:
            acc['operator'] = len(pred_operators) == len(gold_operators)

        # Compare nested box
        pass

        # If this example is correct
        acc['total'] = True
        for key in acc.keys():
            acc['total'] = acc['total'] and acc[key]

        if total_acc:
            for key in total_acc.keys():
                total_acc[key] += acc[key]
        else:
            total_acc = acc

    return total_acc
