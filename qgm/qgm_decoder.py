def qmg_decoder(encoded_nl, encoded_context):
    '''
    1. predict box operator
        1-1. predict operator

    2: predict box - body - quantifiers
        2-1. predict num
        2-2. predict type
        2-3. if type == 's':
            recursive call
           if type == 'f':
            choose from DB table

    3: predict box - body - predicates
        3-1. predict local predicates
        3-1-1. predict num
        3-1-2. predict type
        3-1-3. predict agg, col, operator

        3-2. predict join predicates
        3-2-1. if more than one quantifiers, predict join column

    4: predict box - head
        4-1. predict num
        4-2. predict agg
        4-3. predict col



    :param encoded_nl:
    :param encoded_context:
    :return: [qgm box]
    '''



    return None