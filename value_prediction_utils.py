def is_number_tryexcept(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def find_values(question_arg, question_arg_type, question_origin, mapper):
    values = []
    flag_double_q = False
    flag_double_q_for_schema = False
    cur_val = []
    flag_single_q = False
    flag_single_q_for_schema = False
    flag_upper = False
    cur_upper_val = []
    for idx, (token, tag) in enumerate(zip(question_arg, question_arg_type)):
        if idx == 0:
            continue
        start_idx = mapper[idx][0]
        end_idx = mapper[idx][1]
        if len(token) == 0:
            continue
        if flag_double_q:
            if '"' not in token[0]:
                cur_val.append(" ".join(question_origin[start_idx:end_idx]))
                if tag[0] in ("table", "col"):
                    flag_double_q_for_schema = True
                continue
        if flag_single_q:
            if "'" not in token[0]:
                #                      for i, t in enumerate(token):
                #                          idx = first_substring( question_origin[start_idx:end_idx], t )
                #                          if idx != -1:
                #                              token[i]=question_origin[idx]
                cur_val.append(" ".join(question_origin[start_idx:end_idx]))
                if tag[0] in ("table", "col"):
                    flag_single_q_for_schema = True
                continue

        if flag_upper:
            # If Jason 'Two ... separate
            if (
                len(question_origin[start_idx]) > 0
                and question_origin[start_idx][0].isupper()
                and tag[0] not in ("col", "table")
            ):
                cur_upper_val.append(" ".join(question_origin[start_idx:end_idx]))
                continue
            else:
                values.append(" ".join(cur_upper_val))
                cur_upper_val = []
                flag_upper = False

        def is_year(tok):
            if (
                len(str(tok)) == 4
                and str(tok).isdigit()
                and 15 < int(str(tok)[:2]) < 22
            ):
                return True

        is_inserted_already = False
        if (
            len(token) == 1
            and is_year(token[0])
            and is_number_tryexcept(question_origin[start_idx])
        ):
            is_inserted_already = True
            values.append(question_origin[start_idx])

        if '"' in token[0]:
            if flag_double_q:
                is_inserted_already = True
                flag_double_q = False
                if not flag_double_q_for_schema:
                    values.append(" ".join(cur_val))
                cur_val = []
                flag_double_q_for_schema = False
            elif len(token[0]) == 1:
                is_inserted_already = True
                flag_double_q = True
        elif "'" in token[0]:
            if flag_single_q:
                is_inserted_already = True
                flag_single_q = False
                if not flag_single_q_for_schema:
                    values.append(" ".join(cur_val))
                cur_val = []
                flag_single_q_for_schema = False
            elif len(token[0]) == 1:
                is_inserted_already = True
                flag_single_q = True

        if (
            (not is_inserted_already)
            and len(question_origin[start_idx]) > 0
            and question_origin[start_idx][0].isupper()
            and start_idx != 0
        ):
            if tag[0] not in ("col", "table"):
                is_inserted_already = True
                flag_upper = True
                cur_upper_val.append(" ".join(question_origin[start_idx:end_idx]))
        if (
            (not is_inserted_already)
            and tag[0] in ("value", "*", "db")
            and token[0] != "the"
        ):
            is_inserted_already = True
            values.append(" ".join(question_origin[start_idx:end_idx]))
    return values


def exchange_values(sql, sql_values, value_tok):
    sql = sql.replace(value_tok, " 1")
    cur_index = sql.find(" 1")
    sql_with_value = ""
    before_index = 0
    values_index = 0
    while cur_index != -1 and values_index < len(sql_values):
        sql_with_value = sql_with_value + sql[before_index:cur_index]
        if sql[cur_index - 1] in ("=", ">", "<"):
            cur_value = sql_values[values_index]
            values_index = values_index + 1
            if not is_number_tryexcept(cur_value):
                cur_value = '"' + cur_value + '"'
            sql_with_value = sql_with_value + " " + cur_value
        elif cur_index - 3 > 0 and sql[cur_index - 4 : cur_index] in ("like"):
            cur_value = "%" + sql_values[values_index] + "%"
            values_index = values_index + 1
            if not is_number_tryexcept(cur_value):
                cur_value = '"' + cur_value + '"'
            sql_with_value = sql_with_value + " " + cur_value
        elif cur_index - 6 > 0 and sql[cur_index - 7 : cur_index] in ("between"):
            if values_index + 1 < len(sql_values):
                cur_value1 = sql_values[values_index]
                values_index = values_index + 1
                cur_value2 = sql_values[values_index]
                values_index = values_index + 1
            else:
                cur_value1 = sql_values[values_index]
                cur_value2 = sql_values[values_index]
                values_index = values_index + 1
            if not is_number_tryexcept(cur_value1):
                cur_value1 = "1"
            if not is_number_tryexcept(cur_value2):
                cur_value2 = "2"
            sql_with_value = sql_with_value + " " + cur_value1 + " AND " + cur_value2
            cur_index = cur_index + 6
        else:
            sql_with_value = sql_with_value + sql[cur_index : cur_index + 2]
        before_index = cur_index + 2
        cur_index = sql.find(" 1", cur_index + 1)
    sql_with_value = sql_with_value + sql[before_index:]
    print(sql_with_value)
    return sql_with_value
