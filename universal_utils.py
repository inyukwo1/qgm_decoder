import requests


def download_file_from_google_drive(id, destination):
    print("Download start")
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    print("Confirmed")
    save_response_content(response, destination)
    print("Download done")


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def captum_vis_to_html(datarecords):
    def format_classname(classname):
        return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(
            classname
        )

    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Target Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    return "".join(dom)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        trimmed_words = [w for w in word if not w.startswith("[")]
        word = format_special_tokens(" ".join(trimmed_words))
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def src_attribution_to_html(sentence, attribution):
    return format_word_importances(sentence, attribution)


def tab_col_attribution_to_html(example, attribution_col, attribution_tab):
    dom = ['<table style="border: 1px solid #333333; border-collapse: collapse;">']
    table_dict = [[] for _ in range(example.table_col_len)]
    for col_idx in example.col_table_dict:
        if col_idx > 0:
            for tab_idx in example.col_table_dict[col_idx]:
                table_dict[tab_idx].append(col_idx)

    dom += ['<tr style="border: 1px solid #333333; border-collapse: collapse; ">']
    for tab_idx, table_name in enumerate(example.table_names):
        dom += [
            '<td colspan={}  style="border: 1px solid #333333; border-collapse: collapse;">'.format(
                len(table_dict[tab_idx])
            )
        ]

        word = format_special_tokens(" ".join(table_name))
        color = _get_color(attribution_tab[tab_idx])

        dom += [
            '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
                color=color, word=word
            )
        ]

        dom += ["</td>"]
    dom += ["</tr>"]
    dom += ['<tr style="border: 1px solid #333333; border-collapse: collapse;">']
    for tab_idx, col_indices in enumerate(table_dict):
        for col_idx in col_indices:
            dom += [
                '<td style="border: 1px solid #333333; border-collapse: collapse;">'
            ]

            word = format_special_tokens(" ".join(example.tab_cols[col_idx]))
            color = _get_color(attribution_col[col_idx])
            dom += [
                '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                    color=color, word=word
                )
            ]
            dom += ["</td>"]
    dom += ["</tr>"]

    dom += ["</table>"]
    return "".join(dom)
