from io import BytesIO


import numpy as np
import matplotlib.pyplot as plt

def draw_heat_map(nl_query, weights, title):
    weights = np.array(weights)

    plt.figure(figsize=(18, 15))
    plt.title(title, fontsize=20)
    plt.xticks(np.arange(0.5, len(nl_query), 1), nl_query, rotation=75)
    plt.yticks(np.arange(0.5, len(nl_query), 1), nl_query)
    plt.pcolor(weights)
    plt.colorbar()
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            plt.text(i+0.20, j+0.5, round(weights[j][i], 2))

    # Save as image and Read
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

def draw_inference_score(actions, weights, title):
    assert len(actions) == len(weights), "length different {} {}".format(len(actions), len(weights))
    # Parse
    used_actions = []
    used_weights = []

    for action, weight in zip(actions, weights):
        if weight != float('-inf'):
            used_actions += [action]
            used_weights += [weight]
    used_weights = np.expand_dims(np.array(used_weights), 1)

    # Draw
    plt.figure(figsize=(10, 10))
    plt.title(title, fontsize=20)
    plt.yticks(np.arange(0.5, len(used_actions), 1), used_actions)
    plt.pcolor(used_weights)
    plt.colorbar()
    for i in range(len(used_weights)):
        plt.text(0.40, i+0.5, round(used_weights[i][0], 5))

    # Save as image and read
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return img

if __name__ == "__main__":
    tmp = ["Root -> Sel", "Root -> Sel Filter", "Root -> Sel Filter Super"]
    import torch
    weights = torch.randn(3)
    weights[0] = float('-inf')
    #draw_heat_map(tmp, weights, 0)
    draw_inference_score(tmp, weights, "asdf")
