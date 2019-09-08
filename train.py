import json
import torch
import argparse
import datetime
from commons.utils import train, eval, import_module

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, default='', help='Path for train config json file')
    parser.add_argument('--not_save', action='store_true')
    args = parser.parse_args()

    # Load Model
    H_PARAMS = json.loads(open(args.train_config).read())
    Model = import_module(H_PARAMS['model'])
    model = Model(H_PARAMS)

    # Show model name
    model_name = H_PARAMS['model_name'] if 'model_name' in H_PARAMS.keys() else 'Unknown'
    print('Loading model:{}'.format(model_name))

    # Load Dataloader
    DataLoader = import_module(H_PARAMS['dataloader'])
    dataLoader = DataLoader(H_PARAMS)

    # Load data
    load_option = H_PARAMS['load_option'] if 'load_option' in H_PARAMS.keys() else None
    dataLoader.load_data('train', load_option)

    # Prepare Optimizer
    model.optimizer = torch.optim.Adam(model.parameters(), lr=H_PARAMS['lr'], weight_decay=H_PARAMS['weight_decay'])

    # Epoch
    for epoch in range(H_PARAMS['epoch']):
        print('Epoch {} @ {} '.format(epoch + 1, datetime.datetime.now()), end='')

        # Training
        total_loss = train(model, dataLoader)
        print('Loss: {}'.format(total_loss))

        # Evaluating
        if not epoch % H_PARAMS['eval_freq']:
            print('Evaluating...', end='')
            total_acc = eval(model, dataLoader)
            if not args.not_save:
                # Save model if high acc
                model.save_weights(total_acc)

