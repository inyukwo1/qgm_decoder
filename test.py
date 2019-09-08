import json
import argparse
from commons.utils import eval, test, import_module

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_config', type=str, default='', help='Model Name')
    parser.add_argument('--output_path', type=str, default='./out.sql', help='Path to save predicted SQL')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    # Load Model
    H_PARAMS = json.loads(open(args.train_config))
    Model = import_module(H_PARAMS['model'])
    model = Model(H_PARAMS)

    # Show model name
    model_name = H_PARAMS['model_name'] if 'model_name' in H_PARAMS.keys() else 'Unknown'
    print('Loading model:{}'.format(model_name))

    # Load Saved weights
    model.load_weights()

    # Load Dataloader
    DataLoader = import_module(H_PARAMS['dataloader'])
    dataLoader = DataLoader(H_PARAMS)

    # Load data
    load_option = H_PARAMS['load_option'] if 'load_option' in H_PARAMS.keys() else None
    dataLoader.load_data('test', load_option)

    '''
    TO-DO
    Generalize below
    '''
    if model_name == 'syntaxsql':
        model.set_test_table_dict(dataLoader.schemas, dataLoader.test_data)
        assert dataLoader.batch_size == 1

    # Testing
    if model_name == 'frompredictor':
        acc = eval(model, dataLoader, log=args.log)
        print('Average Acc:{}'.format(acc))
    else:
        test(model, dataLoader, args.output_path)

    print('Predicted SQL output is written at {}'.format(args.output_path))
