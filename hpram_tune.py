from utils.tools import *
from utils.training_tools import *
import optuna
from train import train
from utils.logger import Logger


def read_sweep(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def objective(trial, train_args, sweep_args, train_data, val_data):
    params = {}
    for param_name, param_spec in sweep_args['parameters'].items():
        if param_spec['distribution'] == 'log_uniform_values':
            params[param_name] = trial.suggest_float(param_name, param_spec['min'], param_spec['max'], log=True)
        elif param_spec['distribution'] == 'int_uniform':
            params[param_name] = trial.suggest_int(param_name, param_spec['min'], param_spec['max'])
        elif 'values' in param_spec:
            params[param_name] = trial.suggest_categorical(param_name, param_spec['values'])
        train_args[param_name] = params[param_name]
    train_args['SESSIONAME'] = f'Trial{trial.number + 1}'
    logger = Logger(train_args)
    model = get_model(train_args)
    best_loss = train(model, train_data, val_data, train_args, logger)
    logger.add_hyperparams(params, {sweep_args['metric']['name']: best_loss})
    logger.close()
    return best_loss

def main(train_args, sweep_args):
    train_dataset, val_dataset = get_dataset(train_args)
    direction = 'maximize' if sweep_args['metric']['goal'] == 'maximize' else 'minimize'
    os.makedirs('outputs/studies', exist_ok=True)
    study = optuna.create_study(f"sqlite:///outputs/studies/{train_args['PROJECTNAME']}.db", direction=direction, study_name=f'{train_args["SESSIONAME"]}', load_if_exists=True)
    study.optimize(lambda trial: objective(trial, train_args, sweep_args, train_dataset, val_dataset), n_trials=100)

if __name__ == '__main__':
    train_args = parse_args()
    set_reproducibility(train_args['SEED'])
    sweep_args = read_sweep('./config/sweep_config.yaml')
    main(train_args, sweep_args)

