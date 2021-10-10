from utils.getter import *
import argparse
import os

parser = argparse.ArgumentParser('Evaluate classification')
parser.add_argument('--config', type=str, default="./configs/configs.yaml", help='Config file')
parser.add_argument('--weight', type=str, help='checkpoint file')


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
  
    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=trainset.num_classes)

    if config.task == 'T1':
        metric_type = 'multiclass'
    else:
        metric_type = 'multilabel'


    metric = [
        AccuracyMetric(types=metric_type),
        F1ScoreMetric(types=metric_type)
    ]

    if config.task == 'T1':
        metric.extend([
            BalancedAccuracyMetric(num_classes=trainset.num_classes), 
            ConfusionMatrix(trainset.classes), 
        ])

    criterion = get_loss(config.loss_fn)

    model = Classifier(
            model = net,
            metrics=metric, 
            criterion=criterion,
            device = device)

    if args.weight is not None:                
        load_checkpoint(model, args.weight)

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     visualize_when_val = False)

    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s)')
    print(devices_info)
  
    trainer.evaluate_epoch()

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = Config(args.config)

    train(args, config)
    

