
MAIN_CONFIG = {
    'trainer_module' : 'trainer.classifier',
    'trainer' : 'ClassifierTrainer',
    'classifier_module' : 'model.classifier',
    'classifier' : 'classifier',
    'package_path' : '/nas2/annahung/project/anna_jam_v2/'
}





MODEL_CONFIG = {
'classifier': {
    'hidden_m': 256,
    'bidirectional' : 2,
    'num_layers_en' : 2
}
}

TRAIN_CONFIG = {
'batch_size' : 256, 
'epochs' : 200,
'lr' : 1e-5,
'lr_step1' : 10,
'lr_step2' : 30,
'lr_gamma' : 1e-1
    
}