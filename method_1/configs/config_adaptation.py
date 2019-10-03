
MAIN_CONFIG = {
    'model_module' : 'model.vae',
    'model' : 'VAE_jazz_only_2',
    'pertrain_model' : '', #a model train on TT
    'trainer_module' : 'trainer.vae',
    'trainer' : 'VAE_jazz_only_Trainer',
    'package_path' : '/nas2/annahung/project/anna_jam_v2/',

}


MODEL_CONFIG = {

'vae': {
    'encoder' : {
        'hidden_m': 256, 
        'direction' : True,
        'num_of_layer' : 1,
        'gru_dropout_en' : None
        },

    'decoder' : {
        'direction' : False,
        'num_of_layer' : 1,
        'gru_dropout_de' : None,
        'teacher_forcing_ratio' : 1.0

        }

},
'classifier': {
    'hidden_m': 256,
    'bidirectional' : 2,
    'num_layers_en' : 2
}
    
}


TRAIN_CONFIG = {
'batch_size' : 256,
'epochs' : 200,
'vae' : {
    'lr_vae' : 0.001,
    'lr_step1':40,
    'lr_step2':60,
    'lr_gamma':1,
    'loss_beta' : 0.7,

}


}













