import wandb
import yaml

with open(r'/home/zcemg08/Scratch/Mol_gan2/config_files/GAN_param_grid.yaml') as file:
    sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep_config, entity="zcemg08", project="gan_molecular2")

text_file = open('/home/zcemg08/Scratch/Mol_gan2/id.txt', "w")
text_file.write(sweep_id)
text_file.close()
