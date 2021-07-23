# Notes for submitting jobs with PhillyTool(Amulet)

The main function is located in `src/train.py`

The yaml file of Amulet is located in `./transformer_base.yaml`

## 1. Config PhillyTool(Amulet)

config `project_name`, `storage_account_name`, `local_path`, and `default_output_dir `in `.ptconfig` or `.amltconfig`

## 2. Config wandb

Currently, we are monitoring jobs with https://wandb.ai/home. If you don't use it, send an argument `--wandb_offline` to `train.py`.

To login wandb on remote clusters, a json file `src/wandb_config.json` should be added and configed.

```json
{
  "WANDB_API_KEY": "apikey",
  "WANDB_ENTITY": "username", 
  "WANDB_PROJECT": "projectname"
}
```

## 3. Data preparation

run `.get_data.sh`, which would create a folder named data and download, unzip, and re-name files of wikitext-103 to this folder.

Then upload data to azure with pt/amlt command `pt/amlt upload --config-file transformer_base.yaml`

