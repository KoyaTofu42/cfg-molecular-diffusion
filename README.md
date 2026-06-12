# commands

- start run
```shell
docker compose up -d
```

- end run
```shell
docker compose down
```

- resume run
```shell
docker compose run -d --rm diffusion python main.py --resume outputs/cfg_multi_property --wandb_run_id <run_id>
```