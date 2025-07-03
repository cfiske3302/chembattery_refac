## Entrypoint

`main.py`

## Arguments

| Flag | Description |
| ---- | ----------- |
| `--config` | Path to a config file |
| `--train`  | Train the model |
| `--eval`   | Evaluate the model. If given with `--train`, will train then evaluate the model |

## Configuration

See `configs/example_config.yaml` for capabilities and formatting.

## Running Multiple Config Files

Use `run_script_list.sh` to run several config files in order.  
Takes one argument: the path to a text file containing a list of paths to config files. For example:

```bash
./run_script_list.sh ./configs/crate_tests/paths_to_run.txt
```

---

## Stuff to do

- add better documentation
- add fine tuning stuff
- generalize ensemble models better
- make script to easily test several configs
- make script to easily make configs
- let ensembling train models in parallel on multiple GPUs
- add code to load a model from a directory without know what it is ahead of time (look at saved config)
