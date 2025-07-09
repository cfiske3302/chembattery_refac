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

Use `run_script_list_parallel.sh` to run several config files at the same time. Currently, the output will not look pretty
since all processes will print to the same console. (TODO is to fix)
Takes one argument: the path to a text file containing a list of paths to config files. For example:

```bash
./run_script_list_parallel.sh ./configs/crate_tests/paths_to_run.txt
```

Use `run_script_list_parallel_limited.sh` to run several config files at the same time, with a max number at the same time. Currently, the output will not look pretty since all processes will print to the same console. (TODO is to fix)
Takes two arguments: the path to a text file containing a list of paths to config files and a maximum number of jobs. For example:

```bash
./run_script_list_parallel_limited.sh ./configs/crate_tests/paths_to_run.txt 25
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
