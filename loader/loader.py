import os
import subprocess

if __name__ == "__main__":
    import argparse
    import sys
    arg_list = sys.argv[1:]
    # arg_list = ['v1'
    # , '--working_dir=.', '--a=b', '--c=d']
    parser = argparse.ArgumentParser(description="Run a SpeechBrain experiment")
    parser.add_argument(
        "model_name",
        type=str,
        default='v1',
        # required=False,
        help="Model name in config file",
    )
    # parser.add_argument(
    #     "--working_dir",
    #     type=str,
    #     # required=False,
    #     help="Set working directory",
    # )
    # Accept extra args to override yaml
    cli_run_opts, cli_overrides = parser.parse_known_args(arg_list)
    # print(cli_run_opts)
    # print(cli_overrides)
    version_name = cli_run_opts.model_name
    def kwarg_list_to_dict(kwargs):
        d = {}
        for arg in kwargs:
            if arg.startswith("--"):
                l = arg[len("--"):].split('=')
                if len(l) > 2:
                    raise Exception(f"Parse kwarg failed: len >= 2 {l}")
                elif len(l) == 2:
                    if l[1].isnumeric():
                        d[l[0]] = int(l[1])
                    else:
                        d[l[0]] = l[1]
                else:
                    d[l[0]] = None
            else:
                raise Exception(f"Unsupported arg: {arg}")
        return d
    cli_overrides = kwarg_list_to_dict(cli_overrides)
    # print(cli_overrides)

    import os
    import json
    import yaml
    from hyperpyyaml import load_hyperpyyaml
    from subprocess import call

    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_file_dir, "./hparams/model_list.yaml")

    with open(config_path) as fin:
        hparams = load_hyperpyyaml(fin, cli_overrides, overrides_must_match=False)

    project_dir = os.path.join(cur_file_dir, hparams['project_dir'])

    # Print config
    print("===================")
    print("FOLDERS:")
    print(f"\tproject_dir: {project_dir}")
    print(f"\tloader_config_path: {config_path}")
    print("===================")
    print("VERSION:")
    print(f"using version: {version_name}")
    model_config_map = hparams['models']
    try:
        model_config = model_config_map[version_name]
    except Exception:
        raise Exception(f"Not found version config. Choose version from: {model_config_map.keys()}")
    print(f"{yaml.dump(model_config, indent=4)}")

    training_file_path = os.path.join(cur_file_dir, model_config['training_file'])
    param_file_path = os.path.join(cur_file_dir, model_config['config_path'])
    overrides = model_config["config_overrides"]
    if cli_overrides is not None:
        print(f"CLI CONFIG OVERRIDES:")
        print(f"{yaml.dump(cli_overrides, indent=4)}")
        overrides = {**cli_overrides, **overrides}

    working_dir = hparams['working_dir']
    # if cli_run_opts.working_dir:
    #     print(f"WORKING_DIR OVERRIDE:")
    #     working_dir = cli_run_opts.working_dir
    #     overrides['working_dir'] = working_dir
    #     print(f"working_dir: {working_dir}")
    # if not os.path.isdir(working_dir):
    #     print(f"Working dir not exists: {working_dir} .. creating")
    #     os.makedirs(working_dir)

    # print("===================")
    # print("DOWNLOAD PREPARED CSV:")
    # import os
    # subprocess.call([
    #     os.path.join(cur_file_dir, './download_prepared_csv.sh'),
    #     overrides["output_folder"]
    # ])

    print("===================")
    print(f"CD working dir: {working_dir}")
    os.chdir(working_dir)
    command = ["python", training_file_path, param_file_path] + [f'--{k}={v}' if v else f'--{k}' for k, v in overrides.items()]
    print(f"RUN: {' '.join(command)}")
    subprocess.call(command)
