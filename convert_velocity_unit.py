import click
import json
import pathlib

OLD_UNIT = 0.01414213562373095
NEW_UNIT = 0.0282842712474619


@click.command()
@click.argument("input", type=click.Path(exists=True), required=True)
def main(input):
    input_dir = pathlib.Path(input)
    for json_file in input_dir.glob("*.json"):
        fix_file(
            json_file.as_posix()
        )


def fix_file(file_path):
    with open(file_path, "rt") as fd:
        data = json.load(fd)
    new_x_velocity_distribution = {}
    if (
        max([float(velocity) for velocity in data["x_velocity_distribution"].keys()])
        > 100
    ):
        print(f"Skipping: {file_path} max velocity > 100")
        return

    for velocity, times in data["x_velocity_distribution"].items():
        new_velocity = round(float(velocity) * 2)
        if str(new_velocity) in new_x_velocity_distribution:
            new_x_velocity_distribution[str(new_velocity)] += times
        else:
            new_x_velocity_distribution[str(new_velocity)] = times

    data["x_velocity_distribution"] = new_x_velocity_distribution

    with open(file_path, "wt") as fd:
        json.dump(data, fd, indent=4)


if __name__ == "__main__":
    main()
