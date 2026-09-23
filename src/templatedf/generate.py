"""Stage 06 generate command; see templatedf.inference for the Python API."""


def main():
    from .inference_cli import main as run
    run('generate')


if __name__ == '__main__':
    main()
