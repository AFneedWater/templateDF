"""Stage 06 reconstruct command; see templatedf.inference for the Python API."""


def main():
    from .inference_cli import main as run
    run('reconstruct')


if __name__ == '__main__':
    main()
