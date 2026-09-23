"""Stage 06 export_latents command; see templatedf.inference for the Python API."""


def main():
    from .inference_cli import main as run
    run('export_latents')


if __name__ == '__main__':
    main()
