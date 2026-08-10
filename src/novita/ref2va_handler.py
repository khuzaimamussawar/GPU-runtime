from src.common.h3_runtime import run_h3_job
from src.common.novita_server import serve


def handler(payload):
    return run_h3_job(payload, "h3_ref2va", "novita-ref2va")


if __name__ == "__main__":
    serve(handler, "novita-ref2va")
