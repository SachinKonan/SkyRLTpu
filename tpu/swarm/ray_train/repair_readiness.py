"""Apply only the wrapped-log readiness fix to a running, named host actor."""
import argparse
import inspect
import textwrap

import ray

from .host import Host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    source = textwrap.dedent(inspect.getsource(Host.trainer_ready))

    def repair(host, expected, replacement):
        if host.config.run_id != expected or host.rank != 0 or host.role != "trainer":
            raise RuntimeError("refusing readiness repair on an unexpected actor")
        process = host.processes.get("trainer")
        if host.stopping.is_set() or not process or process.poll() is not None:
            raise RuntimeError("trainer is not running; readiness repair does not apply")
        namespace = {}
        exec(replacement, namespace)
        ready = namespace["trainer_ready"]
        if not ready(host):
            raise RuntimeError("current trainer log does not prove initialization")
        # Ray caches the original function in its method executor. Replacing
        # the class attribute alone would leave remote calls using old code.
        for cls in type(host).__mro__:
            method = cls.__dict__.get("trainer_ready")
            if method is not None:
                if method.__code__.co_freevars or ready.__code__.co_freevars:
                    raise RuntimeError("unexpected readiness closure")
                method.__code__ = ready.__code__
        return dict(run_id=expected, rank=host.rank, trainer_ready=True,
                    trainer_guardian_pid=process.process.pid)

    ray.init(address=args.address, namespace=args.run_id, log_to_driver=False)
    try:
        actor = ray.get_actor("host-0", namespace=args.run_id)
        print(ray.get(actor.__ray_call__.remote(repair, args.run_id, source), timeout=30), flush=True)
        if not ray.get(actor.trainer_ready.remote(), timeout=30):
            raise RuntimeError("remote readiness still false after repair")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
