import argparse
import json
import os
import shlex
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def load_configurations():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def validate_configured_model_paths(configurations):
    models = configurations.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("config.json must contain a non-empty 'models' mapping")

    absolute_paths = []
    missing_paths = []
    for name, path in models.items():
        if os.path.isabs(path):
            absolute_paths.append(f"{name}: {path}")
        elif not os.path.isfile(path):
            missing_paths.append(f"{name}: {path}")

    if absolute_paths:
        raise ValueError(
            "Docker deployments require model paths in config.json to be relative "
            f"to deployment_module: {', '.join(absolute_paths)}"
        )

    if missing_paths:
        raise FileNotFoundError(
            f"Configured model artifact(s) are missing: {', '.join(missing_paths)}"
        )

    return models


def validate_port(port):
    port = str(port)
    if not port.isdigit():
        raise ValueError(f"Port must be numeric, got {port!r}")

    port_number = int(port)
    if port_number < 1 or port_number > 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {port!r}")

    return port


def write_requirements():
    with open("accelera_deployment/requirements.txt", "w", encoding="utf-8") as req:
        req.write("fastapi==0.136.1\n")
        req.write("uvicorn[standard]==0.46.0\n")
        req.write("scikit-learn==1.8.0\n")
        req.write("category-encoders==2.9.0\n")
        req.write("numpy==2.4.4\n")
        req.write("pandas==3.0.2\n")
        req.write("pydantic==2.13.3\n")
        req.write("python-multipart==0.0.27\n")


def write_dockerfile(configurations):
    models = validate_configured_model_paths(configurations)

    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write("FROM python:3.11-slim\n")
        f.write("WORKDIR /app\n")
        f.write("COPY accelera_deployment/requirements.txt requirements.txt \n")
        f.write(
            "RUN python -m pip install --no-cache-dir --prefer-binary "
            "--timeout 120 --retries 10 -r requirements.txt\n"
        )
        f.write("COPY accelera_deployment/server.py server.py\n")
        f.write("COPY accelera_deployment/modelservice.py modelservice.py\n")
        f.write("COPY config.json config.json\n")
        for pkl in models.values():
            f.write(f"COPY {pkl} /app/{pkl}\n")
        f.write("EXPOSE 8000\n")
        f.write(
            'CMD ["sh", "-c", '
            '"uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]\n'
        )
    print("Dockerfile written successfully")


def prepare(_args):
    configurations = load_configurations()
    write_requirements()
    write_dockerfile(configurations)


def _docker_build_command(image_name, no_cache=False):
    command = ["docker", "build", "-t", image_name, "."]
    if no_cache:
        command.insert(2, "--no-cache")
    return command


def build(args):
    subprocess.run(
        _docker_build_command("ml-model", no_cache=getattr(args, "no_cache", False)),
        check=True,
    )


def run_local(_args):
    port = validate_port(os.environ.get("PORT", "8000"))
    running = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"publish={port}"],
        check=True,
        capture_output=True,
        text=True,
    )
    container_ids = running.stdout.split()
    if container_ids:
        subprocess.run(["docker", "stop", *container_ids], check=True)

    print("\n--- Starting container  ---\n")
    print(f" API: http://localhost:{port}")
    subprocess.run(
        ["docker", "run", "--rm", "-p", f"{port}:{port}", "-e", f"PORT={port}", "ml-model"],
        check=True,
    )


def local(_args):
    prepare(_args)
    build(_args)
    run_local(_args)


def heroku_login(_args):
    subprocess.run(["heroku", "login"], check=True)


def heroku_create(args):
    subprocess.run(["heroku", "create", args.app, "--stack", "container"], check=True)


def heroku_container_login(_args):
    subprocess.run(["heroku", "container:login"], check=True)


def heroku_push(args):
    prepare(args)
    subprocess.run(["heroku", "container:push", "web", "--app", args.app], check=True)


def heroku_release(args):
    subprocess.run(["heroku", "container:release", "web", "--app", args.app], check=True)


def heroku_open(args):
    subprocess.run(["heroku", "open", "--app", args.app], check=True)


def heroku_deploy(args):
    heroku_login(args)
    if args.create:
        heroku_create(args)
    heroku_container_login(args)
    heroku_push(args)
    heroku_release(args)
    heroku_open(args)


def ec2_deploy(args):
    prepare(args)

    target = _remote_target(args)
    script = _remote_script(args)
    remote_root = _remote_root(args)

    print(f"Creating remote deployment directory at {remote_root}...")
    _run_remote(args, f"mkdir -p {_quote_remote_path(remote_root)}")

    rsync_target = f"{target}:{args.remote_dir.rstrip('/')}/deployment_module"
    rsync_sources = ["Dockerfile", "config.json", "accelera_deployment", "models"]
    print("Syncing build inputs to EC2...")
    subprocess.run(
        [
            "rsync",
            "-av",
            "--info=progress2",
            "--delete",
            "-e",
            _ssh_transport(args),
            *rsync_sources,
            rsync_target,
        ],
        check=True,
    )

    print("Building and starting the Docker container on EC2...")
    _run_remote(args, script)


def ec2_stop(args):
    container_name = args.container

    cmd = f"sudo docker stop {shlex.quote(container_name)} || true"
    _run_remote(args, cmd)
    print(f"Stopped container '{container_name}' on {args.host}")


def ec2_logs(args):
    container_name = args.container

    cmd = f"sudo docker logs -f {shlex.quote(container_name)}"
    _run_remote(args, cmd)


def _ssh_command(args):
    command = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if args.key:
        command.extend(["-i", os.path.expanduser(args.key)])
    return command


def _ssh_transport(args):
    return " ".join(shlex.quote(part) for part in _ssh_command(args))


def _run_remote(args, command):
    remote_command = f"bash -lc {shlex.quote(command)}"
    subprocess.run([*_ssh_command(args), _remote_target(args), remote_command], check=True)


def _remote_target(args):
    return f"{args.user}@{args.host}"


def _remote_root(args):
    return f"{args.remote_dir.rstrip('/')}/deployment_module"


def _quote_remote_path(path):
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def _remote_script(args):
    remote_root = _remote_root(args)

    port = validate_port(args.port)
    image_name = args.image
    container_name = args.container
    docker_build = "sudo " + " ".join(
        shlex.quote(part)
        for part in _docker_build_command(
            image_name,
            no_cache=getattr(args, "no_cache", False),
        )
    )

    install_docker = ""
    if args.install_docker:
        install_docker = """
if ! command -v docker >/dev/null 2>&1; then
  echo "Installing Docker..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y docker
  else
    echo "No supported package manager found to install Docker" >&2
    exit 1
  fi
  sudo systemctl enable docker
  sudo systemctl start docker
  echo "Docker installed and started"
fi
"""

    docker_check = ""
    if not args.install_docker:
        docker_check = """
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed on the EC2 host. Use --install-docker to install it automatically." >&2
  exit 1
fi
"""

    return f"""
set -e
cd {_quote_remote_path(remote_root)}
{install_docker}{docker_check}
{docker_build}
sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true
sudo docker run -d --restart unless-stopped \
  --name {shlex.quote(container_name)} \
  -p {port}:{port} \
  -e PORT={port} \
  {shlex.quote(image_name)}
echo "Waiting for container health endpoint..."
for attempt in 1 2 3 4 5; do
  if sudo docker exec {shlex.quote(container_name)} python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=3).read()" >/dev/null 2>&1; then
    echo "Local container health check: OK"
    break
  fi
  if [ "$attempt" = "5" ]; then
    echo "Local container health check failed; inspect logs with ec2-logs." >&2
    exit 1
  fi
  sleep 2
done
sudo docker ps --filter name={shlex.quote(container_name)} --format 'container={{{{.Names}}}} image={{{{.Image}}}} status={{{{.Status}}}} ports={{{{.Ports}}}}'
echo "Application URL: http://{args.host}:{port}"
""".strip()



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python accelera_deployment/deployment.py local
  python accelera_deployment/deployment.py heroku-deploy --app accelera1 --create
  python accelera_deployment/deployment.py heroku-push --app accelera1
    python accelera_deployment/deployment.py ec2-deploy --host 1.2.3.4 --user ec2-user --key ~/.ssh/key.pem
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="available commands")
    subparsers.add_parser("prepare", help="write requirements and Dockerfile")
    build_parser = subparsers.add_parser("build", help="build the Docker image")
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers instead of using the cache",
    )

    subparsers.add_parser("run-local", help="run the local Docker container")

    local_parser = subparsers.add_parser("local", help="prepare, build, and run locally")
    local_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers instead of using the cache",
    )

    subparsers.add_parser("heroku-login", help="run Heroku login")

    create_parser = subparsers.add_parser("heroku-create", help="create Heroku app")
    create_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    subparsers.add_parser("heroku-container-login", help="login to Heroku container registry")

    push_parser = subparsers.add_parser("heroku-push", help="push Docker image to Heroku")
    push_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    release_parser = subparsers.add_parser(
        "heroku-release", help="release web container on Heroku"
    )
    release_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    open_parser = subparsers.add_parser("heroku-open", help="open Heroku app")
    open_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    deploy_parser = subparsers.add_parser(
        "heroku-deploy", help="run full Heroku deployment sequence"
    )
    deploy_parser.add_argument("--app", default="accelera1", help="Heroku app name")
    deploy_parser.add_argument(
        "--create",
        action="store_true",
        help="create the app before pushing if needed",
    )

    ec2_parser = subparsers.add_parser(
        "ec2-deploy", help="sync the deployment module to EC2, build the image, and run the container"
    )
    ec2_parser.add_argument("--host", required=True, help="EC2 public IP or DNS")
    ec2_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_parser.add_argument("--key", help="SSH private key file path")
    ec2_parser.add_argument(
        "--remote-dir",
        default="~/deployment-app",
        help="directory to sync the deployment module into on EC2",
    )
    ec2_parser.add_argument("--port", default="8000", help="public port to expose on EC2")
    ec2_parser.add_argument("--image", default="ml-model", help="Docker image name to build")
    ec2_parser.add_argument(
        "--container",
        default="ml-model",
        help="Docker container name to run on EC2",
    )
    ec2_parser.add_argument(
        "--install-docker",
        action="store_true",
        help="attempt to install and start Docker on the EC2 host if it is missing",
    )
    ec2_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers on EC2 instead of using the cache",
    )

    ec2_stop_parser = subparsers.add_parser("ec2-stop", help="stop the EC2 container")
    ec2_stop_parser.add_argument("--host", required=True, help="EC2 public IP or DNS")
    ec2_stop_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_stop_parser.add_argument("--key", help="SSH private key file path")
    ec2_stop_parser.add_argument("--container", default="ml-model", help="Docker container name")

    ec2_logs_parser = subparsers.add_parser("ec2-logs", help="tail logs from the EC2 container")
    ec2_logs_parser.add_argument("--host", required=True, help="EC2 public IP or DNS")
    ec2_logs_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_logs_parser.add_argument("--key", help="SSH private key file path")
    ec2_logs_parser.add_argument("--container", default="ml-model", help="Docker container name")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "prepare": prepare,
        "build": build,
        "run-local": run_local,
        "local": local,
        "heroku-login": heroku_login,
        "heroku-create": heroku_create,
        "heroku-container-login": heroku_container_login,
        "heroku-push": heroku_push,
        "heroku-release": heroku_release,
        "heroku-open": heroku_open,
        "heroku-deploy": heroku_deploy,
        "ec2-deploy": ec2_deploy,
        "ec2-stop": ec2_stop,
        "ec2-logs": ec2_logs,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
