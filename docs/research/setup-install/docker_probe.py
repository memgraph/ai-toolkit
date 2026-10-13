"""#535: what Docker says on a busy port, what an assigned port looks like, how long MAGE takes to answer.

Needs Docker and a Python with the neo4j driver. Removes only its own cg-setup-probe container.
Readiness must use an auto-commit query: `execute_query`'s managed transaction makes Memgraph reject SHOW VERSION.
"""

import subprocess
import time

from neo4j import GraphDatabase

NAME = "cg-setup-probe"
IMAGE = "memgraph/memgraph-mage:3.13.1"


def sh(*argv):
    return subprocess.run(argv, capture_output=True, text=True)


print("tag exists:", sh("docker", "manifest", "inspect", IMAGE).returncode == 0)
sh("docker", "rm", "-f", NAME)
busy = sh("docker", "run", "-d", "--name", NAME, "-p", "127.0.0.1:7687:7687", "memgraph/memgraph-mage:latest")
print("busy port exit", busy.returncode, "stderr:", busy.stderr.strip()[-160:])
print("failed container kept:", NAME in sh("docker", "ps", "-a", "--format", "{{.Names}}").stdout.split())
sh("docker", "rm", "-f", NAME)
t = time.monotonic()
ok = sh("docker", "run", "-d", "--name", NAME, "-p", "127.0.0.1::7687", "memgraph/memgraph-mage:latest")
print("assigned port exit", ok.returncode)
port = sh("docker", "port", NAME, "7687/tcp").stdout.strip()
print("docker port:", port)
host_port = int(port.splitlines()[0].rsplit(":", 1)[1])
driver = GraphDatabase.driver(f"bolt://127.0.0.1:{host_port}", auth=("", ""), connection_timeout=2)
while True:
    try:
        with driver.session() as s:
            records = s.run("SHOW VERSION").data()
        print(f"answers SHOW VERSION after {time.monotonic() - t:.1f}s:", records[0])
        break
    except Exception as exc:
        last = type(exc).__name__
        if time.monotonic() - t > 120:
            print("never ready:", last)
            break
        time.sleep(0.5)
driver.close()
sh("docker", "rm", "-f", NAME)
