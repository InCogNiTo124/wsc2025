# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "exoscale>=0.11.0",
#     "yaspin>=3.1.0"
# ]
# ///

from exoscale.api.v2 import Client

import base64
import os
from datetime import datetime
import hashlib
import time
import pwd

from urllib import request

from yaspin import yaspin
import webbrowser


def b64(s: str):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8") if len(s) > 0 else None


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def get_name():
    username = get_username()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    suffix = hashlib.md5(now.encode("utf-8")).hexdigest()
    return f"{username}-{suffix}"


API_KEY = "EXO28e18b2c0225b593297a9676"
API_SECRET = "QwdVcO8q-Mh4jJgVd_6Twj_wU6EsSvvy1tdwUWbMcdw"
ZONE = "at-vie-2"
TEMPLATE_ID = "ad711e80-f9e3-40e0-8508-0ca49c15c68e"
USER_DATA = r"""#cloud-config
write_files:
  - path: /etc/systemd/system/jupyterlab.service
    permissions: '0644'
    owner: ubuntu
    content: |
      [Unit]
      Description=JupyterLab
      After=network.target

      [Service]
      # https://jupyter-server.readthedocs.io/en/latest/operators/public-server.html#preparing-a-hashed-password
      # this also handles syncing the environment
      ExecStart=/home/ubuntu/.local/bin/uv --project=/home/ubuntu/wsc2025 run jupyter lab --ip=0.0.0.0 --port 8888 --PasswordIdentityProvider.hashed_password='argon2:$argon2id$v=19$m=10240,t=10,p=8$DvuzIlZ+9aso5Ro685iixA$jXk+JdcdSlRSpLqXrv3qR6os4EP1VuCys1Fg3UMyOyc'
      WorkingDirectory=/home/ubuntu/wsc2025
      Restart=always
      Type=simple
      User=ubuntu

      [Install]
      WantedBy=multi-user.target

runcmd:
  - 'su ubuntu -c "jj git clone https://github.com/InCogNiTo124/wsc2025 /home/ubuntu/wsc2025"'
  - 'su ubuntu -c "jj -R /home/ubuntu/wsc2025 new workshop@origin"'
  - 'su ubuntu -c "mkdir -p /home/ubuntu/wsc2025/checkpoints/google"'
  - 'su ubuntu -c "mv /home/ubuntu/checkpoints/google/gemma-3-1b-it /home/ubuntu/wsc2025/checkpoints/google/"'
  - rm -r /home/ubuntu/checkpoints
  - just --completions bash > /usr/share/bash-completion/completions/just
  - systemctl daemon-reload
  - systemctl enable jupyterlab.service
  - systemctl start jupyterlab.service
"""

client = Client(API_KEY, API_SECRET, zone=ZONE)

instance_types_list = client.list_instance_types()
instance_type = next(
    t
    for t in instance_types_list["instance-types"]
    if "a5000" in t.get("family") and t.get("size") == "small"
)

ssh_key = client.list_ssh_keys()["ssh-keys"][0]

operation = client.create_instance(
    name=get_name(),
    instance_type=instance_type,
    template={"id": TEMPLATE_ID},
    disk_size=200,
    ssh_key=ssh_key,
    user_data=b64(USER_DATA),
)

with yaspin(text="Creating instance") as sp:
    while True:
        time.sleep(5)
        op = client.get_operation(id=operation["id"])
        if op["state"] == "success":
            reference = op["reference"]
            created_instance = client.get_instance(id=reference["id"])
            public_ip = created_instance["public-ip"]
            break
    sp.ok("✅")

with yaspin(text="Starting instance") as sp:
    while True:
        time.sleep(5)
        try:
            response = request.urlopen(f"http://{public_ip}:8888")
        except:
            continue
        break
    sp.ok("✅")

print(f"Visit http://{public_ip}:8888 and enter 'iminlovewiththellms'")
webbrowser.open_new_tab(f"http://{public_ip}:8888")
