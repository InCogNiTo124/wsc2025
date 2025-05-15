packer {
  required_plugins {
    exoscale = {
      version = ">= 0.5.1"
      source = "github.com/exoscale/exoscale"
    }
  }
}

variable "api_key" { default = "" }
variable "api_secret" { default = "" }
variable "nvidia_driver_version" { default = "570" }
variable "triton_docker_version" { default = "06" }
variable "jj_version" { default = "v0.30.0" }

source "exoscale" "base" {
  api_key = var.api_key
  api_secret = var.api_secret
  instance_template = "Linux Ubuntu 24.04 LTS 64-bit"
  instance_disk_size = 100
  instance_type = "d9a835e6-7981-48fc-a268-ca43ced32997"
  template_zones = ["at-vie-2"]
  template_name = "wsc2025"
  template_username = "ubuntu"
  ssh_username = "ubuntu"
}

build {
  sources = ["source.exoscale.base"]
  provisioner "shell" {
    environment_vars = ["DEBIAN_FRONTEND=noninteractive"]
    inline = [
      "set -eux",

      # setting up docker registry
      "sudo apt install -y ca-certificates curl wget",
      "sudo install -m 0755 -d /etc/apt/keyrings",
      "sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc",
      "sudo chmod a+r /etc/apt/keyrings/docker.asc", "echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"$${UBUNTU_CODENAME:-$VERSION_CODENAME}\") stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null",

      # setting up nvidia-container-toolkit registry
      "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
      "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",

      # update upgrade
      "sudo apt update",
      "sudo apt upgrade -y",

      # setup nvidia, docker, and tensorrt-llm deps
      "sudo apt install --no-install-recommends -y nvidia-utils-${var.nvidia_driver_version}-server nvidia-driver-${var.nvidia_driver_version}-server nvidia-modprobe docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit python3-dev libopenmpi-dev nvidia-cuda-toolkit",
      "sudo nvidia-ctk runtime configure --runtime=docker",
      
      # setup tooling: uv, jj, just
      "curl -LsSf https://astral.sh/uv/install.sh | sh",
      "wget -O /tmp/jj.tar.gz https://github.com/jj-vcs/jj/releases/download/${var.jj_version}/jj-${var.jj_version}-x86_64-unknown-linux-musl.tar.gz && sudo tar -C /usr/local/bin -xzvf /tmp/jj.tar.gz ./jj",
      "curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sudo bash -s -- --to /usr/bin/",
      "sudo bash -c '/usr/bin/just --completions bash > /usr/share/bash-completion/completions/just'",

      # download triton server to save time
      "sudo docker image pull nvcr.io/nvidia/tritonserver:25.${var.triton_docker_version}-trtllm-python-py3",

      "sudo usermod -aG docker ubuntu",

      "sudo su ubuntu -c 'mkdir /home/ubuntu/checkpoints'"
    ]
  }

  provisioner "file" {
    source = "checkpoints/google"
    destination = "/home/ubuntu/checkpoints/"
  }
}

