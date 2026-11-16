#!/usr/bin/env bash
# Run interactively with sudo on the online Ubuntu build host.
# Sources: Docker Ubuntu apt instructions; NVIDIA Container Toolkit install guide.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run: sudo bash scripts/install_docker_host.sh' >&2; exit 1; }
source /etc/os-release
[[ $ID == ubuntu ]] || { echo 'This script targets Ubuntu.' >&2; exit 1; }
WALL_APT_PARTS="$(mktemp -d /tmp/walloss-apt.XXXXXX)"
chmod 755 "$WALL_APT_PARTS"
trap 'rm -rf "$WALL_APT_PARTS"' EXIT
cat > "$WALL_APT_PARTS/ubuntu.sources" <<EOF
Types: deb
URIs: https://archive.ubuntu.com/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME} ${UBUNTU_CODENAME:-$VERSION_CODENAME}-updates ${UBUNTU_CODENAME:-$VERSION_CODENAME}-security
Components: main universe restricted multiverse
Architectures: $(dpkg --print-architecture)
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
WALL_APT_OPTS=(-o Dir::Etc::sourcelist=/dev/null -o "Dir::Etc::sourceparts=$WALL_APT_PARTS"
  -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30
  -o Acquire::Languages=none -o Acquire::IndexTargets::deb::DEP-11::DefaultEnabled=false)
for package in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc; do
  status="$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)"
  if [[ $status == 'install ok installed' ]]; then
    echo "Conflicting package $package is installed; review existing container services first." >&2
    exit 1
  fi
done
if [[ -n ${WALL_DOCKER_DEBS:-} || -n ${WALL_NVIDIA_DEBS:-} ]]; then
  : "${WALL_DOCKER_DEBS:?Set both verified package directories}"
  : "${WALL_NVIDIA_DEBS:?Set both verified package directories}"
  for package_dir in "$WALL_DOCKER_DEBS" "$WALL_NVIDIA_DEBS"; do
    test -f "$package_dir/SHA256SUMS"
    (cd "$package_dir" && sha256sum -c SHA256SUMS)
  done
  # All required system dependencies must already be present. No online fallback.
  apt-get --simulate --no-install-recommends install \
    "$WALL_DOCKER_DEBS"/*.deb "$WALL_NVIDIA_DEBS"/*.deb
  dpkg --install \
    "$WALL_DOCKER_DEBS"/*.deb "$WALL_NVIDIA_DEBS"/*.deb
else
apt-get "${WALL_APT_OPTS[@]}" update
apt-get "${WALL_APT_OPTS[@]}" install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl --retry 3 -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
ln -s /etc/apt/sources.list.d/docker.sources "$WALL_APT_PARTS/docker.sources"
apt-get "${WALL_APT_OPTS[@]}" update
apt-get "${WALL_APT_OPTS[@]}" install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
ln -s /etc/apt/sources.list.d/nvidia-container-toolkit.list "$WALL_APT_PARTS/nvidia-container-toolkit.list"
apt-get "${WALL_APT_OPTS[@]}" update
apt-get "${WALL_APT_OPTS[@]}" install -y nvidia-container-toolkit
fi
if [[ -f /etc/docker/daemon.json ]]; then
  cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.before-walloss-$(date -u +%Y%m%dT%H%M%SZ)"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker version
if [[ -n ${WALL_GPU_TEST_ARCHIVE:-} ]]; then
  : "${WALL_GPU_TEST_IMAGE:?Set the tag or ID from the local GPU test archive}"
  docker load -i "$WALL_GPU_TEST_ARCHIVE"
  docker run --rm --pull=never --network none --gpus device=0 "$WALL_GPU_TEST_IMAGE" nvidia-smi
else
  docker run --rm --gpus device=0 nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
fi
# NVIDIA driver and user group membership are not changed.
