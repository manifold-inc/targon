#### Docker Installation

1. **Set up Docker's apt repository**

   ```bash
   # Add Docker's official GPG key
   sudo apt-get update
   sudo apt-get install ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg

   # Add the repository to Apt sources
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

1. **Install Docker Engine**

   ```bash
   # Update the package index
   sudo apt-get update

   # Install Docker Engine, containerd, and Docker Compose
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

1. **Verify Installation**

   ```bash
   # Verify Docker Engine installation
   sudo docker run hello-world
   ```

1. **Post-installation Steps**

   ```bash
   # Create the docker group
   sudo groupadd docker

   # Add your user to the docker group
   sudo usermod -aG docker $USER

   # Activate the changes to groups
   newgrp docker
   ```

#### Docker Compose Installation

Docker Compose is included in the Docker Engine installation above. Verify it's
installed:

```bash
docker compose version
```

