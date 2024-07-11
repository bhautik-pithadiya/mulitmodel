check_swap() {
    if free | awk '/^Swap:/ {exit !$2}'; then
        echo "Swap space already exists."
        return 1
    else
        echo "No swap space found. Creating swap space..."
        return 0
    fi
}

swapon() {
    command sudo fallocate -l 20G /swapfile
    command sudo chmod 600 /swapfile
    command sudo mkswap /swapfile
    command sudo swapon /swapfile
    command echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

}


check_swap
swapon