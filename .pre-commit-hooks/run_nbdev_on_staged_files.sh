for arg in "$@"; do
    nbdev_clean --fname "$arg" # removes only metadata
    # nbdev_clean --clear_all --fname "$arg" # removes outputs and metadata
done