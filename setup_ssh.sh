mkdir -p /root/.ssh
chmod 700 /root/.ssh/

if [[ "$1" == "primary" ]]; then
    # Having empty for passphrase is not a good practice
    ssh-keygen -t rsa -f /root/.ssh/id_rsa -q -N "";
else
    # Add public key - id_rsa.pub from primary worker to this file
    touch /root/.ssh/authorized_keys;
    chmod 600 /root/.ssh/authorized_keys;
fi

