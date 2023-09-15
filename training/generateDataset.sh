# wget https://secure.nic.cz/files/knot-resolver/knot-resolver-release.deb
# dpkg -i knot-resolver-release.deb
# apt update
# apt install -y knot-resolver
# sh -c 'echo `hostname -I` `hostname` >> /etc/hosts'
# sh -c 'echo nameserver 127.0.0.1 > /etc/resolv.conf'
# systemctl stop systemd-resolved

# echo "Packages installled \n\n"

# systemctl start kresd@1.service
# systemctl start kresd@2.service
# systemctl start kresd@3.service
# systemctl start kresd@4.service

# dig @localhost google.com

img2dataset --url_list $1 --input_format "tsv" --url_col "URL" --caption_col "top_caption" --output_format webdataset --output_folder $2 --processes_count 32 --thread_count 256 --image_size 256 --enable_wandb False --number_sample_per_shard 10000 --incremental_mode "incremental"
