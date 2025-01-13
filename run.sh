cd /mnt/workspace
cp -r ./sqlcoder/* /root/anaconda3/envs/sqlcoder_env/lib/python3.10/site-packages/sqlcoder/

cd ./sqlcoder
cp -r ./sqlcoder/* /usr/local/lib/python3.9/dist-packages/sqlcoder

sqlcoder launch
