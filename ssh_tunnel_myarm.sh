#/usr/bin/ssh -C -N -g -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -R 8450:127.0.0.1:8450 myarm
ps -ef | grep 'uvicorn.*845[0]' | awk '{print $2}' | xargs -J % kill -9 %
ps -ef | grep 'ssh.*8450.*8450.*myar[m]' | awk '{print $2}' | xargs -J % kill -9 %

source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8450 &
/usr/bin/ssh -C -f -N -g -R 127.0.0.1:8450:127.0.0.1:8450 myarm
